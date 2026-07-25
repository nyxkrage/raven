(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let ttir =
  {|
module {
  tt.func public @raven_add_one(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>) {
    %value = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    %one = arith.constant 1.000000e+00 : f32
    %result = arith.addf %value, %one : f32
    tt.store %arg1, %result {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
    tt.return
  }
}
|}

let kernel =
  Rune_pjrt.Triton.Kernel.create ~name:"raven_add_one" ~ir:ttir
    ~num_warps:1 ~num_stages:1 ()

let fallback x = Nx.add x (Nx.scalar_like x 1.0)

let add_one x =
  Rune_pjrt.Triton.call kernel
    ~inputs:[ Rune_pjrt.Triton.Tensor x ]
    ~fallback:(fun () -> fallback x)

let require message condition =
  if not condition then failwith ("test_triton: " ^ message)

let require_invalid thunk =
  match thunk () with
  | exception Invalid_argument _ -> ()
  | _ -> failwith "test_triton: expected Invalid_argument"

let test_validation () =
  require_invalid (fun () ->
      Rune_pjrt.Triton.Kernel.create ~name:"" ~ir:ttir ());
  require_invalid (fun () ->
      Rune_pjrt.Triton.Kernel.create ~name:"kernel" ~ir:"" ());
  require_invalid (fun () ->
      Rune_pjrt.Triton.Kernel.create ~name:"kernel" ~ir:ttir ~num_warps:3 ());
  require_invalid (fun () ->
      Rune_pjrt.Triton.Kernel.create ~name:"kernel" ~ir:ttir
        ~grid:(0, 1, 1) ())

let test_fallback () =
  let x = Nx.scalar Nx.float32 2.0 in
  require "eager execution did not use the fallback"
    (Nx.to_array (add_one x) = [| 3.0 |]);
  let gradient = Rune.grad (fun value -> Nx.sum (add_one value)) x in
  require "automatic differentiation did not use the fallback"
    (Nx.to_array gradient = [| 1.0 |]);
  let capture = Rune_pjrt.Trace.capture_one ~enable_ffi:false add_one x in
  require "disabled custom kernels did not trace the fallback"
    (List.exists
       (fun node -> Rune_pjrt.Ir.op_name node.Rune_pjrt.Ir.op = "add")
       capture.program.nodes)

let test_trace () =
  let x = Nx.scalar Nx.float32 2.0 in
  let capture = Rune_pjrt.Trace.capture_one add_one x in
  require "PJRT CUDA trace did not contain a Triton call"
    (List.exists
       (fun node ->
         Rune_pjrt.Ir.op_name node.Rune_pjrt.Ir.op
         = "triton_call[raven_add_one]")
       capture.program.nodes);
  let module_text = Rune_pjrt.Stablehlo.of_program capture.program in
  require "StableHLO did not target XLA's Triton custom call"
    (String.starts_with ~prefix:"__gpu$xla.gpu.triton"
       (match
          String.split_on_char '"' module_text
          |> List.find_opt (String.starts_with ~prefix:"__gpu$xla.gpu.triton")
        with
       | Some target -> target
       | None -> ""))

let () =
  test_validation ();
  test_fallback ();
  test_trace ()
