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
  Rune_pjrt.Triton.Kernel.create ~name:"raven_add_one" ~ir:ttir ~num_warps:1
    ~num_stages:1 ()

let fallback x = Nx.add x (Nx.scalar_like x 1.0)

let add_one x =
  Rune_pjrt.Triton.call kernel ~inputs:[ Rune_pjrt.Triton.Tensor x ]
    ~fallback:(fun () -> fallback x)

module D = Rune_pjrt.Triton.Dsl

let define_unary_kernel (type a) ~name (dtype : a D.Dtype.t)
    ?(config = D.Config.make ()) build =
  let block_size = D.Config.block_size config in
  D.Kernel.define ~name
    ~signature:D.Signature.(dtype @-> returning dtype)
    ~config
    ~guard:(fun spec -> D.Spec.input_numel spec 0 = D.Spec.output_numel spec)
    ~grid:(fun spec ->
      let numel = D.Spec.output_numel spec in
      ((numel + block_size - 1) / block_size, 1, 1))
    (fun spec input output ->
      let offsets =
        D.Value.add
          (D.Value.mul (D.Value.program_id D.X)
             (D.Value.int D.Dtype.i32 block_size))
          (D.Value.arange ~start:0 ~stop:block_size)
      in
      let mask =
        D.Value.less offsets
          (D.Value.int D.Dtype.i32 (D.Spec.output_numel spec))
      in
      let values =
        D.Pointer.load ~mask
          ~other:(D.Value.zeros dtype ~shape:[| block_size |])
          (D.Pointer.offset input offsets)
      in
      [
        D.Statement.store ~mask (D.Pointer.offset output offsets) (build values);
      ])

let dsl_kernel =
  let config = D.Config.make ~block_size:128 ~num_warps:4 () in
  define_unary_kernel ~name:"raven_square_plus_one" D.Dtype.f32 ~config
    (fun input ->
      let open D.Syntax in
      (input * input) + D.Value.float D.Dtype.f32 1.0)

let dsl_f16_kernel =
  define_unary_kernel ~name:"raven_f16_add" D.Dtype.f16 (fun input ->
      D.Value.add input (D.Value.float D.Dtype.f16 1.))

let dsl_bf16_kernel =
  define_unary_kernel ~name:"raven_bf16_add" D.Dtype.bf16 (fun input ->
      D.Value.add input (D.Value.float D.Dtype.bf16 1.))

let general_abs_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  let config = D.Config.make ~block_size:128 ~num_warps:4 () in
  D.Kernel.define ~name:"raven_general_abs"
    ~signature:D.Signature.(f32 @-> returning f32)
    ~config
    ~guard:(fun spec -> D.Spec.input_numel spec 0 = D.Spec.output_numel spec)
    ~grid:(fun spec ->
      let numel = D.Spec.output_numel spec in
      ((numel + 127) / 128, 1, 1))
    (fun spec input output ->
      let program = D.Value.program_id D.X in
      let block = D.Value.int D.Dtype.i32 128 in
      let offsets =
        D.Value.add
          (D.Value.mul program block)
          (D.Value.arange ~start:0 ~stop:128)
      in
      let bound = D.Value.int D.Dtype.i32 (D.Spec.output_numel spec) in
      let mask = D.Value.less offsets bound in
      let zero = D.Value.zeros D.Dtype.f32 ~shape:[| 128 |] in
      let values =
        D.Pointer.load ~mask ~other:zero (D.Pointer.offset input offsets)
      in
      let output_values =
        D.Value.where
          (D.Value.less values (D.Value.float D.Dtype.f32 0.))
          (D.Value.neg values) values
      in
      [
        D.Statement.store ~mask (D.Pointer.offset output offsets) output_values;
      ])

let reduce_sum_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  let config = D.Config.make ~block_size:128 ~num_warps:4 () in
  D.Kernel.define ~name:"raven_reduce_sum"
    ~signature:D.Signature.(f32 @-> returning f32)
    ~config
    ~guard:(fun spec ->
      let input = D.Spec.input_shape spec 0 in
      Array.length input = 2
      && input.(1) = 128
      && input.(0) = D.Spec.output_numel spec)
    ~grid:(fun spec -> (D.Spec.output_numel spec, 1, 1))
    (fun spec input output ->
      let rows = D.Spec.output_numel spec in
      let row = D.Value.program_id D.X in
      let row_start = D.Value.mul row (D.Value.int D.Dtype.i32 128) in
      let offsets = D.Value.add row_start (D.Value.arange ~start:0 ~stop:128) in
      let values = D.Pointer.load (D.Pointer.offset input offsets) in
      [
        D.Statement.static_assert (rows > 0) "row count must be positive";
        D.Statement.store
          (D.Pointer.offset output row)
          (D.Value.sum ~axis:0 values);
      ])

let dot_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  let config = D.Config.make ~block_size:256 ~num_warps:4 () in
  D.Kernel.define ~name:"raven_dot_16"
    ~signature:D.Signature.(f16 @-> f16 @-> returning f32)
    ~config
    ~guard:(fun spec ->
      D.Spec.input_shape spec 0 = [| 16; 16 |]
      && D.Spec.input_shape spec 1 = [| 16; 16 |]
      && D.Spec.output_shape spec = [| 16; 16 |])
    ~grid:(fun _ -> (1, 1, 1))
    (fun spec lhs rhs output ->
      let offsets = D.Value.arange ~start:0 ~stop:256 in
      let lhs =
        D.Pointer.load (D.Pointer.offset lhs offsets)
        |> D.Value.reshape ~shape:[| 16; 16 |]
      in
      let rhs =
        D.Pointer.load (D.Pointer.offset rhs offsets)
        |> D.Value.reshape ~shape:[| 16; 16 |]
      in
      let accumulator = D.Value.zeros D.Dtype.f32 ~shape:[| 16; 16 |] in
      let result =
        D.Value.dot lhs rhs accumulator |> D.Value.reshape ~shape:[| 256 |]
      in
      [
        D.Statement.static_assert
          (D.Spec.input_count spec = 2)
          "expected two inputs";
        D.Statement.store (D.Pointer.offset output offsets) result;
      ])

let dsl_square_plus_one =
  D.Kernel.bind dsl_kernel ~fallback:(fun x ->
      Nx.add (Nx.mul x x) (Nx.scalar_like x 1.0))

let require message condition =
  if not condition then failwith ("test_triton: " ^ message)

let contains text pattern =
  let text_length = String.length text in
  let pattern_length = String.length pattern in
  let rec loop offset =
    if offset + pattern_length > text_length then false
    else if String.sub text offset pattern_length = pattern then true
    else loop (offset + 1)
  in
  loop 0

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
      Rune_pjrt.Triton.Kernel.create ~name:"kernel" ~ir:ttir ~grid:(0, 1, 1) ())

let test_dsl_validation () =
  require_invalid (fun () ->
      D.Kernel.define ~name:"invalid-name"
        ~signature:D.Signature.(f32 @-> returning f32)
        ~grid:(fun spec -> (D.Spec.output_numel spec, 1, 1))
        (fun spec input output ->
          let value = D.Pointer.load input in
          [
            D.Statement.static_assert
              (D.Spec.input_count spec = 1)
              "expected one input";
            D.Statement.store output value;
          ])
      |> ignore);
  require_invalid (fun () -> D.Config.make ~block_size:96 () |> ignore);
  require_invalid (fun () -> D.Value.float D.Dtype.f32 Float.nan |> ignore);
  require_invalid (fun () -> D.Value.arange ~start:0 ~stop:96 |> ignore);
  let captured = ref None in
  let capture_kernel =
    D.Kernel.define ~name:"capture"
      ~signature:D.Signature.(f32 @-> returning f32)
      ~grid:(fun spec -> (D.Spec.output_numel spec, 1, 1))
      (fun spec input output ->
        let value = D.Pointer.load input in
        captured := Some value;
        [
          D.Statement.static_assert
            (D.Spec.input_count spec = 1)
            "expected one input";
          D.Statement.store output value;
        ])
  in
  D.Kernel.to_ttir_for capture_kernel ~input_shapes:[ [| 1 |] ]
    ~output_shape:[| 1 |]
  |> ignore;
  let captured =
    match !captured with
    | Some expression -> expression
    | None -> failwith "test_triton: failed to capture DSL expression"
  in
  let stale_kernel =
    D.Kernel.define ~name:"stale_input"
      ~signature:D.Signature.(f32 @-> returning f32)
      ~grid:(fun spec -> (D.Spec.output_numel spec, 1, 1))
      (fun spec input output ->
        let current = D.Pointer.load input in
        [
          D.Statement.static_assert
            (D.Spec.input_count spec = 1)
            "expected one input";
          D.Statement.store output (D.Value.add current captured);
        ])
  in
  require_invalid (fun () ->
      D.Kernel.to_ttir_for stale_kernel ~input_shapes:[ [| 1 |] ]
        ~output_shape:[| 1 |]
      |> ignore)

let test_dsl_rendering () =
  let rendered =
    D.Kernel.to_ttir_for dsl_kernel ~input_shapes:[ [| 257 |] ]
      ~output_shape:[| 257 |]
  in
  require "DSL TTIR did not contain a blocked range"
    (contains rendered "tt.make_range {end = 128 : i32, start = 0 : i32}");
  require "DSL TTIR did not contain a tail mask"
    (contains rendered "arith.cmpi slt");
  require "DSL TTIR did not contain the fused multiply"
    (contains rendered "arith.mulf");
  require "DSL TTIR did not contain a masked store"
    (contains rendered "tt.store");
  require "DSL TTIR did not specialize numel"
    (contains rendered "arith.constant 257 : i32");
  let f16 =
    D.Kernel.to_ttir_for dsl_f16_kernel ~input_shapes:[ [| 32 |] ]
      ~output_shape:[| 32 |]
  in
  let bf16 =
    D.Kernel.to_ttir_for dsl_bf16_kernel ~input_shapes:[ [| 32 |] ]
      ~output_shape:[| 32 |]
  in
  require "DSL TTIR did not preserve f16 pointer types"
    (contains f16 "!tt.ptr<f16, 1>");
  require "DSL TTIR did not preserve bf16 pointer types"
    (contains bf16 "!tt.ptr<bf16, 1>");
  let general =
    Rune_pjrt.Triton.Dsl.Kernel.to_ttir_for general_abs_kernel
      ~input_shapes:[ [| 257 |] ] ~output_shape:[| 257 |]
  in
  require "general DSL TTIR did not use explicit pointer arithmetic"
    (contains general "tt.addptr");
  require "general DSL TTIR did not lower selection"
    (contains general "arith.select");
  let reduction =
    Rune_pjrt.Triton.Dsl.Kernel.to_ttir_for reduce_sum_kernel
      ~input_shapes:[ [| 4; 128 |] ]
      ~output_shape:[| 4 |]
  in
  require "general DSL TTIR did not lower a reduction"
    (contains reduction "\"tt.reduce\"");
  require "general DSL TTIR did not terminate the reduction region"
    (contains reduction "tt.reduce.return");
  let dot =
    Rune_pjrt.Triton.Dsl.Kernel.to_ttir_for dot_kernel
      ~input_shapes:[ [| 16; 16 |]; [| 16; 16 |] ]
      ~output_shape:[| 16; 16 |]
  in
  require "general DSL TTIR did not lower block dot" (contains dot "tt.dot");
  require "general DSL TTIR did not preserve mixed f16/f32 ABI"
    (contains dot "!tt.ptr<f16, 1>" && contains dot "!tt.ptr<f32, 1>");
  let module D = Rune_pjrt.Triton.Dsl in
  let loop_kernel =
    D.Kernel.define ~name:"raven_loop"
      ~signature:D.Signature.(f32 @-> returning f32)
      ~grid:(fun spec ->
        if D.Spec.input_count spec = 1 then (1, 1, 1)
        else invalid_arg "test_triton: expected one loop input")
      (fun spec input output ->
        let sum =
          D.Value.range ~start:(D.Value.int D.Dtype.i32 0)
            ~stop:(D.Value.int D.Dtype.i32 128)
            ~init:(D.Value.float D.Dtype.f32 0.) (fun index accumulator ->
              D.Value.add accumulator
                (D.Pointer.load (D.Pointer.offset input index)))
        in
        [
          D.Statement.static_assert
            (D.Spec.input_count spec = 1)
            "expected one input";
          D.Statement.store output sum;
        ])
  in
  let loop =
    D.Kernel.to_ttir_for loop_kernel ~input_shapes:[ [| 128 |] ]
      ~output_shape:[||]
  in
  require "general DSL TTIR did not lower a device loop"
    (contains loop "scf.for");
  require "general DSL TTIR did not lower a loop-carried value"
    (contains loop "iter_args")

let test_fallback () =
  let x = Nx.scalar Nx.float32 2.0 in
  require "eager execution did not use the fallback"
    (Nx.to_array (add_one x) = [| 3.0 |]);
  let gradient = Rune.grad (fun value -> Nx.sum (add_one value)) x in
  require "automatic differentiation did not use the fallback"
    (Nx.to_array gradient = [| 1.0 |]);
  let values = Nx.create Nx.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  require "DSL eager execution did not use the fallback"
    (Nx.to_array (dsl_square_plus_one values) = [| 2.0; 5.0; 10.0 |]);
  let dsl_gradient =
    Rune.grad
      (fun value -> Nx.sum (dsl_square_plus_one value))
      (Nx.scalar Nx.float32 2.0)
  in
  require "DSL automatic differentiation did not use the fallback"
    (Nx.to_array dsl_gradient = [| 4.0 |]);
  let guarded_values = Nx.create Nx.float32 [| 2 |] [| 3.; 4. |] in
  let guarded_function =
    D.Kernel.bind reduce_sum_kernel ~fallback:(fun values -> values)
  in
  let guarded = guarded_function guarded_values in
  require "DSL shape guard did not return the fallback"
    (Nx.to_array guarded = [| 3.; 4. |]);
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

let test_dsl_trace () =
  let x = Nx.create Nx.float32 [| 257 |] (Array.make 257 2.0) in
  let capture = Rune_pjrt.Trace.capture_one dsl_square_plus_one x in
  require "DSL trace did not contain the generated Triton call"
    (List.exists
       (fun node ->
         Rune_pjrt.Ir.op_name node.Rune_pjrt.Ir.op
         = "triton_call[raven_square_plus_one]")
       capture.program.nodes)

let () =
  test_validation ();
  test_dsl_validation ();
  test_dsl_rendering ();
  test_fallback ();
  test_trace ();
  test_dsl_trace ()
