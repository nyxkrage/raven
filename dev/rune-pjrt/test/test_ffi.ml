(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let failf fmt = Printf.ksprintf failwith fmt
let require msg condition = if not condition then failf "test_ffi: %s" msg

let contains_substring text substring =
  let text_length = String.length text in
  let substring_length = String.length substring in
  let rec loop offset =
    if offset + substring_length > text_length then false
    else if String.sub text offset substring_length = substring then true
    else loop (offset + 1)
  in
  loop 0

let require_values msg expected actual =
  let expected = Nx.to_array expected in
  let actual = Nx.to_array actual in
  if expected <> actual then failf "test_ffi: %s: values differ" msg

let kernel ?fwd ?bwd () =
  Rune_pjrt.Ffi.Kernel.create ~library:"/proc/self/exe" ?fwd ?bwd ()

let square_with kernel x =
  Rune_pjrt.Ffi.call_fwd kernel ~inputs:[ Rune_pjrt.Ffi.Tensor x ]
    ~fallback:(fun () -> Nx.mul x x)

let contains_op name program =
  List.exists
    (fun node -> Rune_pjrt.Ir.op_name node.Rune_pjrt.Ir.op = name)
    program.Rune_pjrt.Ir.nodes

let contains_custom_call program =
  List.exists
    (fun node ->
      match node.Rune_pjrt.Ir.op with
      | Rune_pjrt.Ir.Custom_call _ -> true
      | _ -> false)
    program.Rune_pjrt.Ir.nodes

let test_eager_fallback () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let expected = Nx.create Nx.float32 [| 3 |] [| 1.; 4.; 9. |] in
  require_values "eager forward fallback" expected
    (square_with (kernel ~fwd:"square_fwd" ()) x)

let test_cuda_trace_dispatch () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let capture =
    Rune_pjrt.Trace.capture_one (square_with (kernel ~fwd:"square_fwd" ())) x
  in
  require "CUDA trace records a custom call"
    (contains_custom_call capture.program);
  require "CUDA trace omits fallback operations"
    (not (contains_op "mul" capture.program));
  let stablehlo = Rune_pjrt.Stablehlo.of_program capture.program in
  require "custom-call operand layout is canonical"
    (contains_substring stablehlo
       "operand_layouts = [dense<[0]> : tensor<1xindex>]");
  require "custom-call result layout is canonical"
    (contains_substring stablehlo
       "result_layouts = [dense<[0]> : tensor<1xindex>]");
  match Rune_pjrt.Ir.ffi_handlers capture.program with
  | [ handler ] ->
      require "handler library"
        (handler.library = Unix.realpath "/proc/self/exe");
      require "handler symbol" (handler.symbol = "square_fwd");
      require "content-addressed target prefix"
        (String.starts_with ~prefix:"raven_cuda_" handler.target)
  | handlers ->
      failf "test_ffi: expected one handler, got %d" (List.length handlers)

let test_cpu_trace_fallback () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let capture =
    Rune_pjrt.Trace.capture_one ~enable_ffi:false
      (square_with (kernel ~fwd:"square_fwd" ()))
      x
  in
  require "CPU trace records fallback operations"
    (contains_op "mul" capture.program);
  require "CPU trace omits custom calls"
    (not (contains_custom_call capture.program))

let test_missing_direction_fallback () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let kernel = kernel ~fwd:"square_fwd" () in
  let capture =
    Rune_pjrt.Trace.capture_one
      (fun x ->
        Rune_pjrt.Ffi.call_bwd kernel ~inputs:[ Rune_pjrt.Ffi.Tensor x ]
          ~fallback:(fun () -> Nx.mul x x))
      x
  in
  require "missing backward handler traces fallback"
    (contains_op "mul" capture.program);
  require "missing backward handler emits no custom call"
    (not (contains_custom_call capture.program))

let test_backward_operand_order () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let kernel = kernel ~bwd:"square_bwd" () in
  let capture =
    Rune_pjrt.Trace.capture_one
      (fun x ->
        let y = Nx.add_s x 1.0 in
        let dy = Nx.mul_s x 2.0 in
        Rune_pjrt.Ffi.call_bwd kernel
          ~inputs:
            [
              Rune_pjrt.Ffi.Tensor x;
              Rune_pjrt.Ffi.Tensor y;
              Rune_pjrt.Ffi.Tensor dy;
            ] ~fallback:(fun () -> Nx.mul x x))
      x
  in
  match
    List.find_map
      (fun node ->
        match node.Rune_pjrt.Ir.op with
        | Rune_pjrt.Ir.Custom_call { inputs; _ } -> Some inputs
        | _ -> None)
      capture.program.nodes
  with
  | Some inputs ->
      let names =
        List.map
          (fun input ->
            capture.program.nodes
            |> List.find (fun (node : Rune_pjrt.Ir.node) -> node.id = input)
            |> fun node -> Rune_pjrt.Ir.op_name node.op)
          inputs
      in
      require "backward preserves x, y, dy operand order"
        (names = [ "parameter[0]"; "add"; "mul" ])
  | None -> failf "test_ffi: missing backward custom call"

let test_transforms_use_fallback () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let ones = Nx.ones_like x in
  let square = square_with (kernel ~fwd:"square_fwd" ()) in
  let y, dy = Rune.jvp square x ones in
  require_values "JVP primal" (Nx.mul x x) y;
  require_values "JVP tangent" (Nx.mul_s x 2.) dy;
  let _, dx = Rune.vjp square x ones in
  require_values "VJP cotangent" (Nx.mul_s x 2.) dx;
  let batched = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  require_values "vmap" (Nx.mul batched batched) (Rune.vmap square batched)

let test_content_addressed_target () =
  let path = Filename.temp_file "raven-ffi" ".so" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      let write contents =
        let channel = open_out_bin path in
        Fun.protect
          ~finally:(fun () -> close_out channel)
          (fun () -> output_string channel contents)
      in
      write "first";
      let handler : Rune_pjrt.Ffi.Internal.handler =
        { library = path; symbol = "handler" }
      in
      let first = Rune_pjrt.Ffi.Internal.target handler in
      write "second";
      let second = Rune_pjrt.Ffi.Internal.target handler in
      require "changed library changes target" (first <> second))

let () =
  test_eager_fallback ();
  test_cuda_trace_dispatch ();
  test_cpu_trace_fallback ();
  test_missing_direction_fallback ();
  test_backward_operand_order ();
  test_transforms_use_fallback ();
  test_content_addressed_target ()
