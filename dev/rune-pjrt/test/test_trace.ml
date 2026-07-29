(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let failf fmt = Printf.ksprintf failwith fmt
let require msg condition = if not condition then failf "test_trace: %s" msg

let contains_substring text substring =
  let text_length = String.length text in
  let substring_length = String.length substring in
  let rec loop offset =
    if offset + substring_length > text_length then false
    else if String.sub text offset substring_length = substring then true
    else loop (offset + 1)
  in
  loop 0

let test_basic_trace () =
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let capture =
    Rune_pjrt.Trace.capture_one (fun t -> Nx.add (Nx.mul t t) (Nx.sin t)) x
  in
  let program = capture.program in
  require "single input" (List.length program.inputs = 1);
  require "single output" (List.length program.outputs = 1);
  require "contains mul"
    (List.exists
       (fun node -> Rune_pjrt.Ir.op_name node.Rune_pjrt.Ir.op = "mul")
       program.nodes);
  require "contains sin"
    (List.exists
       (fun node -> Rune_pjrt.Ir.op_name node.Rune_pjrt.Ir.op = "sin")
       program.nodes)

let test_constant_capture () =
  let bias = Nx.full Nx.float32 [| 2; 2 |] 3.0 in
  let x = Nx.ones Nx.float32 [| 2; 2 |] in
  let capture = Rune_pjrt.Trace.capture_one (fun t -> Nx.add t bias) x in
  let constant_nodes =
    List.filter
      (fun node ->
        match node.Rune_pjrt.Ir.op with
        | Rune_pjrt.Ir.Constant _ -> true
        | _ -> false)
      capture.program.nodes
  in
  require "closed tensor constant captured" (constant_nodes <> [])

let test_matmul_capture_uses_shape_placeholder () =
  let lhs = Nx.ones Nx.float32 [| 64; 128 |] in
  let rhs = Nx.ones Nx.float32 [| 128; 256 |] in
  let capture =
    Rune_pjrt.Trace.capture_one (fun input -> Nx.matmul input rhs) lhs
  in
  match capture.outputs with
  | [ Rune_pjrt.Trace.Tensor output ] ->
      require "matmul placeholder has the inferred output shape"
        (Nx.shape output = [| 64; 256 |]);
      require "matmul placeholder has scalar backing storage"
        (Nx_buffer.length (Nx.data output) = 1)
  | _ -> failf "test_trace: expected one matmul output"

let test_erf_lowering () =
  let x = Nx.create Nx.float32 [| 3 |] [| -1.; 0.; 1. |] in
  let capture = Rune_pjrt.Trace.capture_one Nx.erf x in
  let module_text = Rune_pjrt.Stablehlo.of_program capture.program in
  require "erf lowers to CHLO with explicit input and result types"
    (contains_substring module_text
       "chlo.erf %arg0 : tensor<3xf32> -> tensor<3xf32>")

let test_float16_reciprocal_lowering () =
  let x = Nx.ones Nx.float16 [| 3 |] in
  let capture = Rune_pjrt.Trace.capture_one Nx.recip x in
  let module_text = Rune_pjrt.Stablehlo.of_program capture.program in
  require "float16 reciprocal uses a floating-point one literal"
    (contains_substring module_text
       "stablehlo.constant dense<1.000000e+00> : tensor<3xf16>")

let test_float16_non_finite_literal_lowering () =
  let x = Nx.ones Nx.float16 [| 3 |] in
  let negative_infinity = Nx.scalar Nx.float16 (-1.0e9) in
  let capture =
    Rune_pjrt.Trace.capture_one (fun input -> Nx.add input negative_infinity) x
  in
  let module_text = Rune_pjrt.Stablehlo.of_program capture.program in
  require "float16 negative infinity uses its bit-pattern literal"
    (contains_substring module_text
       "stablehlo.constant dense<0xFC00> : tensor<f16>")

let () =
  test_basic_trace ();
  test_constant_capture ();
  test_matmul_capture_uses_shape_placeholder ();
  test_erf_lowering ();
  test_float16_reciprocal_lowering ();
  test_float16_non_finite_literal_lowering ()
