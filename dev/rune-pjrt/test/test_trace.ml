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
    Rune_pjrt.Trace.capture_one
      (fun t -> Nx.add (Nx.mul t t) (Nx.sin t))
      x
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
        match node.Rune_pjrt.Ir.op with Rune_pjrt.Ir.Constant _ -> true | _ -> false)
      capture.program.nodes
  in
  require "closed tensor constant captured" (constant_nodes <> [])

let test_erf_lowering () =
  let x = Nx.create Nx.float32 [| 3 |] [| -1.; 0.; 1. |] in
  let capture = Rune_pjrt.Trace.capture_one Nx.erf x in
  let module_text = Rune_pjrt.Stablehlo.of_program capture.program in
  require "erf lowers to CHLO with explicit input and result types"
    (contains_substring module_text
       "chlo.erf %arg0 : tensor<3xf32> -> tensor<3xf32>")

let () =
  test_basic_trace ();
  test_constant_capture ();
  test_erf_lowering ()
