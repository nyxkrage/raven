(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let rows = 23
let inner = 19
let columns = 21
let groups = 4
let group_values = [| 3l; 0l; 17l; 3l |]
let alternate_group_values = [| 0l; 8l; 1l; 14l |]

let kernel =
  Rune_pjrt.Grouped_gemm.create ~library:"../kernels/grouped_gemm.so" ()

let runner =
  Rune_pjrt.jits_packed ~backend:`Cuda (function
    | [ Rune_pjrt.Tensor lhs; Rune_pjrt.Tensor rhs; Rune_pjrt.Tensor sizes ] ->
        let lhs : (float, Nx.float32_elt) Nx.t = Obj.magic lhs in
        let rhs : (float, Nx.float32_elt) Nx.t = Obj.magic rhs in
        let sizes : Nx.int32_t = Obj.magic sizes in
        [
          Rune_pjrt.Tensor
            (Rune_pjrt.Grouped_gemm.run kernel ~lhs ~rhs ~group_sizes:sizes);
        ]
    | _ -> failwith "test_cuda_grouped_gemm: expected three inputs")

let run lhs rhs group_sizes =
  match
    runner
      [
        Rune_pjrt.Tensor lhs; Rune_pjrt.Tensor rhs; Rune_pjrt.Tensor group_sizes;
      ]
  with
  | [ Rune_pjrt.Tensor output ] -> Obj.magic output
  | _ -> failwith "test_cuda_grouped_gemm: expected one output"

let expected (type a) (dtype : (float, a) Nx.dtype) group_values lhs rhs =
  let lhs_shape = Nx.shape lhs in
  let rhs_shape = Nx.shape rhs in
  let rows = lhs_shape.(0) in
  let inner = lhs_shape.(1) in
  let columns = rhs_shape.(2) in
  let lhs = Nx.to_array lhs in
  let rhs = Nx.to_array rhs in
  let output = Array.make (rows * columns) 0.0 in
  let row_start = ref 0 in
  Array.iteri
    (fun group size ->
      let size = Int32.to_int size in
      for local_row = 0 to size - 1 do
        let row = !row_start + local_row in
        for column = 0 to columns - 1 do
          let sum = ref 0.0 in
          for offset = 0 to inner - 1 do
            sum :=
              !sum
              +. lhs.((row * inner) + offset)
                 *. rhs.((((group * inner) + offset) * columns) + column)
          done;
          output.((row * columns) + column) <- !sum
        done
      done;
      row_start := !row_start + size)
    group_values;
  Nx.create dtype [| rows; columns |] output

let max_abs_error expected actual =
  let expected = Nx.to_array expected in
  let actual = Nx.to_array actual in
  let error = ref 0.0 in
  let error_index = ref 0 in
  Array.iteri
    (fun index value ->
      let difference = Float.abs (value -. actual.(index)) in
      if difference > !error then (
        error := difference;
        error_index := index))
    expected;
  (!error, !error_index)

let print_sample expected actual error_index =
  let expected = Nx.to_array expected in
  let actual = Nx.to_array actual in
  let first = Int.max 0 (error_index - 4) in
  let last = Int.min (Array.length expected - 1) (error_index + 4) in
  for index = first to last do
    Printf.printf "  [%d] expected=%g actual=%g\n%!" index expected.(index)
      actual.(index)
  done

let check_case (type a) name (dtype : (float, a) Nx.dtype) tolerance ~rows
    ~inner ~columns ~groups layouts =
  let lhs =
    Nx.init dtype [| rows; inner |] (fun indices ->
        let row = float_of_int indices.(0) in
        let column = float_of_int indices.(1) in
        Float.sin ((row *. 0.13) +. (column *. 0.07)))
  in
  let rhs =
    Nx.init dtype [| groups; inner; columns |] (fun indices ->
        let group = float_of_int indices.(0) in
        let row = float_of_int indices.(1) in
        let column = float_of_int indices.(2) in
        Float.cos ((group *. 0.19) +. (row *. 0.05) -. (column *. 0.03)))
  in
  let check_layout (layout, group_values) =
    let group_sizes = Nx.create Nx.int32 [| groups |] group_values in
    let expected = expected dtype group_values lhs rhs in
    let reference = Rune_pjrt.Grouped_gemm.reference ~lhs ~rhs ~group_sizes in
    let reference_error, reference_error_index =
      max_abs_error expected reference
    in
    if reference_error > tolerance then (
      print_sample expected reference reference_error_index;
      failwith
        (Printf.sprintf
           "test_cuda_grouped_gemm: %s %s reference tolerance exceeded (%g > \
            %g)"
           name layout reference_error tolerance));
    let actual : (float, a) Nx.t = run lhs rhs group_sizes in
    let error, error_index = max_abs_error expected actual in
    Printf.printf "%s %s max_abs=%.9g\n%!" name layout error;
    if error > tolerance then (
      Printf.printf "max error at row=%d column=%d\n%!" (error_index / columns)
        (error_index mod columns);
      print_sample expected actual error_index;
      failwith
        (Printf.sprintf
           "test_cuda_grouped_gemm: %s %s tolerance exceeded (%g > %g)" name
           layout error tolerance))
  in
  List.iter check_layout layouts

let () =
  if Rune_pjrt.backend_available `Cuda then (
    let generic_layouts =
      [ ("initial", group_values); ("rebalanced", alternate_group_values) ]
    in
    check_case "f32 generic" Nx.float32 2e-5 ~rows ~inner ~columns ~groups
      generic_layouts;
    check_case "f16 generic" Nx.float16 2e-2 ~rows ~inner ~columns ~groups
      generic_layouts;
    check_case "bf16 generic" Nx.bfloat16 8e-2 ~rows ~inner ~columns ~groups
      generic_layouts;
    let tensor32_layouts =
      [
        ("sparse", [| 0l; 1l; 63l; 65l; 0l; 7l |]);
        ("rebalanced", [| 32l; 17l; 0l; 64l; 23l; 0l |]);
      ]
    in
    check_case "f16 tensor32" Nx.float16 2e-2 ~rows:136 ~inner:40 ~columns:72
      ~groups:6 tensor32_layouts;
    check_case "bf16 tensor32" Nx.bfloat16 8e-2 ~rows:136 ~inner:40 ~columns:72
      ~groups:6 tensor32_layouts;
    let tensor64_layouts =
      [ ("sparse", [| 0l; 136l |]); ("rebalanced", [| 65l; 71l |]) ]
    in
    check_case "f16 tensor64" Nx.float16 2e-2 ~rows:136 ~inner:40 ~columns:72
      ~groups:2 tensor64_layouts;
    check_case "bf16 tensor64" Nx.bfloat16 8e-2 ~rows:136 ~inner:40 ~columns:72
      ~groups:2 tensor64_layouts;
    let many_groups = 35 in
    let sparse_groups =
      Array.init many_groups (fun group -> if group = 34 then 70l else 0l)
    in
    let balanced_groups = Array.make many_groups 2l in
    check_case "f16 many-groups" Nx.float16 2e-2 ~rows:70 ~inner:40 ~columns:72
      ~groups:many_groups
      [ ("sparse", sparse_groups); ("balanced", balanced_groups) ])
  else if Sys.getenv_opt "RUNE_PJRT_TEST_REQUIRE_CUDA" <> None then
    failwith ("test_cuda_grouped_gemm: " ^ Rune_pjrt.status ())
  else
    Printf.printf
      "test_cuda_grouped_gemm: CUDA plugin unavailable, skipping\n%!"
