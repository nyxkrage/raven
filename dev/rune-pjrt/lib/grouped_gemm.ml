(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core

type t = Ffi.Kernel.t

type problem = {
  rows : int;
  inner : int;
  columns : int;
  groups : int;
}

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let create ~library ?(symbol = "raven_grouped_gemm_fwd") () =
  Ffi.Kernel.create ~library ~fwd:symbol ()

let validate_dtype (type a) (dtype : (float, a) Dtype.t) =
  match dtype with
  | Float16 | BFloat16 | Float32 -> ()
  | dtype ->
      invalid_argf
        "Grouped_gemm: expected float16, bfloat16, or float32 inputs, got %s"
        (Dtype.to_string dtype)

let uses_float32_accumulation (type a) (dtype : (float, a) Dtype.t) =
  match dtype with
  | Float16 | BFloat16 -> true
  | Float32 -> false
  | _ -> assert false

let validate ~lhs ~rhs ~group_sizes =
  validate_dtype (Nx.dtype lhs);
  let lhs_shape = Nx.shape lhs in
  let rhs_shape = Nx.shape rhs in
  let group_shape = Nx.shape group_sizes in
  if Array.length lhs_shape <> 2 then
    invalid_argf "Grouped_gemm: lhs must have rank 2, got shape %s"
      (Shape.to_string lhs_shape);
  if Array.length rhs_shape <> 3 then
    invalid_argf "Grouped_gemm: rhs must have rank 3, got shape %s"
      (Shape.to_string rhs_shape);
  if Array.length group_shape <> 1 then
    invalid_argf "Grouped_gemm: group_sizes must have rank 1, got shape %s"
      (Shape.to_string group_shape);
  let rows = lhs_shape.(0) in
  let inner = lhs_shape.(1) in
  let groups = rhs_shape.(0) in
  let columns = rhs_shape.(2) in
  if groups <= 0 then
    invalid_argf "Grouped_gemm: rhs must contain at least one group";
  if inner <= 0 || columns <= 0 then
    invalid_argf
      "Grouped_gemm: inner and output dimensions must be positive, got %d and \
       %d"
      inner columns;
  if rhs_shape.(1) <> inner then
    invalid_argf
      "Grouped_gemm: lhs inner dimension %d does not match rhs inner dimension \
       %d"
      inner rhs_shape.(1);
  if group_shape.(0) <> groups then
    invalid_argf
      "Grouped_gemm: expected %d group sizes for rhs, got %d" groups
      group_shape.(0);
  let total = ref 0L in
  Nx.to_array group_sizes
  |> Array.iteri (fun group size ->
         if Int32.compare size 0l < 0 then
           invalid_argf "Grouped_gemm: group size %d is negative" group;
         total := Int64.add !total (Int64.of_int32 size));
  if !total <> Int64.of_int rows then
    invalid_argf
      "Grouped_gemm: group sizes sum to %Ld, expected the lhs row count %d"
      !total rows;
  { rows; inner; columns; groups }

let reference_problem problem ~lhs ~rhs ~group_sizes =
  let output =
    ref (Nx.zeros (Nx.dtype lhs) [| problem.rows; problem.columns |])
  in
  if problem.rows = 0 then !output
  else
    let row_ids = Nx.arange Nx.int32 0 problem.rows 1 in
    let start = ref (Nx.scalar Nx.int32 0l) in
    let false_value = Nx.scalar Nx.bool false in
    for group = 0 to problem.groups - 1 do
      let stop = Nx.add !start (Nx.get [ group ] group_sizes) in
      let after_start = Nx.less_equal !start row_ids in
      let before_stop = Nx.less row_ids stop in
      let in_group = Nx.where after_start before_stop false_value in
      let in_group = Nx.reshape [| problem.rows; 1 |] in_group in
      let group_rhs = Nx.get [ group ] rhs in
      let product =
        if uses_float32_accumulation (Nx.dtype lhs) then
          Nx.matmul
            (Nx.astype Nx.float32 lhs)
            (Nx.astype Nx.float32 group_rhs)
          |> Nx.astype (Nx.dtype lhs)
        else Nx.matmul lhs group_rhs
      in
      output := Nx.where in_group product !output;
      start := stop
    done;
    !output

let reference ~lhs ~rhs ~group_sizes =
  let problem = validate ~lhs ~rhs ~group_sizes in
  reference_problem problem ~lhs ~rhs ~group_sizes

let run kernel ~lhs ~rhs ~group_sizes =
  let problem = validate ~lhs ~rhs ~group_sizes in
  Ffi.call_fwd kernel
    ~inputs:[ Ffi.Tensor lhs; Ffi.Tensor rhs; Ffi.Tensor group_sizes ]
    ~fallback:(fun () -> reference_problem problem ~lhs ~rhs ~group_sizes)
