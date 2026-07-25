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
  Rune_pjrt.Grouped_gemm.create ~library:"grouped-gemm-not-needed.so" ()

let lhs =
  Nx.init Nx.float32 [| rows; inner |] (fun indices ->
      let row = float_of_int indices.(0) in
      let column = float_of_int indices.(1) in
      Float.sin ((row *. 0.13) +. (column *. 0.07)))

let rhs =
  Nx.init Nx.float32 [| groups; inner; columns |] (fun indices ->
      let group = float_of_int indices.(0) in
      let row = float_of_int indices.(1) in
      let column = float_of_int indices.(2) in
      Float.cos ((group *. 0.19) +. (row *. 0.05) -. (column *. 0.03)))

let group_sizes = Nx.create Nx.int32 [| groups |] group_values

let expected group_values =
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
              +. (lhs.((row * inner) + offset)
                 *. rhs.((((group * inner) + offset) * columns) + column))
          done;
          output.((row * columns) + column) <- !sum
        done
      done;
      row_start := !row_start + size)
    group_values;
  Nx.create Nx.float32 [| rows; columns |] output

let max_abs_error expected actual =
  let expected = Nx.to_array expected in
  let actual = Nx.to_array actual in
  let error = ref 0.0 in
  Array.iteri
    (fun index value ->
      error :=
        Float.max !error (Float.abs (value -. actual.(index))))
    expected;
  !error

let require_close name expected actual =
  let error = max_abs_error expected actual in
  if error > 1e-4 then
    failwith (Printf.sprintf "test_grouped_gemm: %s max error is %g" name error)

let test_reference () =
  let actual =
    Rune_pjrt.Grouped_gemm.reference ~lhs ~rhs ~group_sizes
  in
  require_close "reference" (expected group_values) actual

let test_eager_fallback () =
  let actual = Rune_pjrt.Grouped_gemm.run kernel ~lhs ~rhs ~group_sizes in
  require_close "eager fallback" (expected group_values) actual

let test_gradients () =
  let lhs_values = Nx.to_array lhs in
  let rhs_values = Nx.to_array rhs in
  let expected_lhs = Array.make (rows * inner) 0.0 in
  let expected_rhs = Array.make (groups * inner * columns) 0.0 in
  let row_start = ref 0 in
  Array.iteri
    (fun group size ->
      let size = Int32.to_int size in
      for offset = 0 to inner - 1 do
        let rhs_sum = ref 0.0 in
        for column = 0 to columns - 1 do
          rhs_sum :=
            !rhs_sum
            +. rhs_values.((((group * inner) + offset) * columns) + column)
        done;
        let lhs_sum = ref 0.0 in
        for local_row = 0 to size - 1 do
          let row = !row_start + local_row in
          expected_lhs.((row * inner) + offset) <- !rhs_sum;
          lhs_sum := !lhs_sum +. lhs_values.((row * inner) + offset)
        done;
        for column = 0 to columns - 1 do
          expected_rhs.((((group * inner) + offset) * columns) + column) <-
            !lhs_sum
        done
      done;
      row_start := !row_start + size)
    group_values;
  let gradients =
    Rune.grads
      (function
        | [ lhs; rhs ] ->
            Rune_pjrt.Grouped_gemm.run kernel ~lhs ~rhs ~group_sizes
            |> Nx.sum
        | _ -> failwith "test_grouped_gemm: expected two differentiable inputs")
      [ lhs; rhs ]
  in
  match gradients with
  | [ actual_lhs; actual_rhs ] ->
      require_close "lhs gradient"
        (Nx.create Nx.float32 [| rows; inner |] expected_lhs)
        actual_lhs;
      require_close "rhs gradient"
        (Nx.create Nx.float32 [| groups; inner; columns |] expected_rhs)
        actual_rhs
  | _ -> failwith "test_grouped_gemm: expected two gradients"

let cpu_runner =
  Rune_pjrt.jits_packed ~backend:`Cpu (function
    | [ Rune_pjrt.Tensor lhs; Rune_pjrt.Tensor rhs; Rune_pjrt.Tensor sizes ] ->
        let lhs : (float, Nx.float32_elt) Nx.t = Obj.magic lhs in
        let rhs : (float, Nx.float32_elt) Nx.t = Obj.magic rhs in
        let sizes : Nx.int32_t = Obj.magic sizes in
        [
          Rune_pjrt.Tensor
            (Rune_pjrt.Grouped_gemm.run kernel ~lhs ~rhs
               ~group_sizes:sizes);
        ]
    | _ -> failwith "test_grouped_gemm: expected three inputs")

let test_pjrt_cpu () =
  let check name group_values =
    let group_sizes = Nx.create Nx.int32 [| groups |] group_values in
    match
      cpu_runner
        [
          Rune_pjrt.Tensor lhs;
          Rune_pjrt.Tensor rhs;
          Rune_pjrt.Tensor group_sizes;
        ]
    with
    | [ Rune_pjrt.Tensor actual ] ->
        require_close name (expected group_values) (Obj.magic actual)
    | _ -> failwith "test_grouped_gemm: expected one output"
  in
  if Rune_pjrt.backend_available `Cpu then (
    check "PJRT CPU initial" group_values;
    check "PJRT CPU rebalanced" alternate_group_values)

let test_invalid_sizes () =
  let invalid =
    Nx.create Nx.int32 [| groups |] [| 2l; 0l; 17l; 3l |]
  in
  match
    Rune_pjrt.Grouped_gemm.reference ~lhs ~rhs ~group_sizes:invalid
  with
  | _ -> failwith "test_grouped_gemm: invalid group sizes were accepted"
  | exception Invalid_argument _ -> ()

let () =
  test_reference ();
  test_eager_fallback ();
  test_gradients ();
  test_pjrt_cpu ();
  test_invalid_sizes ()
