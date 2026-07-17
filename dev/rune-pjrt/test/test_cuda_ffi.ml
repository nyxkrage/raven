(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let scale = 0.125
let tolerance = 3e-6

let linear_index shape indices =
  let index = ref 0 in
  for axis = 0 to Array.length shape - 1 do
    index := (!index * shape.(axis)) + indices.(axis)
  done;
  !index

let causal_scaled_softmax scores =
  let shape = Nx.shape scores in
  let rank = Array.length shape in
  let rows = shape.(rank - 2) in
  let columns = shape.(rank - 1) in
  let scaled = Nx.mul scores (Nx.scalar_like scores scale) in
  let ones = Nx.full (Nx.dtype scores) [| rows; columns |] 1.0 in
  let mask = Nx.tril ones |> Nx.cast Nx.bool |> Nx.broadcast_to shape in
  let masked = Nx.where mask scaled (Nx.scalar_like scores (-1e9)) in
  Nx.softmax ~axes:[ -1 ] masked

let fused_causal_scaled_softmax scores = causal_scaled_softmax scores
[@@rune.kernel.cuda
  {
    library = "../kernels/causal_scaled_softmax.so";
    fwd = "raven_causal_scaled_softmax_fwd";
    bwd = "raven_causal_scaled_softmax_bwd";
  }]

let max_abs_error expected actual =
  let expected = Nx.to_array expected in
  let actual = Nx.to_array actual in
  if Array.length expected <> Array.length actual then
    failwith "test_cuda_ffi: output lengths differ";
  let error = ref 0.0 in
  for i = 0 to Array.length expected - 1 do
    error := Float.max !error (Float.abs (expected.(i) -. actual.(i)))
  done;
  !error

let forward_invariants ~rows ~sequence probabilities =
  let probabilities = Nx.to_array probabilities in
  let max_sum_error = ref 0.0 in
  let max_masked_value = ref 0.0 in
  for row = 0 to rows - 1 do
    let query = row mod sequence in
    let offset = row * sequence in
    let sum = ref 0.0 in
    for key = 0 to query do
      sum := !sum +. probabilities.(offset + key)
    done;
    max_sum_error := Float.max !max_sum_error (Float.abs (!sum -. 1.0));
    for key = query + 1 to sequence - 1 do
      max_masked_value :=
        Float.max !max_masked_value (Float.abs probabilities.(offset + key))
    done
  done;
  (!max_sum_error, !max_masked_value)

let compiled_value_and_vjp =
  Rune_pjrt.jits ~backend:`Cuda (function
    | [ scores; output_cotangents ] ->
        let probabilities, input_cotangents =
          Rune.vjp fused_causal_scaled_softmax scores output_cotangents
        in
        [ probabilities; input_cotangents ]
    | _ -> failwith "test_cuda_ffi: expected scores and output cotangents")

let compiled_forward = Rune_pjrt.jit ~backend:`Cuda fused_causal_scaled_softmax

let compiled_layout_chain =
  Rune_pjrt.jit ~backend:`Cuda (fun scores ->
      scores
      |> Nx.transpose ~axes:[ 0; 1; 3; 2 ]
      |> fused_causal_scaled_softmax
      |> Nx.transpose ~axes:[ 0; 1; 3; 2 ])

let check_layout_chain () =
  let shape = [| 1; 2; 4; 4 |] in
  let scores =
    Nx.init Nx.float32 shape (fun indices ->
        let i = linear_index shape indices |> float_of_int in
        Float.sin (i *. 0.19) +. (Float.cos (i *. 0.07) *. 0.5))
  in
  let expected =
    scores
    |> Nx.transpose ~axes:[ 0; 1; 3; 2 ]
    |> causal_scaled_softmax
    |> Nx.transpose ~axes:[ 0; 1; 3; 2 ]
  in
  let error = max_abs_error expected (compiled_layout_chain scores) in
  Printf.printf "layout_chain fwd_max_abs=%.9g\n%!" error;
  if error > tolerance then
    failwith "test_cuda_ffi: layout-chain tolerance exceeded"

let check_shape ~batch ~heads ~sequence =
  let shape = [| batch; heads; sequence; sequence |] in
  let count = batch * heads * sequence * sequence in
  let scores =
    Nx.init Nx.float32 shape (fun indices ->
        let i = linear_index shape indices |> float_of_int in
        (Float.sin (i *. 0.013) *. 4.0) +. (Float.cos (i *. 0.007) *. 0.75))
  in
  let output_cotangents =
    Nx.init Nx.float32 shape (fun indices ->
        let i = linear_index shape indices |> float_of_int in
        Float.sin (i *. 0.021) -. (Float.cos (i *. 0.017) *. 0.5))
  in
  if Nx.numel scores <> count then failwith "test_cuda_ffi: invalid test input";
  let expected_probabilities, expected_input_cotangents =
    Rune.vjp causal_scaled_softmax scores output_cotangents
  in
  let forward_probabilities = compiled_forward scores in
  let probabilities, input_cotangents =
    match compiled_value_and_vjp [ scores; output_cotangents ] with
    | [ probabilities; input_cotangents ] -> (probabilities, input_cotangents)
    | _ -> failwith "test_cuda_ffi: expected two outputs"
  in
  let forward_error = max_abs_error expected_probabilities probabilities in
  let isolated_forward_error =
    max_abs_error expected_probabilities forward_probabilities
  in
  let backward_error =
    max_abs_error expected_input_cotangents input_cotangents
  in
  let row_sum_error, masked_value =
    forward_invariants ~rows:(batch * heads * sequence) ~sequence probabilities
  in
  Printf.printf
    "shape=[%d,%d,%d,%d] fwd_max_abs=%.9g isolated_fwd=%.9g row_sum=%.9g \
     masked=%.9g bwd_max_abs=%.9g\n\
     %!"
    batch heads sequence sequence forward_error isolated_forward_error
    row_sum_error masked_value backward_error;
  if isolated_forward_error > tolerance then
    failwith "test_cuda_ffi: isolated forward tolerance exceeded";
  if forward_error > tolerance then
    failwith "test_cuda_ffi: forward tolerance exceeded";
  if backward_error > tolerance then
    failwith "test_cuda_ffi: backward tolerance exceeded";
  if row_sum_error > tolerance then
    failwith "test_cuda_ffi: row-sum tolerance exceeded";
  if masked_value <> 0.0 then
    failwith "test_cuda_ffi: causal mask was not exact"

let check_extreme_mask_semantics () =
  let scores =
    Nx.create Nx.float32 [| 1; 1; 2; 2 |] [| -1.0e10; 2.0; -1.0e10; -1.0e10 |]
  in
  let output_cotangents =
    Nx.create Nx.float32 [| 1; 1; 2; 2 |] [| 0.5; -0.25; 1.0; -2.0 |]
  in
  let expected_probabilities, expected_input_cotangents =
    Rune.vjp causal_scaled_softmax scores output_cotangents
  in
  let probabilities, input_cotangents =
    match compiled_value_and_vjp [ scores; output_cotangents ] with
    | [ probabilities; input_cotangents ] -> (probabilities, input_cotangents)
    | _ -> failwith "test_cuda_ffi: expected two extreme-test outputs"
  in
  let forward_error = max_abs_error expected_probabilities probabilities in
  let backward_error =
    max_abs_error expected_input_cotangents input_cotangents
  in
  Printf.printf "extreme_mask fwd_max_abs=%.9g bwd_max_abs=%.9g\n%!"
    forward_error backward_error;
  if forward_error > tolerance || backward_error > tolerance then
    failwith "test_cuda_ffi: finite mask semantics differ from fallback"

let () =
  if Rune_pjrt.backend_available `Cuda then (
    check_layout_chain ();
    check_shape ~batch:1 ~heads:2 ~sequence:4;
    check_shape ~batch:1 ~heads:12 ~sequence:128;
    check_shape ~batch:1 ~heads:2 ~sequence:129;
    check_shape ~batch:1 ~heads:12 ~sequence:256;
    check_shape ~batch:1 ~heads:12 ~sequence:512;
    check_shape ~batch:1 ~heads:2 ~sequence:513;
    check_shape ~batch:1 ~heads:12 ~sequence:1024;
    check_extreme_mask_semantics ())
  else Printf.printf "test_cuda_ffi: CUDA plugin unavailable, skipping\n%!"
