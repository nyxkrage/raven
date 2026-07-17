(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let causal_scaled_softmax scores =
  let shape = Nx.shape scores in
  let rank = Array.length shape in
  let rows = shape.(rank - 2) in
  let columns = shape.(rank - 1) in
  let scaled = Nx.mul scores (Nx.scalar_like scores 0.125) in
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

let now () = Unix.gettimeofday ()

let percentile sorted fraction =
  let last = Array.length sorted - 1 in
  sorted.(min last (int_of_float (fraction *. float_of_int last)))

let time function_ input =
  let started = now () in
  let output = function_ input in
  let elapsed_ms = (now () -. started) *. 1000.0 in
  (output, elapsed_ms)

let parse_arguments () =
  if Array.length Sys.argv <> 5 then
    failwith
      "usage: softmax_profile.exe (baseline|kernel) SEQUENCE WARMUPS ITERATIONS";
  let implementation = Sys.argv.(1) in
  if implementation <> "baseline" && implementation <> "kernel" then
    invalid_arg "implementation must be baseline or kernel";
  let sequence = int_of_string Sys.argv.(2) in
  let warmups = int_of_string Sys.argv.(3) in
  let iterations = int_of_string Sys.argv.(4) in
  if sequence <= 0 || warmups < 0 || iterations <= 0 then
    invalid_arg "sequence and iterations must be positive; warmups may be zero";
  (implementation, sequence, warmups, iterations)

let () =
  let implementation, sequence, warmups, iterations = parse_arguments () in
  let shape = [| 1; 12; sequence; sequence |] in
  let input = Nx.zeros Nx.float32 shape in
  let body =
    if implementation = "kernel" then fused_causal_scaled_softmax
    else causal_scaled_softmax
  in
  let compiled = Rune_pjrt.jit ~backend:`Cuda body in
  Printf.printf
    "implementation=%s shape=[1,12,%d,%d] elements=%d bytes=%d warmups=%d \
     iterations=%d\n\
     %!"
    implementation sequence sequence (Nx.numel input) (Nx.nbytes input) warmups
    iterations;
  let first, first_ms = time compiled input in
  Printf.printf "first_compile_and_execute_ms=%.6f\n%!" first_ms;
  let last = ref first in
  for _ = 1 to warmups do
    let output, _ = time compiled input in
    last := output
  done;
  let samples = Array.make iterations 0.0 in
  for index = 0 to iterations - 1 do
    let output, elapsed_ms = time compiled input in
    last := output;
    samples.(index) <- elapsed_ms
  done;
  let sorted = Array.copy samples in
  Array.sort Float.compare sorted;
  let mean = Array.fold_left ( +. ) 0.0 samples /. float_of_int iterations in
  Printf.printf
    "steady_e2e_ms mean=%.6f p10=%.6f median=%.6f p90=%.6f min=%.6f max=%.6f\n\
     %!"
    mean (percentile sorted 0.10) (percentile sorted 0.50)
    (percentile sorted 0.90) sorted.(0)
    sorted.(iterations - 1);
  let total : float = Nx.item [] (Nx.sum !last) in
  let expected = float_of_int (12 * sequence) in
  Printf.printf "output_sum=%.9g expected=%.9g abs_error=%.9g\n%!" total
    expected
    (Float.abs (total -. expected))
