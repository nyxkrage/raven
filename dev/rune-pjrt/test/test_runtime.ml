(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let test_runtime_status () =
  let status = Rune_pjrt.Runtime.status () in
  if String.length status = 0 then
    failwith "test_runtime: missing runtime status"

let skip_cuda () = Sys.getenv_opt "RUNE_PJRT_TEST_SKIP_CUDA" <> None

let backend_available backend =
  match backend with
  | `Cuda when skip_cuda () -> false
  | (`Cpu | `Cuda) as backend -> Rune_pjrt.Runtime.backend_available backend

let check_close name expected actual =
  let exp = Nx.to_array expected in
  let got = Nx.to_array actual in
  if Array.length exp <> Array.length got then
    failwith (Printf.sprintf "%s: length mismatch" name);
  Array.iteri
    (fun i x ->
      let y = got.(i) in
      if Float.abs (x -. y) > 1e-4 then
        failwith
          (Printf.sprintf "%s: mismatch at %d expected=%g got=%g" name i x y))
    exp

let test_jit_cpu_executes () =
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let f = Rune_pjrt.jit ~backend:`Cpu (fun t -> Nx.add (Nx.mul t t) (Nx.sin t)) in
  let expected = Nx.add (Nx.mul x x) (Nx.sin x) in
  let actual = f x in
  check_close "jit_cpu_executes" expected actual

let test_jit_cpu_argmax_executes () =
  let x =
    Nx.create Nx.float32 [| 3; 4 |]
      [| 1.; 9.; 3.; 4.; 5.; 6.; 7.; 8.; 4.; 3.; 2.; 1. |]
  in
  let f = Rune_pjrt.jit ~backend:`Cpu (fun t -> Nx.argmax ~axis:1 t) in
  let expected = Nx.argmax ~axis:1 x |> Nx.to_array in
  let actual = f x |> Nx.to_array in
  if actual <> expected then
    failwith "jit_cpu_argmax_executes: mismatch"

let test_jit_cuda_executes () =
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let f =
    Rune_pjrt.jit ~backend:`Cuda (fun t -> Nx.add (Nx.mul t t) (Nx.sin t))
  in
  let expected = Nx.add (Nx.mul x x) (Nx.sin x) in
  let actual = f x in
  check_close "jit_cuda_executes" expected actual

let () =
  test_runtime_status ();
  if backend_available `Cpu then (
    test_jit_cpu_executes ();
    test_jit_cpu_argmax_executes ());
  if backend_available `Cuda then test_jit_cuda_executes ()
