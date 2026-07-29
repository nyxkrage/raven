(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let test_runtime_status () =
  let status = Rune_pjrt.Runtime.status () in
  if String.length status = 0 then
    failwith "test_runtime: missing runtime status"

let skip_cuda () = Sys.getenv_opt "RUNE_PJRT_TEST_SKIP_CUDA" <> None
let require_cuda () = Sys.getenv_opt "RUNE_PJRT_TEST_REQUIRE_CUDA" <> None

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

let test_sigmoid_constants_remain_scalar () =
  let input = Nx.zeros Nx.float32 [| 4096 |] in
  let capture = Rune_pjrt.Trace.capture_one Nx.sigmoid input in
  let _, lifted = Rune_pjrt.Ir.parameterize_constants capture.program in
  if lifted <> [] then
    failwith
      "sigmoid_constants_remain_scalar: sigmoid materialized a tensor constant"

let test_jit_cpu_executes () =
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let f =
    Rune_pjrt.jit ~backend:`Cpu (fun t -> Nx.add (Nx.mul t t) (Nx.sin t))
  in
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
  if actual <> expected then failwith "jit_cpu_argmax_executes: mismatch"

let test_rune_jit_pjrt_cpu_executes () =
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let device = Rune.Device.pjrt (Rune_pjrt.Device.cpu ()) in
  let f = Rune.jit ~device (fun t -> Nx.add t (Nx.scalar Nx.float32 1.0)) in
  let expected = Nx.add x (Nx.scalar Nx.float32 1.0) in
  let actual = f x in
  check_close "rune_jit_pjrt_cpu_executes" expected actual

let test_jit_erf_executes backend =
  let x = Nx.create Nx.float32 [| 7 |] [| -4.; -1.; -0.5; 0.; 0.5; 1.; 4. |] in
  let expected = Nx.erf x in
  let actual = Rune_pjrt.jit ~backend Nx.erf x in
  check_close
    (Printf.sprintf "jit_%s_erf_executes" (Rune_pjrt.Backend.to_string backend))
    expected actual

let test_jit_cuda_executes () =
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let f =
    Rune_pjrt.jit ~backend:`Cuda (fun t -> Nx.add (Nx.mul t t) (Nx.sin t))
  in
  let expected = Nx.add (Nx.mul x x) (Nx.sin x) in
  let actual = f x in
  check_close "jit_cuda_executes" expected actual

let test_jits_cuda_host_multiple_inputs () =
  let size = 1_048_576 in
  let lhs =
    Nx.init Nx.float32 [| size |] (fun indices ->
        float_of_int (indices.(0) mod 257) /. 257.)
  in
  let rhs =
    Nx.init Nx.float32 [| size |] (fun indices ->
        float_of_int (indices.(0) mod 251) /. 251.)
  in
  let compiled =
    Rune_pjrt.jits ~backend:`Cuda (function
      | [ x; y ] -> [ Nx.add x y ]
      | _ -> failwith "expected two inputs")
  in
  let expected = Nx.add lhs rhs in
  for _ = 1 to 3 do
    match compiled [ lhs; rhs ] with
    | [ actual ] -> check_close "jits_cuda_host_multiple_inputs" expected actual
    | _ -> failwith "jits_cuda_host_multiple_inputs: wrong output arity"
  done

let test_jit_cuda_device_executes () =
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let x_device = Rune_pjrt.Device_buffer.of_host x in
  let f = Rune_pjrt.jit_device (fun t -> Nx.add (Nx.mul t t) (Nx.sin t)) in
  let actual_device = f x_device in
  Rune_pjrt.Device_buffer.await actual_device;
  let actual = Rune_pjrt.Device_buffer.to_host actual_device in
  let expected = Nx.add (Nx.mul x x) (Nx.sin x) in
  check_close "jit_cuda_device_executes" expected actual

let test_jit_cuda_device_chains () =
  let x = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let x_device = Rune_pjrt.Device_buffer.of_host x in
  let add_one =
    Rune_pjrt.jit_device (fun t -> Nx.add t (Nx.scalar_like t 1.0))
  in
  let square = Rune_pjrt.jit_device (fun t -> Nx.mul t t) in
  let actual_device = square (add_one x_device) in
  let actual = Rune_pjrt.Device_buffer.to_host actual_device in
  let expected =
    let incremented = Nx.add x (Nx.scalar_like x 1.0) in
    Nx.mul incremented incremented
  in
  check_close "jit_cuda_device_chains" expected actual

let test_jits_cuda_device_multiple_buffers () =
  let lhs = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let rhs = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let compiled =
    Rune_pjrt.jits_device (function
      | [ x; y ] -> [ Nx.add x y; Nx.mul x y ]
      | _ -> failwith "expected two inputs")
  in
  let outputs =
    compiled
      [
        Rune_pjrt.Device_buffer.of_host lhs; Rune_pjrt.Device_buffer.of_host rhs;
      ]
  in
  match outputs with
  | [ sum; product ] ->
      check_close "jits_cuda_device_sum" (Nx.add lhs rhs)
        (Rune_pjrt.Device_buffer.to_host sum);
      check_close "jits_cuda_device_product" (Nx.mul lhs rhs)
        (Rune_pjrt.Device_buffer.to_host product)
  | _ -> failwith "jits_cuda_device_multiple_buffers: wrong output arity"

let test_jit_cuda_device_shape_cache () =
  let compiled =
    Rune_pjrt.jit_device (fun input -> Nx.add input (Nx.scalar_like input 2.0))
  in
  let check values =
    let input = Nx.create Nx.float32 [| Array.length values |] values in
    let actual =
      input |> Rune_pjrt.Device_buffer.of_host |> compiled
      |> Rune_pjrt.Device_buffer.to_host
    in
    check_close "jit_cuda_device_shape_cache"
      (Nx.add input (Nx.scalar_like input 2.0))
      actual
  in
  check [| 1.; 2. |];
  check [| 3.; 4.; 5. |];
  check [| 6.; 7. |]

let () =
  test_runtime_status ();
  test_sigmoid_constants_remain_scalar ();
  if backend_available `Cpu then (
    test_jit_cpu_executes ();
    test_jit_cpu_argmax_executes ();
    test_rune_jit_pjrt_cpu_executes ();
    test_jit_erf_executes `Cpu);
  let cuda_available = backend_available `Cuda in
  if require_cuda () && not cuda_available then
    failwith ("test_runtime: " ^ Rune_pjrt.Runtime.status ());
  if cuda_available then (
    test_jit_cuda_executes ();
    test_jits_cuda_host_multiple_inputs ();
    test_jit_cuda_device_executes ();
    test_jit_cuda_device_chains ();
    test_jits_cuda_device_multiple_buffers ();
    test_jit_cuda_device_shape_cache ();
    test_jit_erf_executes `Cuda)
