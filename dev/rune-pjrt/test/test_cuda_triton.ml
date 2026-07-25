(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let ttir =
  {|
module {
  tt.func public @raven_add_one(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>) {
    %value = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    %one = arith.constant 1.000000e+00 : f32
    %result = arith.addf %value, %one : f32
    tt.store %arg1, %result {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
    tt.return
  }
}
|}

let kernel =
  Rune_pjrt.Triton.Kernel.create ~name:"raven_add_one" ~ir:ttir
    ~num_warps:1 ~num_stages:1 ()

let add_one x =
  Rune_pjrt.Triton.call kernel
    ~inputs:[ Rune_pjrt.Triton.Tensor x ]
    ~fallback:(fun () -> Nx.add x (Nx.scalar_like x 1.0))

let () =
  if Rune_pjrt.backend_available `Cuda then (
    let x = Nx.scalar Nx.float32 41.0 in
    let actual = Rune_pjrt.jit ~backend:`Cuda add_one x |> Nx.item [] in
    if Float.abs (actual -. 42.0) > 1e-6 then
      failwith
        (Printf.sprintf "test_cuda_triton: expected 42, got %.9g" actual);
    Printf.printf "test_cuda_triton: XLA Triton kernel returned %.9g\n%!" actual)
  else Printf.printf "test_cuda_triton: CUDA plugin unavailable, skipping\n%!"
