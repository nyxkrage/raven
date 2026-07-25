(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type packed = Tensor : ('a, 'b) Nx.t -> packed

module Internal = struct
  type kernel = {
    name : string;
    ir : string;
    num_warps : int;
    num_stages : int;
    grid_x : int;
    grid_y : int;
    grid_z : int;
  }

  type ('a, 'b) request = {
    kernel : kernel;
    inputs : packed list;
    fallback : unit -> ('a, 'b) Nx.t;
  }

  type ('a, 'b) decision = Use_kernel of ('a, 'b) Nx.t | Use_fallback
  type _ Effect.t += Call : ('a, 'b) request -> ('a, 'b) decision Effect.t
end

module Kernel = struct
  type t = Internal.kernel

  let positive name value =
    if value <= 0 then
      invalid_arg
        (Printf.sprintf "Rune_pjrt.Triton.Kernel.create: %s must be positive"
           name)

  let create ~name ~ir ?(num_warps = 4) ?(num_stages = 3)
      ?(grid = (1, 1, 1)) () =
    if String.trim name = "" then
      invalid_arg "Rune_pjrt.Triton.Kernel.create: name must not be empty";
    if String.trim ir = "" then
      invalid_arg "Rune_pjrt.Triton.Kernel.create: ir must not be empty";
    positive "num_warps" num_warps;
    if num_warps land (num_warps - 1) <> 0 then
      invalid_arg
        "Rune_pjrt.Triton.Kernel.create: num_warps must be a power of two";
    positive "num_stages" num_stages;
    let grid_x, grid_y, grid_z = grid in
    positive "grid x" grid_x;
    positive "grid y" grid_y;
    positive "grid z" grid_z;
    Internal.
      { name; ir; num_warps; num_stages; grid_x; grid_y; grid_z }
end

let call kernel ~inputs ~fallback =
  let decision =
    try Effect.perform (Internal.Call { kernel; inputs; fallback })
    with Effect.Unhandled (Internal.Call _) -> Internal.Use_fallback
  in
  match decision with
  | Internal.Use_kernel output -> output
  | Internal.Use_fallback -> fallback ()
