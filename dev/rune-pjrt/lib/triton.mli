(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type packed = Tensor : ('a, 'b) Nx.t -> packed

module Kernel : sig
  type t

  (** [create ~name ~ir ?num_warps ?num_stages ?grid ()] creates a Triton
      kernel compiled by the XLA CUDA backend. [ir] is a complete TTIR module
      whose public function is [name]. Its pointer arguments are the packed
      inputs in order followed by the output. *)
  val create :
    name:string ->
    ir:string ->
    ?num_warps:int ->
    ?num_stages:int ->
    ?grid:int * int * int ->
    unit ->
    t
end

(** [call kernel ~inputs ~fallback] uses [kernel] while tracing for PJRT CUDA
    and otherwise evaluates [fallback]. *)
val call :
  Kernel.t ->
  inputs:packed list ->
  fallback:(unit -> ('a, 'b) Nx.t) ->
  ('a, 'b) Nx.t

module Internal : sig
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
