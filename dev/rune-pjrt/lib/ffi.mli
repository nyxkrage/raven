type packed = Tensor : ('a, 'b) Nx.t -> packed

module Kernel : sig
  type t

  val create : library:string -> ?fwd:string -> ?bwd:string -> unit -> t
  (** [create ~library ?fwd ?bwd ()] describes typed XLA FFI handlers exported
      by [library]. At least one handler symbol must be provided. A relative
      [library] path is resolved from the executable's directory. The file is
      opened only when a PJRT CUDA trace selects one of its handlers. *)
end

val call_fwd :
  Kernel.t ->
  inputs:packed list ->
  fallback:(unit -> ('a, 'b) Nx.t) ->
  ('a, 'b) Nx.t
(** [call_fwd kernel ~inputs ~fallback] calls the forward handler while tracing
    for PJRT CUDA. It evaluates [fallback] when no forward handler is present or
    outside a PJRT trace. *)

val call_bwd :
  Kernel.t ->
  inputs:packed list ->
  fallback:(unit -> ('a, 'b) Nx.t) ->
  ('a, 'b) Nx.t
(** [call_bwd kernel ~inputs ~fallback] calls the backward handler while tracing
    for PJRT CUDA. It evaluates [fallback] when no backward handler is present
    or outside a PJRT trace. *)

module Internal : sig
  type handler = { library : string; symbol : string }
  type identity = { library : string; library_digest : string; target : string }

  val identity : handler -> identity
  val target : handler -> string

  type ('a, 'b) request = {
    handler : handler;
    inputs : packed list;
    fallback : unit -> ('a, 'b) Nx.t;
  }

  type ('a, 'b) decision = Use_kernel of ('a, 'b) Nx.t | Use_fallback
  type _ Effect.t += Call : ('a, 'b) request -> ('a, 'b) decision Effect.t
end
