module Backend = Backend
module Causal_lm = Causal_lm
module Device = Device
module Error = Error
module Ffi = Ffi
module Grouped_gemm = Grouped_gemm
module Ir = Ir
module Runtime = Runtime
module Signature = Signature
module Stablehlo = Stablehlo
module Trace = Trace
module Triton = Triton

type packed = Trace.packed = Tensor : ('a, 'b) Nx.t -> packed

module Device_buffer : sig
  type ('a, 'b) t
  type packed = Pack : ('a, 'b) t -> packed

  val of_host :
    ?backend:Backend.t -> ?device_id:int -> ('a, 'b) Nx.t -> ('a, 'b) t
  (** [of_host tensor] transfers [tensor] to a PJRT device. *)

  val to_host : ('a, 'b) t -> ('a, 'b) Nx.t
  (** [to_host buffer] transfers [buffer] to an Nx tensor on the host. *)

  val await : ('a, 'b) t -> unit
  (** [await buffer] waits until the computation producing [buffer] finishes. *)

  val shape : ('a, 'b) t -> int array
  (** [shape buffer] is a copy of [buffer]'s shape. *)

  val dtype : ('a, 'b) t -> ('a, 'b) Nx_core.Dtype.t
  (** [dtype buffer] is [buffer]'s element type. *)
end

val backend_available : Backend.t -> bool
val status : unit -> string

val jit :
  ?backend:Backend.t ->
  ?device_id:int ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t

val jits_packed :
  ?backend:Backend.t ->
  ?device_id:int ->
  (packed list -> packed list) ->
  packed list ->
  packed list

val jits :
  ?backend:Backend.t ->
  ?device_id:int ->
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t list) ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t list

val jit_device :
  ?backend:Backend.t ->
  ?device_id:int ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Device_buffer.t ->
  ('c, 'd) Device_buffer.t
(** [jit_device f] compiles [f] and keeps its inputs and output on the PJRT
    device across calls. *)

val jits_device_packed :
  ?backend:Backend.t ->
  ?device_id:int ->
  (packed list -> packed list) ->
  Device_buffer.packed list ->
  Device_buffer.packed list
(** [jits_device_packed f] is the heterogeneous, multi-input and multi-output
    device-resident form of {!jit_device}. *)

val jits_device :
  ?backend:Backend.t ->
  ?device_id:int ->
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t list) ->
  ('a, 'b) Device_buffer.t list ->
  ('c, 'd) Device_buffer.t list
(** [jits_device f] is the homogeneous, multi-input and multi-output
    device-resident form of {!jit_device}. *)
