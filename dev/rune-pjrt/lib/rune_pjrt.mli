module Backend = Backend
module Causal_lm = Causal_lm
module Error = Error
module Ir = Ir
module Runtime = Runtime
module Signature = Signature
module Stablehlo = Stablehlo
module Trace = Trace

type packed = Trace.packed = Tensor : ('a, 'b) Nx.t -> packed

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
