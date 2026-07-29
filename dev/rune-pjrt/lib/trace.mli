type packed = Tensor : ('a, 'b) Nx.t -> packed

type capture = {
  program : Ir.program;
  outputs : packed list;
}
(** A captured program and output shape/dtype witnesses. The tensor contents in
    [outputs] are unspecified; a captured function must not inspect tensor data
    or branch on it while tracing. *)

val capture_many :
  ?name:string ->
  ?enable_ffi:bool ->
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t list) ->
  ('a, 'b) Nx.t list ->
  capture

val capture_one :
  ?name:string ->
  ?enable_ffi:bool ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  capture
