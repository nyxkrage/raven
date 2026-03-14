type packed = Tensor : ('a, 'b) Nx.t -> packed

type capture = {
  program : Ir.program;
  outputs : packed list;
}

val capture_many :
  ?name:string ->
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t list) ->
  ('a, 'b) Nx.t list ->
  capture

val capture_one :
  ?name:string ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  capture
