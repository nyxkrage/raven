type tensor = {
  shape : int array;
  dtype : string;
}

type t = {
  backend : Backend.t;
  device_id : int;
  inputs : tensor list;
}

val of_packed :
  backend:Backend.t -> device_id:int -> Trace.packed list -> t

val of_tensors :
  backend:Backend.t -> device_id:int -> ('a, 'b) Nx.t list -> t

val key : t -> string
