type t = Tolk_cpu | Pjrt_cpu | Pjrt_cuda

val all : t list
val default : t
val to_string : t -> string
val of_string : string -> t
val of_env : ?var:string -> ?default:t -> unit -> t
val pjrt_device_id_of_env : ?var:string -> ?default:int -> unit -> int
val available : t -> bool
val require : t -> unit
val device : ?tolk_name:string -> ?pjrt_device_id:int -> t -> Jit.Device.t
