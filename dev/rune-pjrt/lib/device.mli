type t
(** PJRT execution device descriptor. *)

val create : ?device_id:int -> Backend.t -> t
(** [create ?device_id backend] selects [backend] and PJRT device index
    [device_id].

    [device_id] defaults to [0]. *)

val cpu : ?device_id:int -> unit -> t
(** [cpu ?device_id ()] selects the PJRT CPU backend. *)

val cuda : ?device_id:int -> unit -> t
(** [cuda ?device_id ()] selects the PJRT CUDA backend. *)

val backend : t -> Backend.t
(** [backend t] is the selected PJRT backend. *)

val device_id : t -> int
(** [device_id t] is the selected PJRT device index. *)
