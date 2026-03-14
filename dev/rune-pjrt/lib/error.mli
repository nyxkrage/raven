type t =
  | Unsupported_effect of string
  | Unsupported_op of string
  | Unsupported_program of string
  | Runtime_unavailable of string

exception Error of t

val raise : t -> 'a
val to_string : t -> string
