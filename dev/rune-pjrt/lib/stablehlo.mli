type lowered = {
  body : string list;
  outputs : (string * Ir.desc) list;
}

val tensor_type : Ir.desc -> string

val lower_program :
  ?indent:string -> arg_name:(int -> string) -> Ir.program -> lowered

val of_program : Ir.program -> string
