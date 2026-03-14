open Nx_core

type node_id = int

type desc = {
  shape : int array;
  dtype : string;
}

type literal = Literal : {
  dtype : ('a, 'b) Dtype.t;
  shape : int array;
  buffer : ('a, 'b) Nx_buffer.t;
} -> literal

type unary =
  | Neg
  | Sin
  | Sqrt
  | Recip
  | Log
  | Exp
  | Cos
  | Abs
  | Sign
  | Tan
  | Asin
  | Acos
  | Atan
  | Sinh
  | Cosh
  | Tanh
  | Trunc
  | Ceil
  | Floor
  | Round
  | Erf
  | Contiguous
  | Copy

type binary =
  | Add
  | Sub
  | Mul
  | Idiv
  | Fdiv
  | Max
  | Min
  | Mod
  | Pow
  | Xor
  | Or
  | And
  | Atan2
  | CmpEq
  | CmpNe
  | CmpLt
  | CmpLe

type reduce = Reduce_sum | Reduce_max | Reduce_min | Reduce_prod
type arg_reduce = Argmax | Argmin

type op =
  | Parameter of int
  | Constant of literal
  | Buffer of { size_in_elements : int }
  | Unary of { op : unary; input : node_id }
  | Binary of { op : binary; lhs : node_id; rhs : node_id }
  | Where of { condition : node_id; if_true : node_id; if_false : node_id }
  | Reduce of {
      op : reduce;
      input : node_id;
      axes : int array;
      keepdims : bool;
    }
  | Arg_reduce of {
      op : arg_reduce;
      input : node_id;
      axis : int;
      keepdims : bool;
    }
  | Reshape of { input : node_id; shape : int array }
  | Expand of { input : node_id; shape : int array }
  | Permute of { input : node_id; axes : int array }
  | Shrink of { input : node_id; limits : (int * int) array }
  | Flip of { input : node_id; dims : bool array }
  | Pad of {
      input : node_id;
      padding : (int * int) array;
      fill_value : string;
    }
  | Cat of { inputs : node_id list; axis : int }
  | Cast of { input : node_id; dtype : string }
  | Gather of { data : node_id; indices : node_id; axis : int }
  | Matmul of { lhs : node_id; rhs : node_id }
  | Assign of { dst : node_id; src : node_id }
  | Unsupported of string

type node = {
  id : node_id;
  desc : desc;
  op : op;
}

type program = {
  name : string option;
  inputs : node_id list;
  outputs : node_id list;
  nodes : node list;
}

type lifted_constant = {
  index : int;
  id : node_id;
  desc : desc;
  literal : literal;
}

val desc_of_tensor : ('a, 'b) Nx.t -> desc
val literal_of_tensor : ('a, 'b) Nx.t -> literal
val operands : op -> node_id list
val op_name : op -> string
val parameters : program -> (int * node_id * desc) list
val parameterize_constants :
  ?min_bytes:int -> program -> program * lifted_constant list
val prune : program -> program
val unsupported_ops : program -> string list
val pp_program : Format.formatter -> program -> unit
val program_to_string : program -> string
