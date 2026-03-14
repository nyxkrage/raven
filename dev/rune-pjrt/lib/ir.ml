(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

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

let desc_of_tensor (type a b) (t : (a, b) Nx.t) =
  { shape = Array.copy (Nx.shape t); dtype = Dtype.to_string (Nx.dtype t) }

let literal_of_tensor (type a b) (t : (a, b) Nx.t) =
  let buffer =
    let t = if Nx.is_c_contiguous t && Nx.offset t = 0 then t else Nx.contiguous t in
    let buffer = Nx.data t in
    if Nx_buffer.length buffer = Nx.numel t then buffer else Nx.data (Nx.copy t)
  in
  Literal { dtype = Nx.dtype t; shape = Array.copy (Nx.shape t); buffer }

let literal_size_bytes (Literal { dtype; buffer; _ }) =
  Nx_buffer.length buffer * Dtype.itemsize dtype

let operands = function
  | Parameter _ | Constant _ | Buffer _ | Unsupported _ -> []
  | Unary { input; _ } -> [ input ]
  | Binary { lhs; rhs; _ } -> [ lhs; rhs ]
  | Where { condition; if_true; if_false } -> [ condition; if_true; if_false ]
  | Reduce { input; _ }
  | Arg_reduce { input; _ }
  | Reshape { input; _ }
  | Expand { input; _ }
  | Permute { input; _ }
  | Shrink { input; _ }
  | Flip { input; _ }
  | Pad { input; _ }
  | Cast { input; _ } ->
      [ input ]
  | Cat { inputs; _ } -> inputs
  | Gather { data; indices; _ } -> [ data; indices ]
  | Matmul { lhs; rhs } -> [ lhs; rhs ]
  | Assign { dst; src } -> [ dst; src ]

let unary_name = function
  | Neg -> "neg"
  | Sin -> "sin"
  | Sqrt -> "sqrt"
  | Recip -> "recip"
  | Log -> "log"
  | Exp -> "exp"
  | Cos -> "cos"
  | Abs -> "abs"
  | Sign -> "sign"
  | Tan -> "tan"
  | Asin -> "asin"
  | Acos -> "acos"
  | Atan -> "atan"
  | Sinh -> "sinh"
  | Cosh -> "cosh"
  | Tanh -> "tanh"
  | Trunc -> "trunc"
  | Ceil -> "ceil"
  | Floor -> "floor"
  | Round -> "round"
  | Erf -> "erf"
  | Contiguous -> "contiguous"
  | Copy -> "copy"

let binary_name = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Idiv -> "idiv"
  | Fdiv -> "fdiv"
  | Max -> "max"
  | Min -> "min"
  | Mod -> "mod"
  | Pow -> "pow"
  | Xor -> "xor"
  | Or -> "or"
  | And -> "and"
  | Atan2 -> "atan2"
  | CmpEq -> "cmpeq"
  | CmpNe -> "cmpne"
  | CmpLt -> "cmplt"
  | CmpLe -> "cmple"

let reduce_name = function
  | Reduce_sum -> "reduce_sum"
  | Reduce_max -> "reduce_max"
  | Reduce_min -> "reduce_min"
  | Reduce_prod -> "reduce_prod"

let arg_reduce_name = function Argmax -> "argmax" | Argmin -> "argmin"

let op_name = function
  | Parameter i -> Printf.sprintf "parameter[%d]" i
  | Constant _ -> "constant"
  | Buffer _ -> "buffer"
  | Unary { op; _ } -> unary_name op
  | Binary { op; _ } -> binary_name op
  | Where _ -> "where"
  | Reduce { op; _ } -> reduce_name op
  | Arg_reduce { op; _ } -> arg_reduce_name op
  | Reshape _ -> "reshape"
  | Expand _ -> "expand"
  | Permute _ -> "permute"
  | Shrink _ -> "shrink"
  | Flip _ -> "flip"
  | Pad _ -> "pad"
  | Cat _ -> "cat"
  | Cast _ -> "cast"
  | Gather _ -> "gather"
  | Matmul _ -> "matmul"
  | Assign _ -> "assign"
  | Unsupported name -> name

let parameters (program : program) =
  program.nodes
  |> List.filter_map (fun (node : node) ->
         match node.op with
         | Parameter index -> Some (index, node.id, node.desc)
         | _ -> None)
  |> List.sort (fun (a, _, _) (b, _, _) -> Int.compare a b)

let parameterize_constants ?(min_bytes = 4096) (program : program) =
  let next_param =
    parameters program
    |> List.fold_left (fun acc (index, _, _) -> max acc (index + 1)) 0
  in
  let next_param = ref next_param in
  let lifted_rev = ref [] in
  let nodes =
    List.map
      (fun (node : node) ->
        match node.op with
        | Constant literal when literal_size_bytes literal >= min_bytes ->
            let index = !next_param in
            incr next_param;
            lifted_rev :=
              { index; id = node.id; desc = node.desc; literal } :: !lifted_rev;
            { node with op = Parameter index }
        | _ -> node)
      program.nodes
  in
  let lifted = List.rev !lifted_rev in
  let inputs = program.inputs @ List.map (fun lifted -> lifted.id) lifted in
  ({ program with inputs; nodes }, lifted)

let prune program =
  let module Int_set = Set.Make (Int) in
  let by_id = Hashtbl.create (List.length program.nodes) in
  List.iter
    (fun (node : node) -> Hashtbl.replace by_id node.id node)
    program.nodes;
  let rec visit seen id =
    if Int_set.mem id seen then seen
    else
      let seen = Int_set.add id seen in
      match Hashtbl.find_opt by_id id with
      | None -> seen
      | Some (node : node) -> List.fold_left visit seen (operands node.op)
  in
  let reachable = List.fold_left visit Int_set.empty program.outputs in
  let nodes =
    List.filter (fun (node : node) -> Int_set.mem node.id reachable) program.nodes
    |> List.sort (fun (a : node) (b : node) -> Int.compare a.id b.id)
  in
  { program with nodes }

let unsupported_ops (program : program) =
  program.nodes
  |> List.filter_map (fun (node : node) ->
         match node.op with
         | Unsupported name -> Some name
         | Assign _ -> Some "assign"
         | Buffer _ -> Some "buffer"
         | _ -> None)
  |> List.sort_uniq String.compare

let pp_shape ppf shape = Format.fprintf ppf "%s" (Shape.to_string shape)

let pp_literal ppf (Literal { dtype; shape; buffer }) =
  let bytes = Nx_buffer.length buffer * Dtype.itemsize dtype in
  Format.fprintf ppf "<%s%s %dB>" (Dtype.to_string dtype)
    (Shape.to_string shape) bytes

let pp_node ppf (node : node) =
  let desc = node.desc in
  let pp_inputs ids =
    ids
    |> List.map (fun id -> Printf.sprintf "%%%d" id)
    |> String.concat ", "
  in
  let pp_fill_value ppf fill = Format.pp_print_string ppf fill in
  Format.fprintf ppf "%%%d : %s%s = " node.id desc.dtype
    (Shape.to_string desc.shape);
  match node.op with
  | Parameter index -> Format.fprintf ppf "parameter[%d]" index
  | Constant literal -> pp_literal ppf literal
  | Buffer { size_in_elements } -> Format.fprintf ppf "buffer[%d]" size_in_elements
  | Unary { op; input } ->
      Format.fprintf ppf "%s(%%%d)" (unary_name op) input
  | Binary { op; lhs; rhs } ->
      Format.fprintf ppf "%s(%%%d, %%%d)" (binary_name op) lhs rhs
  | Where { condition; if_true; if_false } ->
      Format.fprintf ppf "where(%%%d, %%%d, %%%d)" condition if_true if_false
  | Reduce { op; input; axes; keepdims } ->
      Format.fprintf ppf "%s(%%%d, axes=%s, keepdims=%b)" (reduce_name op) input
        (Shape.to_string axes) keepdims
  | Arg_reduce { op; input; axis; keepdims } ->
      Format.fprintf ppf "%s(%%%d, axis=%d, keepdims=%b)" (arg_reduce_name op)
        input axis keepdims
  | Reshape { input; shape } ->
      Format.fprintf ppf "reshape(%%%d -> %a)" input pp_shape shape
  | Expand { input; shape } ->
      Format.fprintf ppf "expand(%%%d -> %a)" input pp_shape shape
  | Permute { input; axes } ->
      Format.fprintf ppf "permute(%%%d, axes=%a)" input pp_shape axes
  | Shrink { input; limits } ->
      let items =
        Array.to_list limits
        |> List.map (fun (lo, hi) -> Printf.sprintf "[%d,%d)" lo hi)
        |> String.concat ", "
      in
      Format.fprintf ppf "shrink(%%%d, %s)" input items
  | Flip { input; dims } ->
      let dims =
        Array.to_list dims
        |> List.map string_of_bool
        |> String.concat ", "
      in
      Format.fprintf ppf "flip(%%%d, dims=[%s])" input dims
  | Pad { input; padding; fill_value } ->
      let items =
        Array.to_list padding
        |> List.map (fun (lo, hi) -> Printf.sprintf "(%d,%d)" lo hi)
        |> String.concat ", "
      in
      Format.fprintf ppf "pad(%%%d, [%s], fill=%a)" input items pp_fill_value
        fill_value
  | Cat { inputs; axis } ->
      Format.fprintf ppf "cat([%s], axis=%d)" (pp_inputs inputs) axis
  | Cast { input; dtype } ->
      Format.fprintf ppf "cast(%%%d -> %s)" input dtype
  | Gather { data; indices; axis } ->
      Format.fprintf ppf "gather(%%%d, %%%d, axis=%d)" data indices axis
  | Matmul { lhs; rhs } -> Format.fprintf ppf "matmul(%%%d, %%%d)" lhs rhs
  | Assign { dst; src } -> Format.fprintf ppf "assign(%%%d, %%%d)" dst src
  | Unsupported name -> Format.fprintf ppf "unsupported(%s)" name

let pp_program ppf program =
  let program = prune program in
  Format.fprintf ppf "@[<v>";
  (match program.name with
  | None -> Format.fprintf ppf "program"
  | Some name -> Format.fprintf ppf "program %s" name);
  Format.fprintf ppf "@,inputs: %s"
    (String.concat ", " (List.map (fun id -> Printf.sprintf "%%%d" id) program.inputs));
  Format.fprintf ppf "@,outputs: %s"
    (String.concat ", " (List.map (fun id -> Printf.sprintf "%%%d" id) program.outputs));
  List.iter (fun node -> Format.fprintf ppf "@,%a" pp_node node) program.nodes;
  Format.fprintf ppf "@]"

let program_to_string program = Format.asprintf "%a@." pp_program program
