(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core

let unsupported name =
  Error.raise (Error.Unsupported_program ("StableHLO lowering missing for " ^ name))

let find_node program id =
  match
    List.find_opt
      (fun (node : Ir.node) -> node.Ir.id = id)
      program.Ir.nodes
  with
  | Some node -> node
  | None ->
      Error.raise
        (Error.Unsupported_program
           (Printf.sprintf "missing node %d while lowering StableHLO" id))

let tensor_type (desc : Ir.desc) =
  let elt =
    match desc.dtype with
    | "float16" -> "f16"
    | "float32" -> "f32"
    | "float64" -> "f64"
    | "int8" -> "i8"
    | "uint8" -> "ui8"
    | "int16" -> "i16"
    | "uint16" -> "ui16"
    | "int32" -> "i32"
    | "uint32" -> "ui32"
    | "int64" -> "i64"
    | "uint64" -> "ui64"
    | "bool" -> "i1"
    | dtype ->
        Error.raise
          (Error.Unsupported_program
             (Printf.sprintf "StableHLO lowering unsupported dtype %s" dtype))
  in
  match Array.to_list desc.shape with
  | [] -> Printf.sprintf "tensor<%s>" elt
  | dims ->
      Printf.sprintf "tensor<%sx%s>"
        (String.concat "x" (List.map string_of_int dims))
        elt

let scalar_literal (type a b) (dtype : (a, b) Dtype.t) (value : a) =
  match dtype with
  | Float16 | Float32 | BFloat16 | Float8_e4m3 | Float8_e5m2 ->
      Printf.sprintf "%.9e" (Obj.magic value : float)
  | Float64 -> Printf.sprintf "%.17e" (Obj.magic value : float)
  | Int4 | UInt4 | Int8 | UInt8 | Int16 | UInt16 ->
      string_of_int (Obj.magic value : int)
  | Int32 | UInt32 -> Int32.to_string (Obj.magic value : int32)
  | Int64 | UInt64 -> Int64.to_string (Obj.magic value : int64)
  | Bool -> if (Obj.magic value : bool) then "true" else "false"
  | Complex64 | Complex128 ->
      unsupported ("complex constant " ^ Dtype.to_string dtype)

let all_same first rest =
  List.for_all (String.equal first) rest

let literal_data (Ir.Literal { dtype; shape; buffer }) =
  let len = Nx_buffer.length buffer in
  let rec build_values i acc =
    if i = len then List.rev acc
    else
      let value = scalar_literal dtype (Nx_buffer.unsafe_get buffer i) in
      build_values (i + 1) (value :: acc)
  in
  let values = build_values 0 [] in
  let rec nest dims values =
    match dims with
    | [] -> (
        match values with
        | x :: rest -> (x, rest)
        | _ ->
            Error.raise
              (Error.Unsupported_program
                 (Printf.sprintf
                    "StableHLO literal shape/value mismatch at scalar leaf \
                     (shape=%s, remaining=%d)"
                    (Shape.to_string shape) (List.length values))))
    | dim :: rest ->
        let rec loop n values acc =
          if n = 0 then ("[" ^ String.concat ", " (List.rev acc) ^ "]", values)
          else
            let rendered, values = nest rest values in
            loop (n - 1) values (rendered :: acc)
        in
        loop dim values []
  in
  if len = 0 then "dense<[]>"
  else
    match values with
    | first :: rest when all_same first rest -> Printf.sprintf "dense<%s>" first
    | _ ->
        let rendered, remaining = nest (Array.to_list shape) values in
        if remaining <> [] then
          Error.raise
            (Error.Unsupported_program
               (Printf.sprintf
                  "StableHLO literal shape/value mismatch after nesting \
                   (shape=%s, remaining=%d)"
                  (Shape.to_string shape) (List.length remaining)));
        Printf.sprintf "dense<%s>" rendered

let pp_int_array arr =
  Array.to_list arr |> List.map string_of_int |> String.concat ", "

let pp_i64_array arr =
  Array.to_list arr |> List.map string_of_int |> String.concat ", "

let pp_bool_array arr =
  arr
  |> Array.to_list
  |> List.mapi (fun i flag -> if flag then Some (string_of_int i) else None)
  |> List.filter_map Fun.id |> String.concat ", "

let pp_type_list descs =
  descs |> List.map tensor_type |> String.concat ", "

let output_type program =
  let descs = List.map (fun id -> (find_node program id).Ir.desc) program.Ir.outputs in
  match descs with
  | [ desc ] -> tensor_type desc
  | _ -> "(" ^ pp_type_list descs ^ ")"

type lowered = {
  body : string list;
  outputs : (string * Ir.desc) list;
}

let output_return ~indent lowered =
  let values, descs = List.split lowered.outputs in
  match values with
  | [ value ] ->
      Printf.sprintf "%sfunc.return %s : %s" indent value
        (tensor_type (List.hd descs))
  | _ ->
      Printf.sprintf "%sfunc.return %s : %s" indent
        (String.concat ", " values)
        (pp_type_list descs)

let parameter_nodes program =
  program.Ir.nodes
  |> List.filter_map (fun node ->
         match node.Ir.op with
         | Ir.Parameter index -> Some (index, node.Ir.desc)
         | _ -> None)
  |> List.sort (fun (a, _) (b, _) -> Int.compare a b)

let op_ref ~arg_name program id =
  match (find_node program id).Ir.op with
  | Ir.Parameter index -> arg_name index
  | _ -> Printf.sprintf "%%v%d" id

let dims_attr arr = "[" ^ pp_int_array arr ^ "]"

let lower_unary op =
  match op with
  | Ir.Neg -> "stablehlo.negate"
  | Sin -> "stablehlo.sine"
  | Sqrt -> "stablehlo.sqrt"
  | Log -> "stablehlo.log"
  | Exp -> "stablehlo.exponential"
  | Cos -> "stablehlo.cosine"
  | Abs -> "stablehlo.abs"
  | Tanh -> "stablehlo.tanh"
  | Contiguous | Copy | Recip -> "identity"
  | op -> unsupported (Ir.op_name (Ir.Unary { op; input = 0 }))

let lower_binary op =
  match op with
  | Ir.Add -> "stablehlo.add"
  | Sub -> "stablehlo.subtract"
  | Mul -> "stablehlo.multiply"
  | Fdiv -> "stablehlo.divide"
  | Max -> "stablehlo.maximum"
  | Min -> "stablehlo.minimum"
  | op -> unsupported (Ir.op_name (Ir.Binary { op; lhs = 0; rhs = 0 }))

let compare_direction = function
  | Ir.CmpEq -> "EQ"
  | Ir.CmpNe -> "NE"
  | Ir.CmpLt -> "LT"
  | Ir.CmpLe -> "LE"
  | op -> unsupported (Ir.op_name (Ir.Binary { op; lhs = 0; rhs = 0 }))

let scalar_one_literal dtype =
  match dtype with
  | "float16" -> "1"
  | "float32" -> "1.000000e+00"
  | "float64" -> "1.0000000000000000e+00"
  | dtype ->
      Error.raise
        (Error.Unsupported_program
           (Printf.sprintf "StableHLO reciprocal unsupported dtype %s" dtype))

let reduce_init_literal op dtype =
  match (op, dtype) with
  | Ir.Reduce_sum, ("float16" | "float32" | "float64") -> "0.000000e+00"
  | Ir.Reduce_sum, ("int8" | "int16" | "int32" | "int64") -> "0"
  | Ir.Reduce_sum, ("uint8" | "uint16" | "uint32" | "uint64") -> "0"
  | Ir.Reduce_sum, "bool" -> "false"
  | Ir.Reduce_max, "float16" -> "0xFC00"
  | Ir.Reduce_max, "float32" -> "0xFF800000"
  | Ir.Reduce_max, "float64" -> "0xFFF0000000000000"
  | Ir.Reduce_max, "int8" -> "-128"
  | Ir.Reduce_max, "int16" -> "-32768"
  | Ir.Reduce_max, "int32" -> "-2147483648"
  | Ir.Reduce_max, "int64" -> "-9223372036854775808"
  | Ir.Reduce_max, ("uint8" | "uint16" | "uint32" | "uint64" | "bool") -> "0"
  | op, dtype ->
      Error.raise
        (Error.Unsupported_program
           (Printf.sprintf "StableHLO %s init unsupported dtype %s"
              (Ir.op_name
                 (Ir.Reduce { op; input = 0; axes = [||]; keepdims = false }))
              dtype))

let reduce_combiner = function
  | Ir.Reduce_sum -> "stablehlo.add"
  | Reduce_max -> "stablehlo.maximum"
  | op ->
      unsupported
        (Ir.op_name
           (Ir.Reduce { op; input = 0; axes = [||]; keepdims = false }))

let expand_dims input_shape output_shape =
  let rank_in = Array.length input_shape in
  let rank_out = Array.length output_shape in
  Array.init rank_in (fun i -> (rank_out - rank_in) + i)

let slice_sizes_attr arr = "array<i64: " ^ pp_i64_array arr ^ ">"

let reverse_dims_attr dims = "array<i64: " ^ pp_i64_array dims ^ ">"

let reduced_shape input_shape axes =
  input_shape
  |> Array.to_list
  |> List.mapi (fun i dim -> (i, dim))
  |> List.filter_map (fun (i, dim) ->
         if Array.exists (( = ) i) axes then None else Some dim)
  |> Array.of_list

let gather_offset_dims ~axis ~indices_rank ~data_rank =
  Array.init (data_rank - 1) (fun i ->
      if i < axis then i else indices_rank + i)

let slice_spec limits =
  limits
  |> Array.to_list
  |> List.map (fun (lo, hi) -> Printf.sprintf "%d:%d" lo hi)
  |> String.concat ", "

let rec strip_axis0_embedding_indices program id =
  let node = find_node program id in
  match node.Ir.op with
  | Ir.Expand { input; _ } -> strip_axis0_embedding_indices program input
  | Reshape { input; shape } when Array.length shape = 2 && shape.(1) = 1 ->
      strip_axis0_embedding_indices program input
  | _ -> id

let lower_constant ~indent (node : Ir.node) =
  match node.Ir.op with
  | Ir.Constant literal ->
      Printf.sprintf "%s%%v%d = stablehlo.constant %s : %s" indent node.Ir.id
        (literal_data literal) (tensor_type node.Ir.desc)
  | _ -> assert false

let lower_node ~indent ~arg_name program (node : Ir.node) =
  let ty = tensor_type node.Ir.desc in
  match node.Ir.op with
  | Ir.Parameter _ -> None
  | Constant _ -> Some (lower_constant ~indent node)
  | Unary { op = Contiguous | Copy; input } ->
      let value = op_ref ~arg_name program input in
      if String.equal value (Printf.sprintf "%%v%d" node.Ir.id) then None
      else
        Some
          (Printf.sprintf "%s%%v%d = stablehlo.reshape %s : (%s) -> %s" indent
             node.Ir.id
             value ty ty)
  | Unary { op = Recip; input } ->
      let input_ty = tensor_type (find_node program input).Ir.desc in
      let one = scalar_one_literal node.Ir.desc.dtype in
      Some
        (String.concat "\n"
           [
             Printf.sprintf
               "%s%%v%d_one = stablehlo.constant dense<%s> : %s" indent
               node.Ir.id one input_ty;
             Printf.sprintf
               "%s%%v%d = stablehlo.divide %%v%d_one, %s : %s" indent
               node.Ir.id node.Ir.id (op_ref ~arg_name program input) input_ty;
           ])
  | Unary { op; input } ->
      Some
        (Printf.sprintf "%s%%v%d = %s %s : %s" indent node.Ir.id
           (lower_unary op) (op_ref ~arg_name program input) ty)
  | Binary { op = (CmpEq | CmpNe | CmpLt | CmpLe) as op; lhs; rhs } ->
      let lhs_ty = tensor_type (find_node program lhs).Ir.desc in
      let rhs_ty = tensor_type (find_node program rhs).Ir.desc in
      Some
        (Printf.sprintf
           "%s%%v%d = stablehlo.compare  %s, %s, %s : (%s, %s) -> %s" indent
           node.Ir.id (compare_direction op) (op_ref ~arg_name program lhs)
           (op_ref ~arg_name program rhs) lhs_ty rhs_ty ty)
  | Binary { op; lhs; rhs } ->
      Some
        (Printf.sprintf "%s%%v%d = %s %s, %s : %s" indent node.Ir.id
           (lower_binary op) (op_ref ~arg_name program lhs)
           (op_ref ~arg_name program rhs) ty)
  | Reshape { input; _ } ->
      let input_ty = tensor_type (find_node program input).Ir.desc in
      Some
        (Printf.sprintf "%s%%v%d = stablehlo.reshape %s : (%s) -> %s" indent
           node.Ir.id (op_ref ~arg_name program input) input_ty ty)
  | Expand { input; shape } ->
      let input_desc = (find_node program input).Ir.desc in
      let dims = expand_dims input_desc.shape shape in
      let input_ty = tensor_type input_desc in
      Some
        (Printf.sprintf
           "%s%%v%d = stablehlo.broadcast_in_dim %s, dims = %s : (%s) -> %s"
           indent node.Ir.id (op_ref ~arg_name program input) (dims_attr dims)
           input_ty ty)
  | Permute { input; axes } ->
      let input_ty = tensor_type (find_node program input).Ir.desc in
      Some
        (Printf.sprintf
           "%s%%v%d = stablehlo.transpose %s, dims = %s : (%s) -> %s" indent
           node.Ir.id (op_ref ~arg_name program input) (dims_attr axes)
           input_ty ty)
  | Cast { input; _ } ->
      let input_ty = tensor_type (find_node program input).Ir.desc in
      Some
        (Printf.sprintf "%s%%v%d = stablehlo.convert %s : (%s) -> %s" indent
           node.Ir.id (op_ref ~arg_name program input) input_ty ty)
  | Where { condition; if_true; if_false } ->
      let cond_ty = tensor_type (find_node program condition).Ir.desc in
      Some
        (Printf.sprintf "%s%%v%d = stablehlo.select %s, %s, %s : %s, %s"
           indent node.Ir.id (op_ref ~arg_name program condition)
           (op_ref ~arg_name program if_true)
           (op_ref ~arg_name program if_false) cond_ty ty)
  | Reduce { op; input; axes; keepdims } ->
      let combiner = reduce_combiner op in
      let init = reduce_init_literal op node.Ir.desc.dtype in
      let input_desc = (find_node program input).Ir.desc in
      let input_ty = tensor_type input_desc in
      let reduced_desc =
        { node.Ir.desc with shape = reduced_shape input_desc.shape axes }
      in
      let reduced_ty = tensor_type reduced_desc in
      let base =
        Printf.sprintf
          "%s%%v%d_raw = stablehlo.reduce(%s init: %%v%d_init) applies %s across \
           dimensions = %s : (%s, %s) -> %s"
          indent node.Ir.id (op_ref ~arg_name program input) node.Ir.id combiner
          (dims_attr axes)
          input_ty
          (tensor_type { shape = [||]; dtype = node.Ir.desc.dtype })
          reduced_ty
      in
      let init_line =
        Printf.sprintf "%s%%v%d_init = stablehlo.constant dense<%s> : %s"
          indent node.Ir.id init
          (tensor_type { shape = [||]; dtype = node.Ir.desc.dtype })
      in
      if keepdims then
        let reshape =
          Printf.sprintf
            "%s%%v%d = stablehlo.reshape %%v%d_raw : (%s) -> %s" indent
            node.Ir.id node.Ir.id reduced_ty ty
        in
        Some (String.concat "\n" [ init_line; base; reshape ])
      else
        Some
          (String.concat "\n"
             [
               init_line;
               base;
               Printf.sprintf
                 "%s%%v%d = stablehlo.reshape %%v%d_raw : (%s) -> %s" indent
                 node.Ir.id node.Ir.id reduced_ty ty;
             ])
  | Arg_reduce { op = Argmax; input; axis; keepdims } ->
      let input_desc = (find_node program input).Ir.desc in
      if not (String.equal node.Ir.desc.dtype "int32") then
        Error.raise
          (Error.Unsupported_program
             (Printf.sprintf "StableHLO argmax unsupported output dtype %s"
                node.Ir.desc.dtype));
      let input_ty = tensor_type input_desc in
      let index_desc = { input_desc with dtype = "int32" } in
      let index_ty = tensor_type index_desc in
      let reduced_value_desc =
        { input_desc with shape = reduced_shape input_desc.shape [| axis |] }
      in
      let reduced_value_ty = tensor_type reduced_value_desc in
      let reduced_index_desc =
        { node.Ir.desc with shape = reduced_shape input_desc.shape [| axis |] }
      in
      let reduced_index_ty = tensor_type reduced_index_desc in
      let input_scalar_ty =
        tensor_type { input_desc with shape = [||] }
      in
      let index_scalar_ty = tensor_type { index_desc with shape = [||] } in
      let init_value =
        reduce_init_literal Ir.Reduce_max input_desc.dtype
      in
      let init_index = "0" in
      let base_name = Printf.sprintf "%%v%d_argmax" node.Ir.id in
      Some
        (String.concat "\n"
           [
             Printf.sprintf "%s%%v%d_iota = stablehlo.iota dim = %d : %s"
               indent node.Ir.id axis index_ty;
             Printf.sprintf
               "%s%%v%d_init_value = stablehlo.constant dense<%s> : %s" indent
               node.Ir.id init_value input_scalar_ty;
             Printf.sprintf
               "%s%%v%d_init_index = stablehlo.constant dense<%s> : %s" indent
               node.Ir.id init_index index_scalar_ty;
             Printf.sprintf
               "%s%s:2 = stablehlo.reduce(%s init: %%v%d_init_value), (%%v%d_iota \
                init: %%v%d_init_index) across dimensions = %s : (%s, %s, %s, \
                %s) -> (%s, %s)"
               indent base_name (op_ref ~arg_name program input) node.Ir.id
               node.Ir.id node.Ir.id (dims_attr [| axis |]) input_ty index_ty
               input_scalar_ty index_scalar_ty reduced_value_ty reduced_index_ty;
             Printf.sprintf
               "%s  reducer(%s_lhs_value: %s, %s_rhs_value: %s) (%s_lhs_index: \
                %s, %s_rhs_index: %s) {"
               indent base_name input_scalar_ty base_name input_scalar_ty base_name
               index_scalar_ty base_name index_scalar_ty;
             Printf.sprintf
               "%s    %s_gt = stablehlo.compare  GT, %s_rhs_value, \
                %s_lhs_value : (%s, %s) -> tensor<i1>"
               indent base_name base_name base_name input_scalar_ty input_scalar_ty;
             Printf.sprintf
               "%s    %s_eq = stablehlo.compare  EQ, %s_rhs_value, \
                %s_lhs_value : (%s, %s) -> tensor<i1>"
               indent base_name base_name base_name input_scalar_ty input_scalar_ty;
             Printf.sprintf
               "%s    %s_idx_lt = stablehlo.compare  LT, %s_rhs_index, \
                %s_lhs_index : (%s, %s) -> tensor<i1>"
               indent base_name base_name base_name index_scalar_ty index_scalar_ty;
             Printf.sprintf
               "%s    %s_value = stablehlo.select %s_gt, %s_rhs_value, \
                %s_lhs_value : tensor<i1>, %s"
               indent base_name base_name base_name base_name input_scalar_ty;
             Printf.sprintf
               "%s    %s_idx_eq = stablehlo.select %s_idx_lt, %s_rhs_index, \
                %s_lhs_index : tensor<i1>, %s"
               indent base_name base_name base_name base_name index_scalar_ty;
             Printf.sprintf
               "%s    %s_idx_gt = stablehlo.select %s_gt, %s_rhs_index, \
                %s_lhs_index : tensor<i1>, %s"
               indent base_name base_name base_name base_name index_scalar_ty;
             Printf.sprintf
               "%s    %s_index = stablehlo.select %s_eq, %s_idx_eq, %s_idx_gt : \
                tensor<i1>, %s"
               indent base_name base_name base_name base_name index_scalar_ty;
             Printf.sprintf "%s    stablehlo.return %s_value, %s_index : %s, %s"
               indent base_name base_name input_scalar_ty index_scalar_ty;
             Printf.sprintf "%s  }" indent;
             Printf.sprintf
               "%s%%v%d = stablehlo.reshape %s#1 : (%s) -> %s" indent
               node.Ir.id base_name reduced_index_ty ty;
           ])
  | Arg_reduce { op; _ } ->
      unsupported (Ir.op_name (Ir.Arg_reduce { op; input = 0; axis = 0; keepdims = false }))
  | Matmul { lhs; rhs } ->
      let lhs_desc = (find_node program lhs).Ir.desc in
      let rhs_desc = (find_node program rhs).Ir.desc in
      let lhs_rank = Array.length lhs_desc.shape in
      let rhs_rank = Array.length rhs_desc.shape in
      let batching, contracting =
        if lhs_rank = 2 && rhs_rank = 2 then ("[] x []", "[1] x [0]")
        else if lhs_rank >= 2 && rhs_rank = 2 then
          ("[] x []", Printf.sprintf "[%d] x [0]" (lhs_rank - 1))
        else if lhs_rank = rhs_rank && lhs_rank >= 3 then
          let batch =
            Array.init (lhs_rank - 2) Fun.id |> dims_attr
          in
          ( Printf.sprintf "%s x %s" batch batch,
            Printf.sprintf "[%d] x [%d]" (lhs_rank - 1) (rhs_rank - 2) )
        else unsupported "matmul ranks"
      in
      Some
        (Printf.sprintf
           "%s%%v%d = stablehlo.dot_general %s, %s, batching_dims = %s, \
            contracting_dims = %s, precision = [] : (%s, %s) -> %s"
           indent node.Ir.id (op_ref ~arg_name program lhs)
           (op_ref ~arg_name program rhs) batching
           contracting (tensor_type lhs_desc) (tensor_type rhs_desc) ty)
  | Cat { inputs; axis } ->
      let refs =
        List.map (op_ref ~arg_name program) inputs |> String.concat ", "
      in
      let tys =
        List.map (fun id -> tensor_type (find_node program id).Ir.desc) inputs
        |> String.concat ", "
      in
      Some
        (Printf.sprintf
           "%s%%v%d = \"stablehlo.concatenate\"(%s) {dimension = %d : i64} : \
            (%s) -> %s"
           indent node.Ir.id refs axis tys ty)
  | Shrink { input; limits } ->
      let input_ty = tensor_type (find_node program input).Ir.desc in
      Some
        (Printf.sprintf "%s%%v%d = stablehlo.slice %s [%s] : (%s) -> %s"
           indent node.Ir.id (op_ref ~arg_name program input)
           (slice_spec limits) input_ty ty)
  | Flip { input; dims } ->
      let input_ty = tensor_type (find_node program input).Ir.desc in
      let dims =
        Array.to_list dims
        |> List.mapi (fun i flag -> if flag then Some i else None)
        |> List.filter_map Fun.id |> Array.of_list
      in
      Some
        (Printf.sprintf
           "%s%%v%d = \"stablehlo.reverse\"(%s) {dimensions = %s} : (%s) -> %s"
           indent node.Ir.id (op_ref ~arg_name program input)
           (reverse_dims_attr dims) input_ty ty)
  | Gather { data; indices; axis = 0 } ->
      let data_desc = (find_node program data).Ir.desc in
      let indices =
        if
          Array.length data_desc.shape = 2
          && Array.length node.Ir.desc.shape = 2
        then strip_axis0_embedding_indices program indices
        else indices
      in
      let indices_desc = (find_node program indices).Ir.desc in
      let data_rank = Array.length data_desc.shape in
      let indices_rank = Array.length indices_desc.shape in
      let offset_dims =
        gather_offset_dims ~axis:0 ~indices_rank ~data_rank
      in
      let slice_sizes =
        Array.mapi (fun i dim -> if i = 0 then 1 else dim) data_desc.shape
      in
      Some
        (Printf.sprintf
           "%s%%v%d = \"stablehlo.gather\"(%s, %s) <{dimension_numbers = \
            #stablehlo.gather<offset_dims = %s, collapsed_slice_dims = [0], \
            start_index_map = [0], index_vector_dim = %d>, indices_are_sorted \
            = false, slice_sizes = %s}> : (%s, %s) -> %s"
           indent node.Ir.id (op_ref ~arg_name program data)
           (op_ref ~arg_name program indices) (dims_attr offset_dims)
           indices_rank (slice_sizes_attr slice_sizes) (tensor_type data_desc)
           (tensor_type indices_desc) ty)
  | Gather _ ->
      unsupported "gather axis"
  | op ->
      unsupported (Ir.op_name op)

let lower_program ?(indent = "  ") ~arg_name program =
  let program = Ir.prune program in
  let body =
    program.Ir.nodes |> List.filter_map (lower_node ~indent ~arg_name program)
  in
  let outputs =
    List.map
      (fun id -> (op_ref ~arg_name program id, (find_node program id).Ir.desc))
      program.Ir.outputs
  in
  { body; outputs }

let of_program program =
  let params =
    parameter_nodes program
    |> List.map (fun (index, desc) ->
           Printf.sprintf "%%arg%d: %s" index (tensor_type desc))
    |> String.concat ", "
  in
  let lowered =
    lower_program program ~arg_name:(fun index -> Printf.sprintf "%%arg%d" index)
  in
  String.concat "\n"
    [
      "module {";
      Printf.sprintf "func.func @main(%s) -> %s {" params (output_type program);
      String.concat "\n" lowered.body;
      output_return ~indent:"  " lowered;
      "}";
      "}";
    ]
