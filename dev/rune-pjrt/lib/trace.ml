(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core
open Nx_effect

type packed = Tensor : ('a, 'b) Nx.t -> packed

type capture = {
  program : Ir.program;
  outputs : packed list;
}

type binding = Binding : ('a, 'b) Nx.t * Ir.node_id -> binding

type env = {
  name : string option;
  mutable next_id : int;
  mutable nodes_rev : Ir.node list;
  mutable bindings : binding list;
  mutable inputs_rev : Ir.node_id list;
}

let create_env ?name () =
  { name; next_id = 0; nodes_rev = []; bindings = []; inputs_rev = [] }

let same_tensor (type a b c d) (a : (a, b) Nx.t) (b : (c, d) Nx.t) =
  Obj.repr a == Obj.repr b

let find_binding env tensor =
  let rec loop = function
    | [] -> None
    | Binding (bound, id) :: rest ->
        if same_tensor bound tensor then Some id else loop rest
  in
  loop env.bindings

let bind env tensor id =
  env.bindings <- Binding (tensor, id) :: env.bindings

let add_node env desc op =
  let id = env.next_id in
  env.next_id <- id + 1;
  env.nodes_rev <- { Ir.id; desc; op } :: env.nodes_rev;
  id

let constant_of_tensor env tensor =
  let id =
    add_node env (Ir.desc_of_tensor tensor) (Ir.Constant (Ir.literal_of_tensor tensor))
  in
  bind env tensor id;
  id

let ensure_id env tensor =
  match find_binding env tensor with Some id -> id | None -> constant_of_tensor env tensor

let register_parameter env index tensor =
  let id = add_node env (Ir.desc_of_tensor tensor) (Ir.Parameter index) in
  bind env tensor id;
  env.inputs_rev <- id :: env.inputs_rev

let bind_node env tensor op =
  let id = add_node env (Ir.desc_of_tensor tensor) op in
  bind env tensor id;
  id

let bind_unsupported env name tensor =
  ignore (bind_node env tensor (Ir.Unsupported name))

let record_assign env dst src =
  let src_id = ensure_id env src in
  bind env dst src_id

let scalar_to_string (type a b) (dtype : (a, b) Dtype.t) (value : a) =
  match dtype with
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 ->
      Printf.sprintf "%g" (Obj.magic value : float)
  | Int4 | UInt4 | Int8 | UInt8 | Int16 | UInt16 ->
      string_of_int (Obj.magic value : int)
  | Int32 | UInt32 -> Int32.to_string (Obj.magic value : int32)
  | Int64 | UInt64 -> Int64.to_string (Obj.magic value : int64)
  | Complex64 | Complex128 ->
      let v = (Obj.magic value : Complex.t) in
      Printf.sprintf "%g+%gi" v.Complex.re v.Complex.im
  | Bool -> string_of_bool (Obj.magic value : bool)

let unary_record env op out input =
  let input = ensure_id env input in
  ignore (bind_node env out (Ir.Unary { op; input }))

let binary_record env op out lhs rhs =
  let lhs = ensure_id env lhs in
  let rhs = ensure_id env rhs in
  ignore (bind_node env out (Ir.Binary { op; lhs; rhs }))

let reduce_record env op out input ~axes ~keepdims =
  let input = ensure_id env input in
  ignore (bind_node env out (Ir.Reduce { op; input; axes; keepdims }))

let arg_reduce_record env op out input ~axis ~keepdims =
  let input = ensure_id env input in
  ignore (bind_node env out (Ir.Arg_reduce { op; input; axis; keepdims }))

let exec_unsupported name thunk =
  let _ = thunk () in
  Error.raise (Error.Unsupported_effect name)

let handler env =
  {
    Effect.Deep.retc = (fun outputs -> outputs);
    exnc = Stdlib.raise;
    effc =
      (fun (type a) (eff : a Effect.t) ->
        match eff with
        | E_view t ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let view = view t in
                Effect.Deep.continue k view)
        | E_buffer { context; dtype; size_in_elements } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = buffer context dtype [| size_in_elements |] in
                ignore
                  (bind_node env result
                     (Ir.Buffer { size_in_elements = Nx.numel result }));
                Effect.Deep.continue k result)
        | E_const_scalar { context; value; dtype } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = const_scalar context value dtype in
                ignore
                  (add_node env (Ir.desc_of_tensor result)
                     (Ir.Constant (Ir.literal_of_tensor result)));
                bind env result (env.next_id - 1);
                Effect.Deep.continue k result)
        | E_from_host { context; array } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = from_host context array in
                ignore
                  (bind_node env result
                     (Ir.Constant (Ir.literal_of_tensor result)));
                Effect.Deep.continue k result)
        | E_add { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                add ~out a b;
                binary_record env Ir.Add out a b;
                Effect.Deep.continue k ())
        | E_sub { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                sub ~out a b;
                binary_record env Ir.Sub out a b;
                Effect.Deep.continue k ())
        | E_mul { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                mul ~out a b;
                binary_record env Ir.Mul out a b;
                Effect.Deep.continue k ())
        | E_idiv { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                div ~out a b;
                binary_record env Ir.Idiv out a b;
                Effect.Deep.continue k ())
        | E_fdiv { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                div ~out a b;
                binary_record env Ir.Fdiv out a b;
                Effect.Deep.continue k ())
        | E_max { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                max ~out a b;
                binary_record env Ir.Max out a b;
                Effect.Deep.continue k ())
        | E_min { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                min ~out a b;
                binary_record env Ir.Min out a b;
                Effect.Deep.continue k ())
        | E_mod { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                mod_ ~out a b;
                binary_record env Ir.Mod out a b;
                Effect.Deep.continue k ())
        | E_pow { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                pow ~out a b;
                binary_record env Ir.Pow out a b;
                Effect.Deep.continue k ())
        | E_xor { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                xor ~out a b;
                binary_record env Ir.Xor out a b;
                Effect.Deep.continue k ())
        | E_or { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                or_ ~out a b;
                binary_record env Ir.Or out a b;
                Effect.Deep.continue k ())
        | E_and { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                and_ ~out a b;
                binary_record env Ir.And out a b;
                Effect.Deep.continue k ())
        | E_atan2 { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                atan2 ~out a b;
                binary_record env Ir.Atan2 out a b;
                Effect.Deep.continue k ())
        | E_cmpeq { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                cmpeq ~out a b;
                binary_record env Ir.CmpEq out a b;
                Effect.Deep.continue k ())
        | E_cmpne { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                cmpne ~out a b;
                binary_record env Ir.CmpNe out a b;
                Effect.Deep.continue k ())
        | E_cmplt { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                cmplt ~out a b;
                binary_record env Ir.CmpLt out a b;
                Effect.Deep.continue k ())
        | E_cmple { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                cmple ~out a b;
                binary_record env Ir.CmpLe out a b;
                Effect.Deep.continue k ())
        | E_neg { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                neg ~out t_in;
                unary_record env Ir.Neg out t_in;
                Effect.Deep.continue k ())
        | E_sin { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                sin ~out t_in;
                unary_record env Ir.Sin out t_in;
                Effect.Deep.continue k ())
        | E_sqrt { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                sqrt ~out t_in;
                unary_record env Ir.Sqrt out t_in;
                Effect.Deep.continue k ())
        | E_recip { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                recip ~out t_in;
                unary_record env Ir.Recip out t_in;
                Effect.Deep.continue k ())
        | E_log { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                log ~out t_in;
                unary_record env Ir.Log out t_in;
                Effect.Deep.continue k ())
        | E_exp { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                exp ~out t_in;
                unary_record env Ir.Exp out t_in;
                Effect.Deep.continue k ())
        | E_cos { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                cos ~out t_in;
                unary_record env Ir.Cos out t_in;
                Effect.Deep.continue k ())
        | E_abs { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                abs ~out t_in;
                unary_record env Ir.Abs out t_in;
                Effect.Deep.continue k ())
        | E_sign { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                sign ~out t_in;
                unary_record env Ir.Sign out t_in;
                Effect.Deep.continue k ())
        | E_tan { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                tan ~out t_in;
                unary_record env Ir.Tan out t_in;
                Effect.Deep.continue k ())
        | E_asin { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                asin ~out t_in;
                unary_record env Ir.Asin out t_in;
                Effect.Deep.continue k ())
        | E_acos { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                acos ~out t_in;
                unary_record env Ir.Acos out t_in;
                Effect.Deep.continue k ())
        | E_atan { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                atan ~out t_in;
                unary_record env Ir.Atan out t_in;
                Effect.Deep.continue k ())
        | E_sinh { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                sinh ~out t_in;
                unary_record env Ir.Sinh out t_in;
                Effect.Deep.continue k ())
        | E_cosh { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                cosh ~out t_in;
                unary_record env Ir.Cosh out t_in;
                Effect.Deep.continue k ())
        | E_tanh { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                tanh ~out t_in;
                unary_record env Ir.Tanh out t_in;
                Effect.Deep.continue k ())
        | E_trunc { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                trunc ~out t_in;
                unary_record env Ir.Trunc out t_in;
                Effect.Deep.continue k ())
        | E_ceil { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                ceil ~out t_in;
                unary_record env Ir.Ceil out t_in;
                Effect.Deep.continue k ())
        | E_floor { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                floor ~out t_in;
                unary_record env Ir.Floor out t_in;
                Effect.Deep.continue k ())
        | E_round { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                round ~out t_in;
                unary_record env Ir.Round out t_in;
                Effect.Deep.continue k ())
        | E_erf { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                erf ~out t_in;
                unary_record env Ir.Erf out t_in;
                Effect.Deep.continue k ())
        | E_where { out; condition; if_true; if_false } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                where ~out condition if_true if_false;
                let condition = ensure_id env condition in
                let if_true = ensure_id env if_true in
                let if_false = ensure_id env if_false in
                ignore
                  (bind_node env out
                     (Ir.Where { condition; if_true; if_false }));
                Effect.Deep.continue k ())
        | E_reduce_sum { out; t_in; axes; keepdims } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                reduce_sum ~out ~axes ~keepdims t_in;
                reduce_record env Ir.Reduce_sum out t_in ~axes ~keepdims;
                Effect.Deep.continue k ())
        | E_reduce_max { out; t_in; axes; keepdims } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                reduce_max ~out ~axes ~keepdims t_in;
                reduce_record env Ir.Reduce_max out t_in ~axes ~keepdims;
                Effect.Deep.continue k ())
        | E_reduce_min { out; t_in; axes; keepdims } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                reduce_min ~out ~axes ~keepdims t_in;
                reduce_record env Ir.Reduce_min out t_in ~axes ~keepdims;
                Effect.Deep.continue k ())
        | E_reduce_prod { out; t_in; axes; keepdims } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                reduce_prod ~out ~axes ~keepdims t_in;
                reduce_record env Ir.Reduce_prod out t_in ~axes ~keepdims;
                Effect.Deep.continue k ())
        | E_argmax { out; t_in; axis; keepdims } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                argmax ~out ~axis ~keepdims t_in;
                arg_reduce_record env Ir.Argmax out t_in ~axis ~keepdims;
                Effect.Deep.continue k ())
        | E_argmin { out; t_in; axis; keepdims } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                argmin ~out ~axis ~keepdims t_in;
                arg_reduce_record env Ir.Argmin out t_in ~axis ~keepdims;
                Effect.Deep.continue k ())
        | E_reshape { t_in; new_shape } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = reshape t_in new_shape in
                let input = ensure_id env t_in in
                ignore (bind_node env result (Ir.Reshape { input; shape = Array.copy new_shape }));
                Effect.Deep.continue k result)
        | E_expand { t_in; new_target_shape } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = expand t_in new_target_shape in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Expand { input; shape = Array.copy new_target_shape }));
                Effect.Deep.continue k result)
        | E_permute { t_in; axes } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = permute t_in axes in
                let input = ensure_id env t_in in
                ignore (bind_node env result (Ir.Permute { input; axes = Array.copy axes }));
                Effect.Deep.continue k result)
        | E_shrink { t_in; limits } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = shrink t_in limits in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Shrink { input; limits = Array.copy limits }));
                Effect.Deep.continue k result)
        | E_flip { t_in; dims_to_flip } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = flip t_in dims_to_flip in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Flip { input; dims = Array.copy dims_to_flip }));
                Effect.Deep.continue k result)
        | E_pad { t_in; padding_config; fill_value } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = pad t_in padding_config fill_value in
                let input = ensure_id env t_in in
                let fill_value = scalar_to_string (Nx.dtype t_in) fill_value in
                ignore
                  (bind_node env result
                     (Ir.Pad
                        {
                          input;
                          padding = Array.copy padding_config;
                          fill_value;
                        }));
                Effect.Deep.continue k result)
        | E_contiguous { t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = contiguous t_in in
                let input = ensure_id env t_in in
                ignore (bind_node env result (Ir.Unary { op = Ir.Contiguous; input }));
                Effect.Deep.continue k result)
        | E_copy { t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = copy t_in in
                let input = ensure_id env t_in in
                ignore (bind_node env result (Ir.Unary { op = Ir.Copy; input }));
                Effect.Deep.continue k result)
        | E_assign { dst; src } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                assign dst src;
                record_assign env dst src;
                Effect.Deep.continue k ())
        | E_cat { t_list; axis } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = Nx.concatenate ~axis t_list in
                let inputs = List.map (ensure_id env) t_list in
                ignore (bind_node env result (Ir.Cat { inputs; axis }));
                Effect.Deep.continue k result)
        | E_cast { t_in; target_dtype } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = Nx.astype target_dtype t_in in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Cast { input; dtype = Dtype.to_string target_dtype }));
                Effect.Deep.continue k result)
        | E_gather { data; indices; axis } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = Nx.take_along_axis ~axis indices data in
                let data = ensure_id env data in
                let indices = ensure_id env indices in
                ignore (bind_node env result (Ir.Gather { data; indices; axis }));
                Effect.Deep.continue k result)
        | E_matmul { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                matmul ~out a b;
                let lhs = ensure_id env a in
                let rhs = ensure_id env b in
                ignore (bind_node env out (Ir.Matmul { lhs; rhs }));
                Effect.Deep.continue k ())
        | E_to_device { t_in; context } ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                ignore (to_device context t_in);
                Error.raise (Error.Unsupported_effect "to_device"))
        | E_sort _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "sort"))
        | E_argsort _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "argsort"))
        | E_associative_scan _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "associative_scan"))
        | E_scatter _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "scatter"))
        | E_threefry _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "threefry"))
        | E_unfold _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "unfold"))
        | E_fold _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "fold"))
        | E_fft _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "fft"))
        | E_ifft _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "ifft"))
        | E_rfft _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "rfft"))
        | E_irfft _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "irfft"))
        | E_psum _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "psum"))
        | E_cholesky _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "cholesky"))
        | E_qr _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "qr"))
        | E_svd _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "svd"))
        | E_eig _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "eig"))
        | E_eigh _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "eigh"))
        | E_triangular_solve _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "triangular_solve"))
        | _ ->
            Some
              (fun (_k : (a, _) Effect.Deep.continuation) ->
                Error.raise (Error.Unsupported_effect "unknown_effect")))
  }

let finalize env outputs =
  let outputs =
    List.map
      (fun (Tensor tensor as packed) ->
        ignore (ensure_id env tensor);
        packed)
      outputs
  in
  let output_ids =
    List.map
      (fun (Tensor tensor) ->
        match find_binding env tensor with Some id -> id | None -> assert false)
      outputs
  in
  {
    program =
      Ir.prune
        {
          Ir.name = env.name;
          inputs = List.rev env.inputs_rev;
          outputs = output_ids;
          nodes = List.rev env.nodes_rev;
        };
    outputs;
  }

let capture_many ?name f inputs =
  let env = create_env ?name () in
  List.iteri (register_parameter env) inputs;
  let outputs =
    Effect.Deep.match_with (fun xs -> f xs |> List.map (fun t -> Tensor t)) inputs
      (handler env)
  in
  finalize env outputs

let capture_one ?name f input =
  capture_many ?name (fun inputs ->
      match inputs with
      | [ x ] -> [ f x ]
      | _ -> assert false)
    [ input ]
