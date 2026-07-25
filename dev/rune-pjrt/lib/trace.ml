(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core
open Nx_effect

type packed = Tensor : ('a, 'b) Nx.t -> packed
type capture = { program : Ir.program; outputs : packed list }
type binding = Binding : ('a, 'b) Nx.t * Ir.node_id -> binding

type env = {
  name : string option;
  enable_ffi : bool;
  mutable next_id : int;
  mutable nodes_rev : Ir.node list;
  mutable bindings : binding list;
  mutable inputs_rev : Ir.node_id list;
}

let create_env ?name ?(enable_ffi = true) () =
  {
    name;
    enable_ffi;
    next_id = 0;
    nodes_rev = [];
    bindings = [];
    inputs_rev = [];
  }

let same_tensor (type a b c d) (a : (a, b) Nx.t) (b : (c, d) Nx.t) =
  Obj.repr a == Obj.repr b

let find_binding env tensor =
  let rec loop = function
    | [] -> None
    | Binding (bound, id) :: rest ->
        if same_tensor bound tensor then Some id else loop rest
  in
  loop env.bindings

let bind env tensor id = env.bindings <- Binding (tensor, id) :: env.bindings

let add_node env desc op =
  let id = env.next_id in
  env.next_id <- id + 1;
  env.nodes_rev <- { Ir.id; desc; op } :: env.nodes_rev;
  id

let constant_of_tensor env tensor =
  let id =
    add_node env (Ir.desc_of_tensor tensor)
      (Ir.Constant (Ir.literal_of_tensor tensor))
  in
  bind env tensor id;
  id

let ensure_id env tensor =
  match find_binding env tensor with
  | Some id -> id
  | None -> constant_of_tensor env tensor

let register_parameter env index tensor =
  let id = add_node env (Ir.desc_of_tensor tensor) (Ir.Parameter index) in
  bind env tensor id;
  env.inputs_rev <- id :: env.inputs_rev

let bind_node env tensor op =
  let id = add_node env (Ir.desc_of_tensor tensor) op in
  bind env tensor id;
  id

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

let backend_binary op a b = T (op (unwrap a) (unwrap b))
let backend_unary op t_in = T (op (unwrap t_in))

let backend_reduce op ~axes ~keepdims t_in =
  T (op ~axes ~keepdims (unwrap t_in))

let backend_arg_reduce op ~axis ~keepdims t_in =
  T (op ~axis ~keepdims (unwrap t_in))

let continue_binary env k backend_op ir_op a b =
  let result = backend_binary backend_op a b in
  binary_record env ir_op result a b;
  Effect.Deep.continue k result

let continue_unary env k backend_op ir_op t_in =
  let result = backend_unary backend_op t_in in
  unary_record env ir_op result t_in;
  Effect.Deep.continue k result

let continue_reduce env k backend_op ir_op t_in ~axes ~keepdims =
  let result = backend_reduce backend_op ~axes ~keepdims t_in in
  reduce_record env ir_op result t_in ~axes ~keepdims;
  Effect.Deep.continue k result

let continue_arg_reduce env k backend_op ir_op t_in ~axis ~keepdims =
  let result = backend_arg_reduce backend_op ~axis ~keepdims t_in in
  arg_reduce_record env ir_op result t_in ~axis ~keepdims;
  Effect.Deep.continue k result

let handler env =
  {
    Effect.Deep.retc = (fun outputs -> outputs);
    exnc = Stdlib.raise;
    effc =
      (fun (type a) (eff : a Effect.t) ->
        let unsupported name =
          Some
            (fun (k : (a, _) Effect.Deep.continuation) ->
              ignore k;
              Error.raise (Error.Unsupported_effect name))
        in
        match eff with
        | Triton.Internal.Call request when env.enable_ffi ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let output = request.fallback () in
                let inputs =
                  List.map
                    (fun (Triton.Tensor input) -> ensure_id env input)
                    request.inputs
                in
                let source = request.kernel in
                let kernel : Ir.triton_kernel =
                  {
                    name = source.name;
                    ir = source.ir;
                    num_warps = source.num_warps;
                    num_stages = source.num_stages;
                    grid_x = source.grid_x;
                    grid_y = source.grid_y;
                    grid_z = source.grid_z;
                  }
                in
                ignore
                  (bind_node env output (Ir.Triton_call { kernel; inputs }));
                Effect.Deep.continue k (Triton.Internal.Use_kernel output))
        | Triton.Internal.Call _ ->
            Some (fun k -> Effect.Deep.continue k Triton.Internal.Use_fallback)
        | Ffi.Internal.Call request when env.enable_ffi ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let output = request.fallback () in
                let inputs =
                  List.map
                    (fun (Ffi.Tensor input) -> ensure_id env input)
                    request.inputs
                in
                let ffi_handler = request.handler in
                let identity = Ffi.Internal.identity ffi_handler in
                let handler : Ir.ffi_handler =
                  {
                    library = identity.library;
                    library_digest = identity.library_digest;
                    symbol = ffi_handler.symbol;
                    target = identity.target;
                  }
                in
                ignore
                  (bind_node env output (Ir.Custom_call { handler; inputs }));
                Effect.Deep.continue k (Ffi.Internal.Use_kernel output))
        | Ffi.Internal.Call _ ->
            Some (fun k -> Effect.Deep.continue k Ffi.Internal.Use_fallback)
        | E_view t ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let view = Nx_backend.view (unwrap t) in
                Effect.Deep.continue k view)
        | E_buffer { context; dtype; size_in_elements } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result =
                  T (Nx_backend.buffer context dtype [| size_in_elements |])
                in
                ignore (bind_node env result (Ir.Buffer { size_in_elements }));
                Effect.Deep.continue k result)
        | E_const_scalar { context; value; dtype } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = T (Nx_backend.full context dtype [||] value) in
                ignore
                  (add_node env (Ir.desc_of_tensor result)
                     (Ir.Constant (Ir.literal_of_tensor result)));
                bind env result (env.next_id - 1);
                Effect.Deep.continue k result)
        | E_from_host { context; array } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = T (Nx_backend.from_host context array) in
                ignore
                  (bind_node env result
                     (Ir.Constant (Ir.literal_of_tensor result)));
                Effect.Deep.continue k result)
        | E_add { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.add Ir.Add a b)
        | E_sub { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.sub Ir.Sub a b)
        | E_mul { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.mul Ir.Mul a b)
        | E_idiv { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.div Ir.Idiv a b)
        | E_fdiv { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.div Ir.Fdiv a b)
        | E_max { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.max Ir.Max a b)
        | E_min { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.min Ir.Min a b)
        | E_mod { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.mod_ Ir.Mod a b)
        | E_pow { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.pow Ir.Pow a b)
        | E_xor { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.xor Ir.Xor a b)
        | E_or { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.or_ Ir.Or a b)
        | E_and { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.and_ Ir.And a b)
        | E_atan2 { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.atan2 Ir.Atan2 a b)
        | E_cmpeq { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.cmpeq Ir.CmpEq a b)
        | E_cmpne { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.cmpne Ir.CmpNe a b)
        | E_cmplt { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.cmplt Ir.CmpLt a b)
        | E_cmple { a; b } ->
            Some (fun k -> continue_binary env k Nx_backend.cmple Ir.CmpLe a b)
        | E_neg { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.neg Ir.Neg t_in)
        | E_sin { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.sin Ir.Sin t_in)
        | E_sqrt { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.sqrt Ir.Sqrt t_in)
        | E_recip { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.recip Ir.Recip t_in)
        | E_log { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.log Ir.Log t_in)
        | E_exp { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.exp Ir.Exp t_in)
        | E_cos { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.cos Ir.Cos t_in)
        | E_abs { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.abs Ir.Abs t_in)
        | E_sign { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.sign Ir.Sign t_in)
        | E_tan { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.tan Ir.Tan t_in)
        | E_asin { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.asin Ir.Asin t_in)
        | E_acos { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.acos Ir.Acos t_in)
        | E_atan { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.atan Ir.Atan t_in)
        | E_sinh { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.sinh Ir.Sinh t_in)
        | E_cosh { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.cosh Ir.Cosh t_in)
        | E_tanh { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.tanh Ir.Tanh t_in)
        | E_trunc { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.trunc Ir.Trunc t_in)
        | E_ceil { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.ceil Ir.Ceil t_in)
        | E_floor { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.floor Ir.Floor t_in)
        | E_round { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.round Ir.Round t_in)
        | E_erf { t_in } ->
            Some (fun k -> continue_unary env k Nx_backend.erf Ir.Erf t_in)
        | E_where { condition; if_true; if_false } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result =
                  T
                    (Nx_backend.where (unwrap condition) (unwrap if_true)
                       (unwrap if_false))
                in
                let condition = ensure_id env condition in
                let if_true = ensure_id env if_true in
                let if_false = ensure_id env if_false in
                ignore
                  (bind_node env result
                     (Ir.Where { condition; if_true; if_false }));
                Effect.Deep.continue k result)
        | E_reduce_sum { t_in; axes; keepdims } ->
            Some
              (fun k ->
                continue_reduce env k Nx_backend.reduce_sum Ir.Reduce_sum t_in
                  ~axes ~keepdims)
        | E_reduce_max { t_in; axes; keepdims } ->
            Some
              (fun k ->
                continue_reduce env k Nx_backend.reduce_max Ir.Reduce_max t_in
                  ~axes ~keepdims)
        | E_reduce_min { t_in; axes; keepdims } ->
            Some
              (fun k ->
                continue_reduce env k Nx_backend.reduce_min Ir.Reduce_min t_in
                  ~axes ~keepdims)
        | E_reduce_prod { t_in; axes; keepdims } ->
            Some
              (fun k ->
                continue_reduce env k Nx_backend.reduce_prod Ir.Reduce_prod t_in
                  ~axes ~keepdims)
        | E_argmax { t_in; axis; keepdims } ->
            Some
              (fun k ->
                continue_arg_reduce env k Nx_backend.argmax Ir.Argmax t_in ~axis
                  ~keepdims)
        | E_argmin { t_in; axis; keepdims } ->
            Some
              (fun k ->
                continue_arg_reduce env k Nx_backend.argmin Ir.Argmin t_in ~axis
                  ~keepdims)
        | E_reshape { t_in; new_shape } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = T (Nx_backend.reshape (unwrap t_in) new_shape) in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Reshape { input; shape = Array.copy new_shape }));
                Effect.Deep.continue k result)
        | E_expand { t_in; new_target_shape } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result =
                  T (Nx_backend.expand (unwrap t_in) new_target_shape)
                in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Expand { input; shape = Array.copy new_target_shape }));
                Effect.Deep.continue k result)
        | E_permute { t_in; axes } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = T (Nx_backend.permute (unwrap t_in) axes) in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Permute { input; axes = Array.copy axes }));
                Effect.Deep.continue k result)
        | E_shrink { t_in; limits } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = T (Nx_backend.shrink (unwrap t_in) limits) in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Shrink { input; limits = Array.copy limits }));
                Effect.Deep.continue k result)
        | E_flip { t_in; dims_to_flip } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = T (Nx_backend.flip (unwrap t_in) dims_to_flip) in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Flip { input; dims = Array.copy dims_to_flip }));
                Effect.Deep.continue k result)
        | E_pad { t_in; padding_config; fill_value } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result =
                  T (Nx_backend.pad (unwrap t_in) padding_config fill_value)
                in
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
                let result = T (Nx_backend.contiguous (unwrap t_in)) in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Unary { op = Ir.Contiguous; input }));
                Effect.Deep.continue k result)
        | E_copy { t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = T (Nx_backend.copy (unwrap t_in)) in
                let input = ensure_id env t_in in
                ignore (bind_node env result (Ir.Unary { op = Ir.Copy; input }));
                Effect.Deep.continue k result)
        | E_assign { dst; src } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                Nx_backend.assign (unwrap dst) (unwrap src);
                record_assign env dst src;
                Effect.Deep.continue k ())
        | E_cat { t_list; axis } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result =
                  T (Nx_backend.cat (List.map unwrap t_list) ~axis)
                in
                let inputs = List.map (ensure_id env) t_list in
                ignore (bind_node env result (Ir.Cat { inputs; axis }));
                Effect.Deep.continue k result)
        | E_cast { t_in; target_dtype } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result =
                  T (Nx_backend.cast ~dtype:target_dtype (unwrap t_in))
                in
                let input = ensure_id env t_in in
                ignore
                  (bind_node env result
                     (Ir.Cast { input; dtype = Dtype.to_string target_dtype }));
                Effect.Deep.continue k result)
        | E_gather { data; indices; axis } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result =
                  T (Nx_backend.gather (unwrap data) (unwrap indices) ~axis)
                in
                let data = ensure_id env data in
                let indices = ensure_id env indices in
                ignore
                  (bind_node env result (Ir.Gather { data; indices; axis }));
                Effect.Deep.continue k result)
        | E_matmul { a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = T (Nx_backend.matmul (unwrap a) (unwrap b)) in
                let lhs = ensure_id env a in
                let rhs = ensure_id env b in
                ignore (bind_node env result (Ir.Matmul { lhs; rhs }));
                Effect.Deep.continue k result)
        | E_to_device { t_in; context } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                ignore k;
                ignore (to_device context t_in);
                Error.raise (Error.Unsupported_effect "to_device"))
        | E_sort _ -> unsupported "sort"
        | E_argsort _ -> unsupported "argsort"
        | E_associative_scan _ -> unsupported "associative_scan"
        | E_scatter _ -> unsupported "scatter"
        | E_threefry _ -> unsupported "threefry"
        | E_unfold _ -> unsupported "unfold"
        | E_fold _ -> unsupported "fold"
        | E_fft _ -> unsupported "fft"
        | E_ifft _ -> unsupported "ifft"
        | E_rfft _ -> unsupported "rfft"
        | E_irfft _ -> unsupported "irfft"
        | E_psum _ -> unsupported "psum"
        | E_cholesky _ -> unsupported "cholesky"
        | E_qr _ -> unsupported "qr"
        | E_svd _ -> unsupported "svd"
        | E_eig _ -> unsupported "eig"
        | E_eigh _ -> unsupported "eigh"
        | E_triangular_solve _ -> unsupported "triangular_solve"
        | _ -> None);
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
        match find_binding env tensor with
        | Some id -> id
        | None -> assert false)
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

let capture_many ?name ?(enable_ffi = true) f inputs =
  let env = create_env ?name ~enable_ffi () in
  List.iteri (register_parameter env) inputs;
  let outputs =
    Effect.Deep.match_with
      (fun xs -> f xs |> List.map (fun t -> Tensor t))
      inputs (handler env)
  in
  finalize env outputs

let capture_one ?name ?(enable_ffi = true) f input =
  capture_many ?name ~enable_ffi
    (fun inputs -> match inputs with [ x ] -> [ f x ] | _ -> assert false)
    [ input ]
