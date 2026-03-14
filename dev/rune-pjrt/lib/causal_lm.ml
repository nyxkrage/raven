(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let desc shape dtype = Ir.{ shape; dtype }

let desc_of_tensor (type a b) (t : (a, b) Nx.t) = Ir.desc_of_tensor t

let reduced_shape shape axis =
  shape
  |> Array.to_list
  |> List.mapi (fun i dim -> (i, dim))
  |> List.filter_map (fun (i, dim) -> if i = axis then None else Some dim)
  |> Array.of_list

let reduce_max_init_literal dtype =
  match dtype with
  | "float16" -> "0xFC00"
  | "float32" -> "0xFF800000"
  | "float64" -> "0xFFF0000000000000"
  | "int8" -> "-128"
  | "int16" -> "-32768"
  | "int32" -> "-2147483648"
  | "int64" -> "-9223372036854775808"
  | "uint8" | "uint16" | "uint32" | "uint64" | "bool" -> "0"
  | dtype ->
      Error.raise
        (Error.Unsupported_program
           (Printf.sprintf
              "causal_lm.greedy_decode: unsupported argmax dtype %s" dtype))

let argmax_lines ~indent ~base ~input ~input_desc ~axis =
  if axis < 0 || axis >= Array.length input_desc.Ir.shape then
    invalid_argf "Rune_pjrt.Causal_lm.greedy_decode: axis %d out of bounds"
      axis;
  let input_ty = Stablehlo.tensor_type input_desc in
  let index_desc = { input_desc with dtype = "int32" } in
  let index_ty = Stablehlo.tensor_type index_desc in
  let reduced_value_desc =
    { input_desc with shape = reduced_shape input_desc.shape axis }
  in
  let reduced_value_ty = Stablehlo.tensor_type reduced_value_desc in
  let output_desc = { reduced_value_desc with dtype = "int32" } in
  let output_ty = Stablehlo.tensor_type output_desc in
  let input_scalar_ty =
    Stablehlo.tensor_type { input_desc with shape = [||] }
  in
  let index_scalar_ty =
    Stablehlo.tensor_type { index_desc with shape = [||] }
  in
  let init_value = reduce_max_init_literal input_desc.dtype in
  [
    Printf.sprintf "%s%%%s_iota = stablehlo.iota dim = %d : %s" indent base axis
      index_ty;
    Printf.sprintf "%s%%%s_init_value = stablehlo.constant dense<%s> : %s"
      indent base init_value input_scalar_ty;
    Printf.sprintf "%s%%%s_init_index = stablehlo.constant dense<0> : %s" indent
      base index_scalar_ty;
    Printf.sprintf
      "%s%%%s_pair:2 = stablehlo.reduce(%s init: %%%s_init_value), \
       (%%%s_iota init: %%%s_init_index) across dimensions = [%d] : (%s, %s, \
       %s, %s) -> (%s, %s)"
      indent base input base base base axis input_ty index_ty input_scalar_ty
      index_scalar_ty reduced_value_ty output_ty;
    Printf.sprintf
      "%s  reducer(%%%s_lhs_value: %s, %%%s_rhs_value: %s) \
       (%%%s_lhs_index: %s, %%%s_rhs_index: %s) {"
      indent base input_scalar_ty base input_scalar_ty base index_scalar_ty base
      index_scalar_ty;
    Printf.sprintf
      "%s    %%%s_gt = stablehlo.compare  GT, %%%s_rhs_value, %%%s_lhs_value : \
       (%s, %s) -> tensor<i1>"
      indent base base base input_scalar_ty input_scalar_ty;
    Printf.sprintf
      "%s    %%%s_eq = stablehlo.compare  EQ, %%%s_rhs_value, %%%s_lhs_value : \
       (%s, %s) -> tensor<i1>"
      indent base base base input_scalar_ty input_scalar_ty;
    Printf.sprintf
      "%s    %%%s_idx_lt = stablehlo.compare  LT, %%%s_rhs_index, \
       %%%s_lhs_index : (%s, %s) -> tensor<i1>"
      indent base base base index_scalar_ty index_scalar_ty;
    Printf.sprintf
      "%s    %%%s_value = stablehlo.select %%%s_gt, %%%s_rhs_value, \
       %%%s_lhs_value : tensor<i1>, %s"
      indent base base base base input_scalar_ty;
    Printf.sprintf
      "%s    %%%s_idx_eq = stablehlo.select %%%s_idx_lt, %%%s_rhs_index, \
       %%%s_lhs_index : tensor<i1>, %s"
      indent base base base base index_scalar_ty;
    Printf.sprintf
      "%s    %%%s_idx_gt = stablehlo.select %%%s_gt, %%%s_rhs_index, \
       %%%s_lhs_index : tensor<i1>, %s"
      indent base base base base index_scalar_ty;
    Printf.sprintf
      "%s    %%%s_index = stablehlo.select %%%s_eq, %%%s_idx_eq, \
       %%%s_idx_gt : tensor<i1>, %s"
      indent base base base base index_scalar_ty;
    Printf.sprintf "%s    stablehlo.return %%%s_value, %%%s_index : %s, %s"
      indent base base input_scalar_ty index_scalar_ty;
    Printf.sprintf "%s  }" indent;
    Printf.sprintf "%s%%%s = stablehlo.reshape %%%s_pair#1 : (%s) -> %s" indent
      base base output_ty output_ty;
  ]

let validate_input_ids (type a) (input_ids : (int32, a) Nx.t) =
  let dtype = Nx_core.Dtype.to_string (Nx.dtype input_ids) in
  if not (String.equal dtype "int32") then
    invalid_argf
      "Rune_pjrt.Causal_lm.greedy_decode: expected int32 input ids, got %s"
      dtype;
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 then
    invalid_argf
      "Rune_pjrt.Causal_lm.greedy_decode: expected rank-2 input ids, got rank %d"
      (Array.length shape);
  let batch = shape.(0) in
  let prompt_len = shape.(1) in
  if batch <= 0 then
    invalid_arg
      "Rune_pjrt.Causal_lm.greedy_decode: batch size must be positive";
  if prompt_len <= 0 then
    invalid_arg
      "Rune_pjrt.Causal_lm.greedy_decode: prompt length must be positive";
  (batch, prompt_len)

let compile ~backend ~device_id ~max_tokens
    (forward : (int32, 'a) Nx.t -> ('b, 'c) Nx.t)
    (input_ids : (int32, 'a) Nx.t) =
  if device_id < 0 then
    invalid_arg "Rune_pjrt.Causal_lm.greedy_decode: device_id must be >= 0";
  if max_tokens < 0 then
    invalid_arg "Rune_pjrt.Causal_lm.greedy_decode: max_tokens must be >= 0";
  let batch, prompt_len = validate_input_ids input_ids in
  let max_seq = prompt_len + max_tokens in
  let tokens_sample : (int32, 'a) Nx.t =
    Obj.magic (Nx.zeros Nx.int32 [| batch; max_seq |])
  in
  let capture = Trace.capture_one ~name:"causal_lm_forward" forward tokens_sample in
  if List.length capture.program.Ir.inputs <> 1 then
    Error.raise
      (Error.Unsupported_program
         "causal_lm.greedy_decode expected a single token input");
  let program, lifted_constants =
    Ir.parameterize_constants capture.program
  in
  let lowered =
    Stablehlo.lower_program ~indent:"      "
      ~arg_name:(fun index ->
        if index = 0 then "%iterTokens"
        else Printf.sprintf "%%arg%d" index)
      program
  in
  let logits_ref, logits_desc =
    match lowered.outputs with
    | [ output ] -> output
    | _ ->
        Error.raise
          (Error.Unsupported_program
             "causal_lm.greedy_decode expected a single logits output")
  in
  if Array.length logits_desc.shape <> 3 then
    Error.raise
      (Error.Unsupported_program
         (Printf.sprintf
            "causal_lm.greedy_decode expected logits with rank 3, got rank %d"
            (Array.length logits_desc.shape)));
  if logits_desc.shape.(0) <> batch || logits_desc.shape.(1) <> max_seq then
    Error.raise
      (Error.Unsupported_program
         (Printf.sprintf
            "causal_lm.greedy_decode expected logits shape [%d;%d;vocab], got %s"
            batch max_seq (Nx_core.Shape.to_string logits_desc.shape)));
  let vocab = logits_desc.shape.(2) in
  let signature = Signature.of_tensors ~backend ~device_id [ input_ids ] in
  let extra_inputs =
    List.map
      (fun (lifted : Ir.lifted_constant) ->
        (lifted.desc, Runtime.data_string_of_literal lifted.literal))
      lifted_constants
  in
  let prompt_desc = desc [| batch; prompt_len |] "int32" in
  let tokens_desc = desc [| batch; max_seq |] "int32" in
  let tail_desc = desc [| batch; max_tokens |] "int32" in
  let scalar_i32 = desc [||] "int32" in
  let logits_row_desc = desc [| batch; 1; vocab |] logits_desc.dtype in
  let logits_flat_desc = desc [| batch; vocab |] logits_desc.dtype in
  let next_desc = desc [| batch |] "int32" in
  let next_patch_desc = desc [| batch; 1 |] "int32" in
  let prompt_ty = Stablehlo.tensor_type prompt_desc in
  let tokens_ty = Stablehlo.tensor_type tokens_desc in
  let tail_ty = Stablehlo.tensor_type tail_desc in
  let scalar_i32_ty = Stablehlo.tensor_type scalar_i32 in
  let logits_ty = Stablehlo.tensor_type logits_desc in
  let logits_row_ty = Stablehlo.tensor_type logits_row_desc in
  let logits_flat_ty = Stablehlo.tensor_type logits_flat_desc in
  let next_ty = Stablehlo.tensor_type next_desc in
  let next_patch_ty = Stablehlo.tensor_type next_patch_desc in
  let params =
    ("%arg0: " ^ prompt_ty)
    :: (Ir.parameters program
       |> List.filter_map (fun (index, _, desc) ->
              if index = 0 then None
              else
                Some
                  (Printf.sprintf "%%arg%d: %s" index
                     (Stablehlo.tensor_type desc))))
    |> String.concat ", "
  in
  let tokens_init_lines =
    if max_tokens = 0 then
      [
        Printf.sprintf "  %%tokens0 = stablehlo.reshape %%arg0 : (%s) -> %s"
          prompt_ty tokens_ty;
      ]
    else
      [
        Printf.sprintf "  %%tail_zeros = stablehlo.constant dense<0> : %s"
          tail_ty;
        Printf.sprintf
          "  %%tokens0 = \"stablehlo.concatenate\"(%%arg0, %%tail_zeros) \
           {dimension = 1 : i64} : (%s, %s) -> %s"
          prompt_ty tail_ty tokens_ty;
      ]
  in
  let body_lines =
    [
      Printf.sprintf "  %%zero_i32 = stablehlo.constant dense<0> : %s"
        scalar_i32_ty;
      Printf.sprintf "  %%one_i32 = stablehlo.constant dense<1> : %s"
        scalar_i32_ty;
      Printf.sprintf "  %%max_steps = stablehlo.constant dense<%d> : %s"
        max_tokens scalar_i32_ty;
      Printf.sprintf "  %%pos0 = stablehlo.constant dense<%d> : %s"
        (prompt_len - 1) scalar_i32_ty;
    ]
    @ tokens_init_lines
    @ [
        Printf.sprintf "  %%count0 = stablehlo.constant dense<0> : %s"
          scalar_i32_ty;
        Printf.sprintf
          "  %%loop:3 = stablehlo.while(%%iterCount = %%count0, %%iterPos = \
           %%pos0, %%iterTokens = %%tokens0) : %s, %s, %s"
          scalar_i32_ty scalar_i32_ty tokens_ty;
        "    cond {";
        Printf.sprintf
          "      %%keep_going = stablehlo.compare  LT, %%iterCount, %%max_steps \
           : (%s, %s) -> tensor<i1>"
          scalar_i32_ty scalar_i32_ty;
        "      stablehlo.return %keep_going : tensor<i1>";
        "    } do {";
      ]
    @ lowered.body
    @ [
        Printf.sprintf
          "      %%logits_row = \"stablehlo.dynamic_slice\"(%s, %%zero_i32, \
           %%iterPos, %%zero_i32) {slice_sizes = array<i64: %d, 1, %d>} : \
           (%s, %s, %s, %s) -> %s"
          logits_ref batch vocab logits_ty scalar_i32_ty scalar_i32_ty
          scalar_i32_ty logits_row_ty;
        Printf.sprintf
          "      %%logits_flat = stablehlo.reshape %%logits_row : (%s) -> %s"
          logits_row_ty logits_flat_ty;
      ]
    @ argmax_lines ~indent:"      " ~base:"next_token" ~input:"%logits_flat"
        ~input_desc:logits_flat_desc ~axis:1
    @ [
        Printf.sprintf
          "      %%next_patch = stablehlo.reshape %%next_token : (%s) -> %s"
          next_ty next_patch_ty;
        Printf.sprintf
          "      %%next_pos = stablehlo.add %%iterPos, %%one_i32 : %s"
          scalar_i32_ty;
        Printf.sprintf
          "      %%next_tokens = \"stablehlo.dynamic_update_slice\"(%%iterTokens, \
           %%next_patch, %%zero_i32, %%next_pos) : (%s, %s, %s, %s) -> %s"
          tokens_ty next_patch_ty scalar_i32_ty scalar_i32_ty tokens_ty;
        Printf.sprintf
          "      %%next_count = stablehlo.add %%iterCount, %%one_i32 : %s"
          scalar_i32_ty;
        Printf.sprintf
          "      stablehlo.return %%next_count, %%next_pos, %%next_tokens : %s, \
           %s, %s"
          scalar_i32_ty scalar_i32_ty tokens_ty;
        "    }";
        Printf.sprintf "  func.return %%loop#2 : %s" tokens_ty;
      ]
  in
  let module_text =
    String.concat "\n"
      ([ "module {";
         Printf.sprintf "func.func @main(%s) -> %s {" params tokens_ty ]
      @ body_lines @ [ "}"; "}" ])
  in
  Runtime.compile_stablehlo ~backend ~device_id ~signature ~module_text
    ~output_descs:[ tokens_desc ] ~extra_inputs

let greedy_decode ?(backend = `Cuda) ?(device_id = 0) ~max_tokens
    (forward : (int32, 'a) Nx.t -> ('b, 'c) Nx.t) =
  let cache = Hashtbl.create 8 in
  fun (input_ids : (int32, 'a) Nx.t) ->
    let signature = Signature.of_tensors ~backend ~device_id [ input_ids ] in
    let key = Signature.key signature in
    let compiled =
      match Hashtbl.find_opt cache key with
      | Some compiled -> compiled
      | None ->
          let compiled = compile ~backend ~device_id ~max_tokens forward input_ids in
          Hashtbl.replace cache key compiled;
          compiled
    in
    match Runtime.execute compiled [ input_ids ] with
    | [ Trace.Tensor tokens ] -> (Obj.magic tokens : (int32, 'a) Nx.t)
    | _ ->
        Error.raise
          (Error.Runtime_unavailable
             "causal_lm.greedy_decode returned an unexpected output arity")
