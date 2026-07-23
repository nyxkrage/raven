(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let validate_config ~ctx ~num_heads ~num_kv_heads ~head_dim ~dropout =
  if num_heads <= 0 then invalid_argf "%s: num_heads must be positive" ctx;
  if num_kv_heads <= 0 then invalid_argf "%s: num_kv_heads must be positive" ctx;
  if num_heads mod num_kv_heads <> 0 then
    invalid_argf "%s: num_heads (%d) must be divisible by num_kv_heads (%d)" ctx
      num_heads num_kv_heads;
  if head_dim <= 0 || head_dim mod 2 <> 0 then
    invalid_argf "%s: head_dim must be positive and even, got %d" ctx head_dim;
  if dropout < 0.0 || dropout >= 1.0 then
    invalid_argf "%s: expected 0.0 <= dropout < 1.0, got %g" ctx dropout

let projections ~hidden_size ~num_heads ~num_kv_heads ~head_dim ~bias
    ?weight_init () =
  let projection out_features =
    Projection.linear ~in_features:hidden_size ~out_features ~bias ?weight_init
      ()
  in
  ( projection (num_heads * head_dim),
    projection (num_kv_heads * head_dim),
    projection (num_kv_heads * head_dim),
    Projection.linear ~in_features:(num_heads * head_dim)
      ~out_features:hidden_size ~bias ?weight_init () )

let reshape_heads ~batch ~seq ~heads ~head_dim t =
  Nx.reshape [| batch; seq; heads; head_dim |] t
  |> Nx.transpose ~axes:[ 0; 2; 1; 3 ]

let repeat_kv ~batch ~seq ~num_heads ~num_kv_heads ~head_dim t =
  if num_kv_heads = num_heads then t
  else
    let repetitions = num_heads / num_kv_heads in
    Nx.expand_dims [ 2 ] t
    |> Nx.broadcast_to [| batch; num_kv_heads; repetitions; seq; head_dim |]
    |> Nx.contiguous
    |> Nx.reshape [| batch; num_heads; seq; head_dim |]

let attend ~dtype ~training ~dropout ~head_dim ~score_scale ~logit_softcap ~mask
    ~q ~k ~v =
  let scale =
    Option.value score_scale
      ~default:(1.0 /. Stdlib.sqrt (float_of_int head_dim))
  in
  let scores =
    Nx.matmul q (Nx.transpose k ~axes:[ 0; 1; 3; 2 ]) |> fun scores ->
    Nx.mul scores (Nx.scalar dtype scale)
  in
  let scores =
    match logit_softcap with
    | None -> scores
    | Some cap ->
        let cap = Nx.scalar dtype cap in
        Nx.mul cap (Nx.tanh (Nx.div scores cap))
  in
  let scores = Nx.where mask scores (Nx.scalar dtype (-1.0 *. 1e9)) in
  let probabilities = Nx.softmax ~axes:[ -1 ] scores in
  let probabilities =
    if training && dropout > 0.0 then
      Kaun.Fn.dropout ~rate:dropout probabilities
    else probabilities
  in
  Nx.matmul probabilities v

let boolean_and left right = Nx.where left right (Nx.scalar Nx.bool false)

let any_bool ~axes value =
  Nx.sum ~axes (Nx.cast Nx.int32 value) |> fun count ->
  Nx.greater count (Nx.scalar Nx.int32 0l)

let self_attention ~hidden_size ~num_heads ~num_kv_heads ~head_dim ~rope_theta
    ?window ?score_scale ?logit_softcap ?qk_norm_eps ?(dropout = 0.0)
    ?(bias = false) ?weight_init () =
  let op_ctx = "Dense_attention.self_attention" in
  validate_config ~ctx:op_ctx ~num_heads ~num_kv_heads ~head_dim ~dropout;
  Option.iter
    (fun value ->
      if value <= 0.0 then
        invalid_argf "%s: score_scale must be positive, got %g" op_ctx value)
    score_scale;
  Option.iter
    (fun value ->
      if value <= 0.0 then
        invalid_argf "%s: logit_softcap must be positive, got %g" op_ctx value)
    logit_softcap;
  let q_proj, k_proj, v_proj, o_proj =
    projections ~hidden_size ~num_heads ~num_kv_heads ~head_dim ~bias
      ?weight_init ()
  in
  let q_norm, k_norm =
    match qk_norm_eps with
    | None -> (None, None)
    | Some eps ->
        ( Some (Norm.gemma_rms ~dim:head_dim ~eps ()),
          Some (Norm.gemma_rms ~dim:head_dim ~eps ()) )
  in
  let names =
    [ "q_proj"; "k_proj"; "v_proj"; "o_proj" ]
    @ match qk_norm_eps with None -> [] | Some _ -> [ "q_norm"; "k_norm" ]
  in
  {
    Kaun.Layer.init =
      (fun ~dtype ->
        let children =
          [
            ("q_proj", q_proj.Kaun.Layer.init);
            ("k_proj", k_proj.Kaun.Layer.init);
            ("v_proj", v_proj.Kaun.Layer.init);
            ("o_proj", o_proj.Kaun.Layer.init);
          ]
          @
          match (q_norm, k_norm) with
          | Some q_norm, Some k_norm ->
              [
                ("q_norm", q_norm.Kaun.Layer.init);
                ("k_norm", k_norm.Kaun.Layer.init);
              ]
          | _ -> []
        in
        Layer_util.init_children dtype children);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        let x = Layer_util.require_same_float_dtype ~ctx:op_ctx dtype x in
        let shape = Nx.shape x in
        if Array.length shape <> 3 || shape.(2) <> hidden_size then
          invalid_argf "%s: expected [batch; seq; %d], got [%s]" op_ctx
            hidden_size
            (String.concat "; " (List.map string_of_int (Array.to_list shape)));
        let batch = shape.(0) in
        let seq = shape.(1) in
        let apply projection name =
          Layer_util.apply_child ~ctx:op_ctx projection ~name ~params ~state
            ~dtype ~training ?call_ctx:ctx x
        in
        let q, q_state = apply q_proj "q_proj" in
        let k, k_state = apply k_proj "k_proj" in
        let v, v_state = apply v_proj "v_proj" in
        let q = reshape_heads ~batch ~seq ~heads:num_heads ~head_dim q in
        let k = reshape_heads ~batch ~seq ~heads:num_kv_heads ~head_dim k in
        let v = reshape_heads ~batch ~seq ~heads:num_kv_heads ~head_dim v in
        let q, q_norm_state, k, k_norm_state =
          match (q_norm, k_norm) with
          | Some q_norm, Some k_norm ->
              let q, q_state =
                Layer_util.apply_child ~ctx:op_ctx q_norm ~name:"q_norm" ~params
                  ~state ~dtype ~training ?call_ctx:ctx q
              in
              let k, k_state =
                Layer_util.apply_child ~ctx:op_ctx k_norm ~name:"k_norm" ~params
                  ~state ~dtype ~training ?call_ctx:ctx k
              in
              (q, [ q_state ], k, [ k_state ])
          | _ -> (q, [], k, [])
        in
        let positions = Rope.positions_from_context ~batch ~seq ctx in
        let q =
          Rope.apply ~theta:rope_theta ~rotary_dim:head_dim ~positions q
        in
        let k =
          Rope.apply ~theta:rope_theta ~rotary_dim:head_dim ~positions k
        in
        let k = repeat_kv ~batch ~seq ~num_heads ~num_kv_heads ~head_dim k in
        let v = repeat_kv ~batch ~seq ~num_heads ~num_kv_heads ~head_dim v in
        let mask =
          Mask.combined ?window ~batch ~query_len:seq ~key_len:seq ctx
        in
        let attended =
          attend ~dtype ~training ~dropout ~head_dim ~score_scale ~logit_softcap
            ~mask ~q ~k ~v
        in
        let attended =
          Nx.transpose attended ~axes:[ 0; 2; 1; 3 ]
          |> Nx.contiguous
          |> Nx.reshape [| batch; seq; num_heads * head_dim |]
        in
        let output, o_state =
          Layer_util.apply_child ~ctx:op_ctx o_proj ~name:"o_proj" ~params
            ~state ~dtype ~training ?call_ctx:ctx attended
        in
        let state =
          Layer_util.merge_state ~names
            ([ q_state; k_state; v_state; o_state ]
            @ q_norm_state @ k_norm_state)
        in
        (output, state));
  }

let cached_self_attention ~hidden_size ~num_heads ~num_kv_heads ~head_dim
    ~rope_theta ?window ?score_scale ?logit_softcap ?qk_norm_eps
    ?(dropout = 0.0) ?(bias = false) ~params ~state ~dtype ~training ~position
    ~valid ~key_cache ~value_cache x =
  let ctx = "Dense_attention.cached_self_attention" in
  validate_config ~ctx ~num_heads ~num_kv_heads ~head_dim ~dropout;
  let x = Layer_util.require_same_float_dtype ~ctx dtype x in
  let shape = Nx.shape x in
  if Array.length shape <> 3 || shape.(2) <> hidden_size then
    invalid_argf "%s: expected [batch; seq; %d], got [%s]" ctx hidden_size
      (String.concat "; " (List.map string_of_int (Array.to_list shape)));
  let batch = shape.(0) in
  let seq = shape.(1) in
  let cache_shape = Nx.shape key_cache in
  if
    Array.length cache_shape <> 4
    || cache_shape.(0) <> batch
    || cache_shape.(1) <> num_kv_heads
    || cache_shape.(3) <> head_dim
  then
    invalid_argf "%s: invalid key cache shape [%s]" ctx
      (String.concat "; " (List.map string_of_int (Array.to_list cache_shape)));
  if Nx.shape value_cache <> cache_shape then
    invalid_argf "%s: key and value cache shapes differ" ctx;
  let max_length = cache_shape.(2) in
  if Nx.shape valid <> [| batch; max_length |] then
    invalid_argf "%s: valid mask must have shape [%d; %d]" ctx batch max_length;
  if Nx.shape position <> [||] then
    invalid_argf "%s: position must be a scalar" ctx;
  let q_proj, k_proj, v_proj, o_proj =
    projections ~hidden_size ~num_heads ~num_kv_heads ~head_dim ~bias ()
  in
  let q_norm, k_norm =
    match qk_norm_eps with
    | None -> (None, None)
    | Some eps ->
        ( Some (Norm.gemma_rms ~dim:head_dim ~eps ()),
          Some (Norm.gemma_rms ~dim:head_dim ~eps ()) )
  in
  let apply projection name input =
    Layer_util.apply_child ~ctx projection ~name ~params ~state ~dtype ~training
      input
  in
  let q, q_state = apply q_proj "q_proj" x in
  let k, k_state = apply k_proj "k_proj" x in
  let v, v_state = apply v_proj "v_proj" x in
  let q = reshape_heads ~batch ~seq ~heads:num_heads ~head_dim q in
  let k = reshape_heads ~batch ~seq ~heads:num_kv_heads ~head_dim k in
  let v = reshape_heads ~batch ~seq ~heads:num_kv_heads ~head_dim v in
  let q, q_norm_state, k, k_norm_state =
    match (q_norm, k_norm) with
    | Some q_norm, Some k_norm ->
        let q, q_state = apply q_norm "q_norm" q in
        let k, k_state = apply k_norm "k_norm" k in
        (q, [ q_state ], k, [ k_state ])
    | _ -> (q, [], k, [])
  in
  let query_positions =
    Nx.add position (Nx.arange Nx.int32 0 seq 1)
    |> Nx.reshape [| 1; seq |]
    |> Nx.broadcast_to [| batch; seq |]
    |> Nx.contiguous
  in
  let q =
    Rope.apply ~theta:rope_theta ~rotary_dim:head_dim ~positions:query_positions
      q
  in
  let k =
    Rope.apply ~theta:rope_theta ~rotary_dim:head_dim ~positions:query_positions
      k
  in
  let slots = Nx.arange Nx.int32 0 max_length 1 in
  let writes =
    Nx.equal
      (Nx.reshape [| seq; 1 |] (Nx.add position (Nx.arange Nx.int32 0 seq 1)))
      (Nx.reshape [| 1; max_length |] slots)
  in
  let write_values cache values =
    let weights = Nx.cast dtype writes in
    let inserted =
      Nx.matmul (Nx.transpose values ~axes:[ 0; 1; 3; 2 ]) weights
      |> Nx.transpose ~axes:[ 0; 1; 3; 2 ]
    in
    let occupied =
      any_bool ~axes:[ 0 ] writes |> Nx.reshape [| 1; 1; max_length; 1 |]
    in
    Nx.where occupied inserted cache
  in
  let key_cache = write_values key_cache k in
  let value_cache = write_values value_cache v in
  let query_positions = Nx.reshape [| 1; 1; seq; 1 |] query_positions in
  let key_positions = Nx.reshape [| 1; 1; 1; max_length |] slots in
  let mask = Nx.less_equal key_positions query_positions in
  let mask =
    match window with
    | None -> mask
    | Some size ->
        let first =
          Nx.sub query_positions (Nx.scalar Nx.int32 (Int32.of_int (size - 1)))
        in
        boolean_and mask (Nx.greater_equal key_positions first)
  in
  let valid = Nx.reshape [| batch; 1; 1; max_length |] valid in
  let mask =
    boolean_and mask valid |> Nx.broadcast_to [| batch; 1; seq; max_length |]
  in
  let k =
    repeat_kv ~batch ~seq:max_length ~num_heads ~num_kv_heads ~head_dim
      key_cache
  in
  let v =
    repeat_kv ~batch ~seq:max_length ~num_heads ~num_kv_heads ~head_dim
      value_cache
  in
  let attended =
    attend ~dtype ~training ~dropout ~head_dim ~score_scale ~logit_softcap ~mask
      ~q ~k ~v
  in
  let attended =
    Nx.transpose attended ~axes:[ 0; 2; 1; 3 ]
    |> Nx.contiguous
    |> Nx.reshape [| batch; seq; num_heads * head_dim |]
  in
  let output, o_state = apply o_proj "o_proj" attended in
  let state =
    Layer_util.merge_state
      ~names:
        ([ "q_proj"; "k_proj"; "v_proj"; "o_proj" ]
        @ match qk_norm_eps with None -> [] | Some _ -> [ "q_norm"; "k_norm" ])
      ([ q_state; k_state; v_state; o_state ] @ q_norm_state @ k_norm_state)
  in
  (output, state, key_cache, value_cache)
