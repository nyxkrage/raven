(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun
module Core = Hugr_core
module Config = Falcon_config

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type config = Config.t = {
  vocab_size : int;
  hidden_size : int;
  ffn_hidden_size : int;
  num_hidden_layers : int;
  num_attention_heads : int;
  num_key_value_heads : int;
  max_position_embeddings : int;
  layer_norm_epsilon : float;
  rope_theta : float;
  hidden_dropout : float;
  attention_dropout : float;
  initializer_range : float;
  bias : bool;
  new_decoder_architecture : bool;
  multi_query : bool;
  num_ln_in_parallel_attn : int;
  tie_word_embeddings : bool;
}

let config = Config.make
let position_ids_key = Core.Rope.position_ids_key
let attention_mask_key = Core.Mask.attention_mask_key
let weight_init cfg = Init.normal ~stddev:cfg.initializer_range ()
let head_dim cfg = cfg.hidden_size / cfg.num_attention_heads

let embedding cfg =
  Core.Embedding.token ~vocab_size:cfg.vocab_size ~hidden_size:cfg.hidden_size
    ~weight_init:(weight_init cfg) ()

let norm cfg =
  Layer.layer_norm ~dim:cfg.hidden_size ~eps:cfg.layer_norm_epsilon ()

let attention cfg =
  Core.Dense_attention.self_attention ~hidden_size:cfg.hidden_size
    ~num_heads:cfg.num_attention_heads ~num_kv_heads:cfg.num_key_value_heads
    ~head_dim:(head_dim cfg) ~rope_theta:cfg.rope_theta
    ~dropout:cfg.attention_dropout ~bias:cfg.bias ~weight_init:(weight_init cfg)
    ()

let mlp cfg =
  let dense_h_to_4h =
    Core.Projection.linear ~in_features:cfg.hidden_size
      ~out_features:cfg.ffn_hidden_size ~bias:cfg.bias
      ~weight_init:(weight_init cfg) ()
  in
  let dense_4h_to_h =
    Core.Projection.linear ~in_features:cfg.ffn_hidden_size
      ~out_features:cfg.hidden_size ~bias:cfg.bias
      ~weight_init:(weight_init cfg) ()
  in
  {
    Layer.init =
      (fun ~dtype ->
        Core.Layer_util.init_children dtype
          [
            ("dense_h_to_4h", dense_h_to_4h.Layer.init);
            ("dense_4h_to_h", dense_4h_to_h.Layer.init);
          ]);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        let hidden, first_state =
          Core.Layer_util.apply_child ~ctx:"Falcon.mlp" dense_h_to_4h
            ~name:"dense_h_to_4h" ~params ~state ~dtype ~training ?call_ctx:ctx
            x
        in
        let hidden = Activation.gelu hidden in
        let output, second_state =
          Core.Layer_util.apply_child ~ctx:"Falcon.mlp" dense_4h_to_h
            ~name:"dense_4h_to_h" ~params ~state ~dtype ~training ?call_ctx:ctx
            hidden
        in
        ( output,
          Core.Layer_util.merge_state
            ~names:[ "dense_h_to_4h"; "dense_4h_to_h" ]
            [ first_state; second_state ] ));
  }

let decoder_block cfg () =
  let self_attention = attention cfg in
  let mlp = mlp cfg in
  let first_norm = norm cfg in
  let second_norm =
    if cfg.num_ln_in_parallel_attn = 2 then Some (norm cfg) else None
  in
  let norm_names =
    match second_norm with
    | None -> [ "input_layernorm" ]
    | Some _ -> [ "ln_attn"; "ln_mlp" ]
  in
  let names = norm_names @ [ "self_attention"; "mlp" ] in
  {
    Layer.init =
      (fun ~dtype ->
        let norm_children =
          match second_norm with
          | None -> [ ("input_layernorm", first_norm.Layer.init) ]
          | Some second ->
              [
                ("ln_attn", first_norm.Layer.init); ("ln_mlp", second.Layer.init);
              ]
        in
        Core.Layer_util.init_children dtype
          (norm_children
          @ [
              ("self_attention", self_attention.Layer.init);
              ("mlp", mlp.Layer.init);
            ]));
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        let op_ctx = "Falcon.decoder_block" in
        let x = Core.Layer_util.require_same_float_dtype ~ctx:op_ctx dtype x in
        let apply layer name input =
          Core.Layer_util.apply_child ~ctx:op_ctx layer ~name ~params ~state
            ~dtype ~training ?call_ctx:ctx input
        in
        let attention_input, mlp_input, norm_states =
          match second_norm with
          | None ->
              let normalized, norm_state =
                apply first_norm "input_layernorm" x
              in
              (normalized, normalized, [ norm_state ])
          | Some second ->
              let attention_input, attention_norm_state =
                apply first_norm "ln_attn" x
              in
              let mlp_input, mlp_norm_state = apply second "ln_mlp" x in
              ( attention_input,
                mlp_input,
                [ attention_norm_state; mlp_norm_state ] )
        in
        let attended, attention_state =
          apply self_attention "self_attention" attention_input
        in
        let transformed, mlp_state = apply mlp "mlp" mlp_input in
        let combined = Nx.add attended transformed in
        let combined =
          if training && cfg.hidden_dropout > 0.0 then
            Kaun.Fn.dropout ~rate:cfg.hidden_dropout combined
          else combined
        in
        ( Nx.add x combined,
          Core.Layer_util.merge_state ~names
            (norm_states @ [ attention_state; mlp_state ]) ));
  }

let lm_head cfg =
  Core.Projection.linear ~in_features:cfg.hidden_size
    ~out_features:cfg.vocab_size ~bias:false ~weight_init:(weight_init cfg) ()

let init_model_vars ~with_lm_head cfg dtype =
  let embedding_vars =
    Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
        (embedding cfg).Layer.init ~dtype)
  in
  let block = decoder_block cfg () in
  let layer_vars =
    List.init cfg.num_hidden_layers (fun _ ->
        Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () -> block.Layer.init ~dtype))
  in
  let norm_vars =
    Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
        (norm cfg).Layer.init ~dtype)
  in
  let params =
    [
      ("word_embeddings", Layer.params embedding_vars);
      ("h", Ptree.list (List.map Layer.params layer_vars));
      ("ln_f", Layer.params norm_vars);
    ]
  in
  let state =
    [
      ("word_embeddings", Layer.state embedding_vars);
      ("h", Ptree.list (List.map Layer.state layer_vars));
      ("ln_f", Layer.state norm_vars);
    ]
  in
  let params, state =
    if with_lm_head then
      let head_vars =
        Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
            (lm_head cfg).Layer.init ~dtype)
      in
      ( params @ [ ("lm_head", Layer.params head_vars) ],
        state @ [ ("lm_head", Layer.state head_vars) ] )
    else (params, state)
  in
  Layer.make_vars ~params:(Ptree.dict params) ~state:(Ptree.dict state) ~dtype

let decode (type layout input_layout) ~(cfg : config) ~params ~state
    ~(dtype : (float, layout) Nx.dtype) ~training ?ctx
    (input_ids : (int32, input_layout) Nx.t) =
  let params_root = Core.Layer_util.fields ~ctx:"Falcon.decode.params" params in
  let state_root = Core.Layer_util.fields ~ctx:"Falcon.decode.state" state in
  let param name =
    Core.Layer_util.find ~ctx:"Falcon.decode.params" name params_root
  in
  let child_state name =
    Core.Layer_util.find ~ctx:"Falcon.decode.state" name state_root
  in
  let hidden, embedding_state =
    (embedding cfg).Layer.apply ~params:(param "word_embeddings")
      ~state:(child_state "word_embeddings")
      ~dtype ~training ?ctx input_ids
  in
  let layer_params =
    Ptree.List.items_exn ~ctx:"Falcon.decode.params.h" (param "h")
    |> Array.of_list
  in
  let layer_states =
    Ptree.List.items_exn ~ctx:"Falcon.decode.state.h" (child_state "h")
    |> Array.of_list
  in
  if
    Array.length layer_params <> cfg.num_hidden_layers
    || Array.length layer_states <> cfg.num_hidden_layers
  then invalid_argf "Falcon.decode: layer parameter/state count mismatch";
  let block = decoder_block cfg () in
  let output_states = Array.make cfg.num_hidden_layers Ptree.empty in
  let rec apply_layers layer_index hidden =
    if layer_index = cfg.num_hidden_layers then hidden
    else
      let hidden, output_state =
        block.Layer.apply ~params:layer_params.(layer_index)
          ~state:layer_states.(layer_index) ~dtype ~training ?ctx hidden
      in
      output_states.(layer_index) <- output_state;
      apply_layers (layer_index + 1) hidden
  in
  let hidden = apply_layers 0 hidden in
  let hidden, norm_state =
    (norm cfg).Layer.apply ~params:(param "ln_f") ~state:(child_state "ln_f")
      ~dtype ~training ?ctx hidden
  in
  ( hidden,
    Ptree.dict
      [
        ("word_embeddings", embedding_state);
        ("h", Ptree.list (Array.to_list output_states));
        ("ln_f", norm_state);
      ] )

let decoder cfg () =
  {
    Layer.init = (fun ~dtype -> init_model_vars ~with_lm_head:false cfg dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx input_ids ->
        decode ~cfg ~params ~state ~dtype ~training ?ctx input_ids);
  }

let for_causal_lm cfg () =
  let use_head = not cfg.tie_word_embeddings in
  {
    Layer.init =
      (fun ~dtype -> init_model_vars ~with_lm_head:use_head cfg dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx input_ids ->
        let original_state = state in
        let hidden, decoder_state =
          decode ~cfg ~params ~state ~dtype ~training ?ctx input_ids
        in
        let params_root =
          Core.Layer_util.fields ~ctx:"Falcon.for_causal_lm.params" params
        in
        if cfg.tie_word_embeddings then
          let embeddings =
            Core.Layer_util.find ~ctx:"Falcon.for_causal_lm.params"
              "word_embeddings" params_root
            |> Core.Layer_util.fields
                 ~ctx:"Falcon.for_causal_lm.params.word_embeddings"
          in
          let weight = Core.Layer_util.get embeddings ~name:"weight" dtype in
          (Nx.matmul hidden (Nx.transpose weight ~axes:[ 1; 0 ]), decoder_state)
        else
          let state_root =
            Core.Layer_util.fields ~ctx:"Falcon.for_causal_lm.state"
              original_state
          in
          let logits, head_state =
            (lm_head cfg).Layer.apply
              ~params:
                (Core.Layer_util.find ~ctx:"Falcon.for_causal_lm.params"
                   "lm_head" params_root)
              ~state:
                (Core.Layer_util.find ~ctx:"Falcon.for_causal_lm.state"
                   "lm_head" state_root)
              ~dtype ~training ?ctx hidden
          in
          let decoder_fields =
            Core.Layer_util.fields ~ctx:"Falcon.for_causal_lm.decoder_state"
              decoder_state
          in
          (logits, Ptree.dict (decoder_fields @ [ ("lm_head", head_state) ])));
  }

module Cache = struct
  type 'layout t = 'layout Dense_cache.t

  let create cfg ~batch_size ~max_length ~dtype =
    Dense_cache.create ~num_layers:cfg.num_hidden_layers
      ~num_kv_heads:cfg.num_key_value_heads ~head_dim:(head_dim cfg)
      ~max_position_embeddings:cfg.max_position_embeddings ~batch_size
      ~max_length ~dtype

  let batch_size = Dense_cache.batch_size
  let max_length = Dense_cache.max_length
  let length = Dense_cache.length
end

let cached_decoder_block cfg ~params ~state ~dtype ~position ~valid ~key_cache
    ~value_cache x =
  let op_ctx = "Falcon.cached_decoder_block" in
  let mlp = mlp cfg in
  let first_norm = norm cfg in
  let second_norm =
    if cfg.num_ln_in_parallel_attn = 2 then Some (norm cfg) else None
  in
  let apply layer name input =
    Core.Layer_util.apply_child ~ctx:op_ctx layer ~name ~params ~state ~dtype
      ~training:false input
  in
  let attention_input, mlp_input, norm_names, norm_states =
    match second_norm with
    | None ->
        let normalized, norm_state = apply first_norm "input_layernorm" x in
        (normalized, normalized, [ "input_layernorm" ], [ norm_state ])
    | Some second ->
        let attention_input, attention_norm_state =
          apply first_norm "ln_attn" x
        in
        let mlp_input, mlp_norm_state = apply second "ln_mlp" x in
        ( attention_input,
          mlp_input,
          [ "ln_attn"; "ln_mlp" ],
          [ attention_norm_state; mlp_norm_state ] )
  in
  let attention_params, attention_input_state =
    Core.Layer_util.child_vars ~ctx:op_ctx ~params ~state "self_attention"
  in
  let attended, attention_state, key_cache, value_cache =
    Core.Dense_attention.cached_self_attention ~hidden_size:cfg.hidden_size
      ~num_heads:cfg.num_attention_heads ~num_kv_heads:cfg.num_key_value_heads
      ~head_dim:(head_dim cfg) ~rope_theta:cfg.rope_theta
      ~dropout:cfg.attention_dropout ~bias:cfg.bias ~params:attention_params
      ~state:attention_input_state ~dtype ~training:false ~position ~valid
      ~key_cache ~value_cache attention_input
  in
  let transformed, mlp_state = apply mlp "mlp" mlp_input in
  let state =
    Core.Layer_util.merge_state
      ~names:(norm_names @ [ "self_attention"; "mlp" ])
      (norm_states @ [ attention_state; mlp_state ])
  in
  (Nx.add x (Nx.add attended transformed), state, key_cache, value_cache)

let cached_causal_lm ~cfg ~params ~state ~dtype ?attention_mask cache input_ids
    =
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 then
    invalid_argf "Falcon.cached_causal_lm: input IDs must have rank 2";
  let batch = shape.(0) in
  let seq = shape.(1) in
  if batch <> cache.Dense_cache.batch_size then
    invalid_argf "Falcon.cached_causal_lm: batch size mismatch";
  if seq <= 0 || cache.length + seq > cache.max_length then
    invalid_argf "Falcon.cached_causal_lm: invalid sequence length or capacity";
  let token_valid =
    match attention_mask with
    | None -> Nx.ones Nx.bool [| batch; seq |]
    | Some mask ->
        if Nx.shape mask <> [| batch; seq |] then
          invalid_argf "Falcon.cached_causal_lm: attention mask shape mismatch";
        mask
  in
  let valid = Dense_cache.append_valid cache token_valid seq in
  let params_root =
    Core.Layer_util.fields ~ctx:"Falcon.cached_causal_lm.params" params
  in
  let state_root =
    Core.Layer_util.fields ~ctx:"Falcon.cached_causal_lm.state" state
  in
  let param name =
    Core.Layer_util.find ~ctx:"Falcon.cached_causal_lm.params" name params_root
  in
  let child_state name =
    Core.Layer_util.find ~ctx:"Falcon.cached_causal_lm.state" name state_root
  in
  let hidden, embedding_state =
    (embedding cfg).Layer.apply ~params:(param "word_embeddings")
      ~state:(child_state "word_embeddings")
      ~dtype ~training:false input_ids
  in
  let layer_params =
    Ptree.List.items_exn ~ctx:"Falcon.cached_causal_lm.params.h" (param "h")
    |> Array.of_list
  in
  let layer_states =
    Ptree.List.items_exn ~ctx:"Falcon.cached_causal_lm.state.h"
      (child_state "h")
    |> Array.of_list
  in
  let keys = Array.copy cache.keys in
  let values = Array.copy cache.values in
  let output_states = Array.make cfg.num_hidden_layers Ptree.empty in
  let rec apply_layers layer_index hidden =
    if layer_index = cfg.num_hidden_layers then hidden
    else
      let hidden, output_state, key, value =
        cached_decoder_block cfg ~params:layer_params.(layer_index)
          ~state:layer_states.(layer_index) ~dtype ~position:cache.position
          ~valid ~key_cache:keys.(layer_index) ~value_cache:values.(layer_index)
          hidden
      in
      keys.(layer_index) <- key;
      values.(layer_index) <- value;
      output_states.(layer_index) <- output_state;
      apply_layers (layer_index + 1) hidden
  in
  let hidden = apply_layers 0 hidden in
  let hidden, norm_state =
    (norm cfg).Layer.apply ~params:(param "ln_f") ~state:(child_state "ln_f")
      ~dtype ~training:false hidden
  in
  let decoder_state =
    [
      ("word_embeddings", embedding_state);
      ("h", Ptree.list (Array.to_list output_states));
      ("ln_f", norm_state);
    ]
  in
  let logits, output_state =
    if cfg.tie_word_embeddings then
      let embeddings =
        Core.Layer_util.fields ~ctx:"Falcon.cached_causal_lm.word_embeddings"
          (param "word_embeddings")
      in
      let weight = Core.Layer_util.get embeddings ~name:"weight" dtype in
      ( Nx.matmul hidden (Nx.transpose weight ~axes:[ 1; 0 ]),
        Ptree.dict decoder_state )
    else
      let logits, head_state =
        (lm_head cfg).Layer.apply ~params:(param "lm_head")
          ~state:(child_state "lm_head") ~dtype ~training:false hidden
      in
      (logits, Ptree.dict (decoder_state @ [ ("lm_head", head_state) ]))
  in
  let cache =
    {
      cache with
      Dense_cache.keys;
      values;
      valid;
      position = Nx.add cache.position (Nx.scalar Nx.int32 (Int32.of_int seq));
      length = cache.length + seq;
    }
  in
  (logits, output_state, cache)

let prefill cfg vars cache ?attention_mask input_ids =
  let logits, _, cache =
    cached_causal_lm ~cfg ~params:(Layer.params vars) ~state:(Layer.state vars)
      ~dtype:(Layer.dtype vars) ?attention_mask cache input_ids
  in
  (logits, cache)

let decode_step cfg vars cache input_ids =
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 || shape.(1) <> 1 then
    invalid_argf "Falcon.decode_step: input IDs must have shape [batch; 1]";
  prefill cfg vars cache input_ids

module Pjrt = struct
  type 'layout t = 'layout Dense_pjrt.t

  let compile ?(device_id = 0) cfg vars =
    let dtype = Layer.dtype vars in
    let params = Layer.params vars in
    let state = Layer.state vars in
    Dense_pjrt.compile ~device_id ~layer_count:cfg.num_hidden_layers ~dtype
      (fun ~attention_mask cache input_ids ->
        let logits, _, cache =
          cached_causal_lm ~cfg ~params ~state ~dtype ~attention_mask cache
            input_ids
        in
        (logits, cache))

  let prefill = Dense_pjrt.prefill
  let decode_step = Dense_pjrt.decode_step
end

let qkv_indices cfg =
  let dim = head_dim cfg in
  let q = ref [] in
  let k = ref [] in
  let v = ref [] in
  let add_range target first length =
    for index = first to first + length - 1 do
      target := index :: !target
    done
  in
  if cfg.new_decoder_architecture then
    let queries_per_kv = cfg.num_attention_heads / cfg.num_key_value_heads in
    let group_width = (queries_per_kv + 2) * dim in
    for group = 0 to cfg.num_key_value_heads - 1 do
      let first = group * group_width in
      add_range q first (queries_per_kv * dim);
      add_range k (first + (queries_per_kv * dim)) dim;
      add_range v (first + ((queries_per_kv + 1) * dim)) dim
    done
  else if cfg.multi_query then (
    add_range q 0 cfg.hidden_size;
    add_range k cfg.hidden_size dim;
    add_range v (cfg.hidden_size + dim) dim)
  else
    for head = 0 to cfg.num_attention_heads - 1 do
      let first = head * 3 * dim in
      add_range q first dim;
      add_range k (first + dim) dim;
      add_range v (first + (2 * dim)) dim
    done;
  ( Array.of_list (List.rev !q),
    Array.of_list (List.rev !k),
    Array.of_list (List.rev !v) )

let map_hf_weights ~cfg ~dtype tensors =
  let module Hf = Core.Hf in
  let weights = Hf.weights tensors in
  let vector name size =
    Hf.tensor weights ~name ~shape:[| size |] |> Hf.cast dtype
  in
  let matrix name ~rows ~cols = Hf.matrix weights dtype ~name ~rows ~cols in
  let projection name ~in_features ~out_features =
    let fields =
      [
        ( "weight",
          matrix (name ^ ".weight") ~rows:out_features ~cols:in_features );
      ]
    in
    let fields =
      if cfg.bias then
        fields @ [ ("bias", vector (name ^ ".bias") out_features) ]
      else fields
    in
    Ptree.dict fields
  in
  let q_indices, k_indices, v_indices = qkv_indices cfg in
  let qkv_out =
    if cfg.new_decoder_architecture then
      (cfg.num_attention_heads + (2 * cfg.num_key_value_heads)) * head_dim cfg
    else if cfg.multi_query then cfg.hidden_size + (2 * head_dim cfg)
    else 3 * cfg.hidden_size
  in
  let layer layer_index =
    let prefix = Printf.sprintf "transformer.h.%d" layer_index in
    let (Ptree.P fused_weight) =
      Hf.tensor weights
        ~name:(prefix ^ ".self_attention.query_key_value.weight")
        ~shape:[| qkv_out; cfg.hidden_size |]
    in
    let fused_weight = Nx.cast dtype fused_weight in
    let fused_bias =
      if cfg.bias then
        let (Ptree.P bias) =
          Hf.tensor weights
            ~name:(prefix ^ ".self_attention.query_key_value.bias")
            ~shape:[| qkv_out |]
        in
        Some (Nx.cast dtype bias)
      else None
    in
    let split_projection indices =
      let indices =
        Array.map Int32.of_int indices
        |> Nx.create Nx.int32 [| Array.length indices |]
      in
      let fields =
        [
          ( "weight",
            Nx.take ~axis:0 indices fused_weight
            |> Nx.transpose ~axes:[ 1; 0 ]
            |> Ptree.tensor );
        ]
      in
      let fields =
        match fused_bias with
        | None -> fields
        | Some bias ->
            fields @ [ ("bias", Nx.take ~axis:0 indices bias |> Ptree.tensor) ]
      in
      Ptree.dict fields
    in
    let layer_norm name =
      Ptree.dict
        [
          ("gamma", vector (prefix ^ "." ^ name ^ ".weight") cfg.hidden_size);
          ("beta", vector (prefix ^ "." ^ name ^ ".bias") cfg.hidden_size);
        ]
    in
    let norm_fields =
      if cfg.num_ln_in_parallel_attn = 2 then
        [ ("ln_attn", layer_norm "ln_attn"); ("ln_mlp", layer_norm "ln_mlp") ]
      else [ ("input_layernorm", layer_norm "input_layernorm") ]
    in
    Ptree.dict
      (norm_fields
      @ [
          ( "self_attention",
            Ptree.dict
              [
                ("q_proj", split_projection q_indices);
                ("k_proj", split_projection k_indices);
                ("v_proj", split_projection v_indices);
                ( "o_proj",
                  projection
                    (prefix ^ ".self_attention.dense")
                    ~in_features:cfg.hidden_size ~out_features:cfg.hidden_size
                );
              ] );
          ( "mlp",
            Ptree.dict
              [
                ( "dense_h_to_4h",
                  projection
                    (prefix ^ ".mlp.dense_h_to_4h")
                    ~in_features:cfg.hidden_size
                    ~out_features:cfg.ffn_hidden_size );
                ( "dense_4h_to_h",
                  projection
                    (prefix ^ ".mlp.dense_4h_to_h")
                    ~in_features:cfg.ffn_hidden_size
                    ~out_features:cfg.hidden_size );
              ] );
        ])
  in
  let params =
    [
      ( "word_embeddings",
        Ptree.dict
          [
            ( "weight",
              Hf.tensor weights ~name:"transformer.word_embeddings.weight"
                ~shape:[| cfg.vocab_size; cfg.hidden_size |]
              |> Hf.cast dtype );
          ] );
      ("h", Ptree.list (List.init cfg.num_hidden_layers layer));
      ( "ln_f",
        Ptree.dict
          [
            ("gamma", vector "transformer.ln_f.weight" cfg.hidden_size);
            ("beta", vector "transformer.ln_f.bias" cfg.hidden_size);
          ] );
    ]
  in
  let params =
    if cfg.tie_word_embeddings then params
    else
      params
      @ [
          ( "lm_head",
            Ptree.dict
              [
                ( "weight",
                  matrix "lm_head.weight" ~rows:cfg.vocab_size
                    ~cols:cfg.hidden_size );
              ] );
        ]
  in
  Hf.ensure_consumed weights ~allow:(fun name ->
      cfg.tie_word_embeddings && name = "lm_head.weight");
  Ptree.dict params

let from_pretrained ?token ?cache_dir ?offline ?revision ~model_id ~dtype () =
  let json =
    Kaun_hf.load_config ?token ?cache_dir ?offline ?revision ~model_id ()
  in
  let cfg = Config.of_json json in
  let weights =
    Kaun_hf.load_weights ?token ?cache_dir ?offline ?revision ~model_id ()
  in
  (cfg, map_hf_weights ~cfg ~dtype weights)
