(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun
module Core = Hugr_core
module Config = Gemma2_config

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type config = Config.t = {
  vocab_size : int;
  hidden_size : int;
  intermediate_size : int;
  num_hidden_layers : int;
  num_attention_heads : int;
  num_key_value_heads : int;
  head_dim : int;
  max_position_embeddings : int;
  sliding_window : int;
  rms_norm_eps : float;
  rope_theta : float;
  attention_dropout : float;
  attention_logit_softcapping : float;
  final_logit_softcapping : float;
  query_pre_attn_scalar : float;
  initializer_range : float;
  tie_word_embeddings : bool;
}

let config = Config.make
let position_ids_key = Core.Rope.position_ids_key
let attention_mask_key = Core.Mask.attention_mask_key
let weight_init cfg = Init.normal ~stddev:cfg.initializer_range ()

let token_embedding cfg =
  Core.Embedding.token ~vocab_size:cfg.vocab_size ~hidden_size:cfg.hidden_size
    ~scale:true ~weight_init:(weight_init cfg) ()

let norm cfg = Core.Norm.gemma_rms ~dim:cfg.hidden_size ~eps:cfg.rms_norm_eps ()

type profile = {
  window : int -> int option;
  rope_theta : int -> float;
  qk_norm_eps : float option;
  attention_logit_softcapping : float option;
  final_logit_softcapping : float option;
}

let default_profile cfg =
  {
    window =
      (fun layer_index ->
        if layer_index mod 2 = 0 then Some cfg.sliding_window else None);
    rope_theta = (fun _ -> cfg.rope_theta);
    qk_norm_eps = None;
    attention_logit_softcapping = Some cfg.attention_logit_softcapping;
    final_logit_softcapping = Some cfg.final_logit_softcapping;
  }

let self_attention cfg profile layer_index =
  Core.Dense_attention.self_attention ~hidden_size:cfg.hidden_size
    ~num_heads:cfg.num_attention_heads ~num_kv_heads:cfg.num_key_value_heads
    ~head_dim:cfg.head_dim
    ~rope_theta:(profile.rope_theta layer_index)
    ?window:(profile.window layer_index)
    ~score_scale:(1.0 /. Stdlib.sqrt cfg.query_pre_attn_scalar)
    ?logit_softcap:profile.attention_logit_softcapping
    ?qk_norm_eps:profile.qk_norm_eps ~dropout:cfg.attention_dropout
    ~weight_init:(weight_init cfg) ()

let mlp cfg =
  Core.Ffn.gated ~hidden_size:cfg.hidden_size
    ~intermediate_size:cfg.intermediate_size ~activation:Core.Ffn.Gelu_approx
    ~weight_init:(weight_init cfg) ()

let decoder_block_with_profile cfg profile ~layer_index () =
  if layer_index < 0 || layer_index >= cfg.num_hidden_layers then
    invalid_argf "Gemma2.decoder_block: layer index %d is out of bounds"
      layer_index;
  let input_layernorm = norm cfg in
  let self_attn = self_attention cfg profile layer_index in
  let post_attention_layernorm = norm cfg in
  let pre_feedforward_layernorm = norm cfg in
  let mlp = mlp cfg in
  let post_feedforward_layernorm = norm cfg in
  let names =
    [
      "input_layernorm";
      "self_attn";
      "post_attention_layernorm";
      "pre_feedforward_layernorm";
      "mlp";
      "post_feedforward_layernorm";
    ]
  in
  {
    Layer.init =
      (fun ~dtype ->
        Core.Layer_util.init_children dtype
          [
            ("input_layernorm", input_layernorm.Layer.init);
            ("self_attn", self_attn.Layer.init);
            ("post_attention_layernorm", post_attention_layernorm.Layer.init);
            ("pre_feedforward_layernorm", pre_feedforward_layernorm.Layer.init);
            ("mlp", mlp.Layer.init);
            ("post_feedforward_layernorm", post_feedforward_layernorm.Layer.init);
          ]);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        let op_ctx = "Gemma2.decoder_block" in
        let x = Core.Layer_util.require_same_float_dtype ~ctx:op_ctx dtype x in
        let apply layer name input =
          Core.Layer_util.apply_child ~ctx:op_ctx layer ~name ~params ~state
            ~dtype ~training ?call_ctx:ctx input
        in
        let normalized, input_norm_state =
          apply input_layernorm "input_layernorm" x
        in
        let attended, attention_state =
          apply self_attn "self_attn" normalized
        in
        let attended, post_attention_state =
          apply post_attention_layernorm "post_attention_layernorm" attended
        in
        let x = Nx.add x attended in
        let normalized, pre_feedforward_state =
          apply pre_feedforward_layernorm "pre_feedforward_layernorm" x
        in
        let transformed, mlp_state = apply mlp "mlp" normalized in
        let transformed, post_feedforward_state =
          apply post_feedforward_layernorm "post_feedforward_layernorm"
            transformed
        in
        let state =
          Core.Layer_util.merge_state ~names
            [
              input_norm_state;
              attention_state;
              post_attention_state;
              pre_feedforward_state;
              mlp_state;
              post_feedforward_state;
            ]
        in
        (Nx.add x transformed, state));
  }

let decoder_block cfg ~layer_index () =
  decoder_block_with_profile cfg (default_profile cfg) ~layer_index ()

let lm_head cfg =
  Core.Projection.linear ~in_features:cfg.hidden_size
    ~out_features:cfg.vocab_size ~weight_init:(weight_init cfg) ()

let init_model_vars ~profile ~with_lm_head cfg dtype =
  let embeddings_vars =
    Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
        (token_embedding cfg).Layer.init ~dtype)
  in
  let layer_vars =
    List.init cfg.num_hidden_layers (fun layer_index ->
        Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
            (decoder_block_with_profile cfg profile ~layer_index ()).Layer.init
              ~dtype))
  in
  let norm_vars =
    Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
        (norm cfg).Layer.init ~dtype)
  in
  let params =
    [
      ("embed_tokens", Layer.params embeddings_vars);
      ("layers", Ptree.list (List.map Layer.params layer_vars));
      ("norm", Layer.params norm_vars);
    ]
  in
  let state =
    [
      ("embed_tokens", Layer.state embeddings_vars);
      ("layers", Ptree.list (List.map Layer.state layer_vars));
      ("norm", Layer.state norm_vars);
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

let decode (type layout input_layout) ~profile ~(cfg : config) ~params ~state
    ~(dtype : (float, layout) Nx.dtype) ~training ?ctx
    (input_ids : (int32, input_layout) Nx.t) =
  let params_root = Core.Layer_util.fields ~ctx:"Gemma2.decode.params" params in
  let state_root = Core.Layer_util.fields ~ctx:"Gemma2.decode.state" state in
  let param name =
    Core.Layer_util.find ~ctx:"Gemma2.decode.params" name params_root
  in
  let child_state name =
    Core.Layer_util.find ~ctx:"Gemma2.decode.state" name state_root
  in
  let hidden, embeddings_state =
    (token_embedding cfg).Layer.apply ~params:(param "embed_tokens")
      ~state:(child_state "embed_tokens")
      ~dtype ~training ?ctx input_ids
  in
  let layer_params =
    Ptree.List.items_exn ~ctx:"Gemma2.decode.params.layers" (param "layers")
    |> Array.of_list
  in
  let layer_states =
    Ptree.List.items_exn ~ctx:"Gemma2.decode.state.layers"
      (child_state "layers")
    |> Array.of_list
  in
  if
    Array.length layer_params <> cfg.num_hidden_layers
    || Array.length layer_states <> cfg.num_hidden_layers
  then invalid_argf "Gemma2.decode: layer parameter/state count mismatch";
  let output_states = Array.make cfg.num_hidden_layers Ptree.empty in
  let rec apply_layers layer_index hidden =
    if layer_index = cfg.num_hidden_layers then hidden
    else
      let block = decoder_block_with_profile cfg profile ~layer_index () in
      let hidden, output_state =
        block.Layer.apply ~params:layer_params.(layer_index)
          ~state:layer_states.(layer_index) ~dtype ~training ?ctx hidden
      in
      output_states.(layer_index) <- output_state;
      apply_layers (layer_index + 1) hidden
  in
  let hidden = apply_layers 0 hidden in
  let hidden, norm_state =
    (norm cfg).Layer.apply ~params:(param "norm") ~state:(child_state "norm")
      ~dtype ~training ?ctx hidden
  in
  ( hidden,
    Ptree.dict
      [
        ("embed_tokens", embeddings_state);
        ("layers", Ptree.list (Array.to_list output_states));
        ("norm", norm_state);
      ] )

let decoder_with_profile cfg profile () =
  {
    Layer.init =
      (fun ~dtype -> init_model_vars ~profile ~with_lm_head:false cfg dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx input_ids ->
        decode ~profile ~cfg ~params ~state ~dtype ~training ?ctx input_ids);
  }

let decoder cfg () = decoder_with_profile cfg (default_profile cfg) ()

let softcap cap logits =
  match cap with
  | None -> logits
  | Some value ->
      let cap = Nx.scalar (Nx.dtype logits) value in
      Nx.mul cap (Nx.tanh (Nx.div logits cap))

let for_causal_lm_with_profile cfg profile () =
  let use_head = not cfg.tie_word_embeddings in
  {
    Layer.init =
      (fun ~dtype -> init_model_vars ~profile ~with_lm_head:use_head cfg dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx input_ids ->
        let original_state = state in
        let hidden, decoder_state =
          decode ~profile ~cfg ~params ~state ~dtype ~training ?ctx input_ids
        in
        let params_root =
          Core.Layer_util.fields ~ctx:"Gemma2.for_causal_lm.params" params
        in
        let logits, state =
          if cfg.tie_word_embeddings then
            let embeddings =
              Core.Layer_util.find ~ctx:"Gemma2.for_causal_lm.params"
                "embed_tokens" params_root
              |> Core.Layer_util.fields
                   ~ctx:"Gemma2.for_causal_lm.params.embed_tokens"
            in
            let weight = Core.Layer_util.get embeddings ~name:"weight" dtype in
            ( Nx.matmul hidden (Nx.transpose weight ~axes:[ 1; 0 ]),
              decoder_state )
          else
            let state_root =
              Core.Layer_util.fields ~ctx:"Gemma2.for_causal_lm.state"
                original_state
            in
            let logits, head_state =
              (lm_head cfg).Layer.apply
                ~params:
                  (Core.Layer_util.find ~ctx:"Gemma2.for_causal_lm.params"
                     "lm_head" params_root)
                ~state:
                  (Core.Layer_util.find ~ctx:"Gemma2.for_causal_lm.state"
                     "lm_head" state_root)
                ~dtype ~training ?ctx hidden
            in
            let decoder_fields =
              Core.Layer_util.fields ~ctx:"Gemma2.for_causal_lm.decoder_state"
                decoder_state
            in
            (logits, Ptree.dict (decoder_fields @ [ ("lm_head", head_state) ]))
        in
        (softcap profile.final_logit_softcapping logits, state));
  }

let for_causal_lm cfg () =
  for_causal_lm_with_profile cfg (default_profile cfg) ()

module Cache = struct
  type 'layout t = 'layout Dense_cache.t

  let create cfg ~batch_size ~max_length ~dtype =
    Dense_cache.create ~num_layers:cfg.num_hidden_layers
      ~num_kv_heads:cfg.num_key_value_heads ~head_dim:cfg.head_dim
      ~max_position_embeddings:cfg.max_position_embeddings ~batch_size
      ~max_length ~dtype

  let batch_size = Dense_cache.batch_size
  let max_length = Dense_cache.max_length
  let length = Dense_cache.length
end

let cached_decoder_block cfg profile ~layer_index ~params ~state ~dtype
    ~position ~valid ~key_cache ~value_cache x =
  let op_ctx = "Gemma2.cached_decoder_block" in
  let input_layernorm = norm cfg in
  let post_attention_layernorm = norm cfg in
  let pre_feedforward_layernorm = norm cfg in
  let mlp = mlp cfg in
  let post_feedforward_layernorm = norm cfg in
  let apply layer name input =
    Core.Layer_util.apply_child ~ctx:op_ctx layer ~name ~params ~state ~dtype
      ~training:false input
  in
  let normalized, input_norm_state =
    apply input_layernorm "input_layernorm" x
  in
  let attention_params, attention_input_state =
    Core.Layer_util.child_vars ~ctx:op_ctx ~params ~state "self_attn"
  in
  let attended, attention_state, key_cache, value_cache =
    Core.Dense_attention.cached_self_attention ~hidden_size:cfg.hidden_size
      ~num_heads:cfg.num_attention_heads ~num_kv_heads:cfg.num_key_value_heads
      ~head_dim:cfg.head_dim
      ~rope_theta:(profile.rope_theta layer_index)
      ?window:(profile.window layer_index)
      ~score_scale:(1.0 /. Stdlib.sqrt cfg.query_pre_attn_scalar)
      ?logit_softcap:profile.attention_logit_softcapping
      ?qk_norm_eps:profile.qk_norm_eps ~dropout:cfg.attention_dropout
      ~params:attention_params ~state:attention_input_state ~dtype
      ~training:false ~position ~valid ~key_cache ~value_cache normalized
  in
  let attended, post_attention_state =
    apply post_attention_layernorm "post_attention_layernorm" attended
  in
  let x = Nx.add x attended in
  let normalized, pre_feedforward_state =
    apply pre_feedforward_layernorm "pre_feedforward_layernorm" x
  in
  let transformed, mlp_state = apply mlp "mlp" normalized in
  let transformed, post_feedforward_state =
    apply post_feedforward_layernorm "post_feedforward_layernorm" transformed
  in
  let state =
    Core.Layer_util.merge_state
      ~names:
        [
          "input_layernorm";
          "self_attn";
          "post_attention_layernorm";
          "pre_feedforward_layernorm";
          "mlp";
          "post_feedforward_layernorm";
        ]
      [
        input_norm_state;
        attention_state;
        post_attention_state;
        pre_feedforward_state;
        mlp_state;
        post_feedforward_state;
      ]
  in
  (Nx.add x transformed, state, key_cache, value_cache)

let cached_causal_lm ~profile ~cfg ~params ~state ~dtype ?attention_mask cache
    input_ids =
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 then
    invalid_argf "Gemma2.cached_causal_lm: input IDs must have rank 2";
  let batch = shape.(0) in
  let seq = shape.(1) in
  if batch <> cache.Dense_cache.batch_size then
    invalid_argf "Gemma2.cached_causal_lm: batch size mismatch";
  if seq <= 0 || cache.length + seq > cache.max_length then
    invalid_argf "Gemma2.cached_causal_lm: invalid sequence length or capacity";
  let token_valid =
    match attention_mask with
    | None -> Nx.ones Nx.bool [| batch; seq |]
    | Some mask ->
        if Nx.shape mask <> [| batch; seq |] then
          invalid_argf "Gemma2.cached_causal_lm: attention mask shape mismatch";
        mask
  in
  let valid = Dense_cache.append_valid cache token_valid seq in
  let params_root =
    Core.Layer_util.fields ~ctx:"Gemma2.cached_causal_lm.params" params
  in
  let state_root =
    Core.Layer_util.fields ~ctx:"Gemma2.cached_causal_lm.state" state
  in
  let param name =
    Core.Layer_util.find ~ctx:"Gemma2.cached_causal_lm.params" name params_root
  in
  let child_state name =
    Core.Layer_util.find ~ctx:"Gemma2.cached_causal_lm.state" name state_root
  in
  let hidden, embeddings_state =
    (token_embedding cfg).Layer.apply ~params:(param "embed_tokens")
      ~state:(child_state "embed_tokens")
      ~dtype ~training:false input_ids
  in
  let layer_params =
    Ptree.List.items_exn ~ctx:"Gemma2.cached_causal_lm.params.layers"
      (param "layers")
    |> Array.of_list
  in
  let layer_states =
    Ptree.List.items_exn ~ctx:"Gemma2.cached_causal_lm.state.layers"
      (child_state "layers")
    |> Array.of_list
  in
  let keys = Array.copy cache.keys in
  let values = Array.copy cache.values in
  let output_states = Array.make cfg.num_hidden_layers Ptree.empty in
  let rec apply_layers layer_index hidden =
    if layer_index = cfg.num_hidden_layers then hidden
    else
      let hidden, output_state, key, value =
        cached_decoder_block cfg profile ~layer_index
          ~params:layer_params.(layer_index) ~state:layer_states.(layer_index)
          ~dtype ~position:cache.position ~valid ~key_cache:keys.(layer_index)
          ~value_cache:values.(layer_index) hidden
      in
      keys.(layer_index) <- key;
      values.(layer_index) <- value;
      output_states.(layer_index) <- output_state;
      apply_layers (layer_index + 1) hidden
  in
  let hidden = apply_layers 0 hidden in
  let hidden, norm_state =
    (norm cfg).Layer.apply ~params:(param "norm") ~state:(child_state "norm")
      ~dtype ~training:false hidden
  in
  let decoder_state =
    [
      ("embed_tokens", embeddings_state);
      ("layers", Ptree.list (Array.to_list output_states));
      ("norm", norm_state);
    ]
  in
  let logits, output_state =
    if cfg.tie_word_embeddings then
      let embeddings =
        Core.Layer_util.fields ~ctx:"Gemma2.cached_causal_lm.embed_tokens"
          (param "embed_tokens")
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
  (softcap profile.final_logit_softcapping logits, output_state, cache)

let prefill_with_profile cfg profile vars cache ?attention_mask input_ids =
  let logits, _, cache =
    cached_causal_lm ~profile ~cfg ~params:(Layer.params vars)
      ~state:(Layer.state vars) ~dtype:(Layer.dtype vars) ?attention_mask cache
      input_ids
  in
  (logits, cache)

let prefill cfg vars cache ?attention_mask input_ids =
  prefill_with_profile cfg (default_profile cfg) vars cache ?attention_mask
    input_ids

let decode_step_with_profile cfg profile vars cache input_ids =
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 || shape.(1) <> 1 then
    invalid_argf "Gemma2.decode_step: input IDs must have shape [batch; 1]";
  prefill_with_profile cfg profile vars cache input_ids

let decode_step cfg vars cache input_ids =
  decode_step_with_profile cfg (default_profile cfg) vars cache input_ids

module Pjrt = struct
  type 'layout t = 'layout Dense_pjrt.t

  let compile_with_profile ?(device_id = 0) cfg profile vars =
    let dtype = Layer.dtype vars in
    let params = Layer.params vars in
    let state = Layer.state vars in
    Dense_pjrt.compile ~device_id ~layer_count:cfg.num_hidden_layers ~dtype
      (fun ~attention_mask cache input_ids ->
        let logits, _, cache =
          cached_causal_lm ~profile ~cfg ~params ~state ~dtype ~attention_mask
            cache input_ids
        in
        (logits, cache))

  let compile ?device_id cfg vars =
    compile_with_profile ?device_id cfg (default_profile cfg) vars

  let prefill = Dense_pjrt.prefill
  let decode_step = Dense_pjrt.decode_step
end

let map_hf_weights ?(qk_norm = false) ?(model_prefix = "model")
    ?(allow_unconsumed = fun _ -> false) ~cfg ~dtype tensors =
  let module Hf = Core.Hf in
  let weights = Hf.weights tensors in
  let vector name size =
    Hf.tensor weights ~name ~shape:[| size |] |> Hf.cast dtype
  in
  let matrix name ~rows ~cols = Hf.matrix weights dtype ~name ~rows ~cols in
  let projection name ~in_features ~out_features =
    Ptree.dict [ ("weight", matrix name ~rows:out_features ~cols:in_features) ]
  in
  let layer layer_index =
    let prefix = Printf.sprintf "%s.layers.%d" model_prefix layer_index in
    let norm name =
      Ptree.dict
        [ ("weight", vector (prefix ^ "." ^ name ^ ".weight") cfg.hidden_size) ]
    in
    Ptree.dict
      [
        ("input_layernorm", norm "input_layernorm");
        ( "self_attn",
          Ptree.dict
            ([
               ( "q_proj",
                 projection
                   (prefix ^ ".self_attn.q_proj.weight")
                   ~in_features:cfg.hidden_size
                   ~out_features:(cfg.num_attention_heads * cfg.head_dim) );
               ( "k_proj",
                 projection
                   (prefix ^ ".self_attn.k_proj.weight")
                   ~in_features:cfg.hidden_size
                   ~out_features:(cfg.num_key_value_heads * cfg.head_dim) );
               ( "v_proj",
                 projection
                   (prefix ^ ".self_attn.v_proj.weight")
                   ~in_features:cfg.hidden_size
                   ~out_features:(cfg.num_key_value_heads * cfg.head_dim) );
               ( "o_proj",
                 projection
                   (prefix ^ ".self_attn.o_proj.weight")
                   ~in_features:(cfg.num_attention_heads * cfg.head_dim)
                   ~out_features:cfg.hidden_size );
             ]
            @
            if qk_norm then
              [
                ( "q_norm",
                  Ptree.dict
                    [
                      ( "weight",
                        vector
                          (prefix ^ ".self_attn.q_norm.weight")
                          cfg.head_dim );
                    ] );
                ( "k_norm",
                  Ptree.dict
                    [
                      ( "weight",
                        vector
                          (prefix ^ ".self_attn.k_norm.weight")
                          cfg.head_dim );
                    ] );
              ]
            else []) );
        ("post_attention_layernorm", norm "post_attention_layernorm");
        ("pre_feedforward_layernorm", norm "pre_feedforward_layernorm");
        ( "mlp",
          Ptree.dict
            [
              ( "gate_proj",
                projection
                  (prefix ^ ".mlp.gate_proj.weight")
                  ~in_features:cfg.hidden_size
                  ~out_features:cfg.intermediate_size );
              ( "up_proj",
                projection
                  (prefix ^ ".mlp.up_proj.weight")
                  ~in_features:cfg.hidden_size
                  ~out_features:cfg.intermediate_size );
              ( "down_proj",
                projection
                  (prefix ^ ".mlp.down_proj.weight")
                  ~in_features:cfg.intermediate_size
                  ~out_features:cfg.hidden_size );
            ] );
        ("post_feedforward_layernorm", norm "post_feedforward_layernorm");
      ]
  in
  let params =
    [
      ( "embed_tokens",
        Ptree.dict
          [
            ( "weight",
              Hf.tensor weights
                ~name:(model_prefix ^ ".embed_tokens.weight")
                ~shape:[| cfg.vocab_size; cfg.hidden_size |]
              |> Hf.cast dtype );
          ] );
      ("layers", Ptree.list (List.init cfg.num_hidden_layers layer));
      ( "norm",
        Ptree.dict
          [ ("weight", vector (model_prefix ^ ".norm.weight") cfg.hidden_size) ]
      );
    ]
  in
  let params =
    if cfg.tie_word_embeddings then params
    else
      params
      @ [
          ( "lm_head",
            projection "lm_head.weight" ~in_features:cfg.hidden_size
              ~out_features:cfg.vocab_size );
        ]
  in
  Hf.ensure_consumed weights ~allow:(fun name ->
      (cfg.tie_word_embeddings && name = "lm_head.weight")
      || String.ends_with ~suffix:".rotary_emb.inv_freq" name
      || allow_unconsumed name);
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

module Internal = struct
  type nonrec profile = profile

  let profile ~window ~rope_theta ~qk_norm_eps ~attention_logit_softcapping
      ~final_logit_softcapping =
    {
      window;
      rope_theta;
      qk_norm_eps;
      attention_logit_softcapping;
      final_logit_softcapping;
    }

  let decoder_block_with_profile = decoder_block_with_profile
  let decoder_with_profile = decoder_with_profile
  let for_causal_lm_with_profile = for_causal_lm_with_profile
  let prefill_with_profile = prefill_with_profile
  let decode_step_with_profile = decode_step_with_profile
  let pjrt_compile_with_profile = Pjrt.compile_with_profile
  let map_hf_weights = map_hf_weights
end
