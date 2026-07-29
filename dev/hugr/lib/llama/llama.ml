(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun
module Core = Hugr_core

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type config = Config.t = {
  vocab_size : int;
  hidden_size : int;
  intermediate_size : int;
  num_hidden_layers : int;
  num_attention_heads : int;
  num_key_value_heads : int;
  max_position_embeddings : int;
  rms_norm_eps : float;
  rope_theta : float;
  attention_dropout : float;
  initializer_range : float;
  tie_word_embeddings : bool;
}

let config = Config.make
let position_ids_key = Core.Rope.position_ids_key
let attention_mask_key = Core.Mask.attention_mask_key
let weight_init cfg = Init.normal ~stddev:cfg.initializer_range ()

module Cache = struct
  type 'layout t = 'layout Dense_cache.t

  let create cfg ~batch_size ~max_length ~dtype =
    Dense_cache.create ~num_layers:cfg.num_hidden_layers
      ~num_kv_heads:cfg.num_key_value_heads
      ~head_dim:(cfg.hidden_size / cfg.num_attention_heads)
      ~max_position_embeddings:cfg.max_position_embeddings ~batch_size
      ~max_length ~dtype

  let batch_size = Dense_cache.batch_size
  let max_length = Dense_cache.max_length
  let length = Dense_cache.length
end

let token_embedding cfg =
  Core.Embedding.token ~vocab_size:cfg.vocab_size ~hidden_size:cfg.hidden_size
    ~weight_init:(weight_init cfg) ()

let final_norm cfg =
  Layer.rms_norm ~dim:cfg.hidden_size ~eps:cfg.rms_norm_eps ()

let decoder_block_with_window ?window cfg () =
  let input_layernorm =
    Layer.rms_norm ~dim:cfg.hidden_size ~eps:cfg.rms_norm_eps ()
  in
  let self_attn =
    Core.Dense_attention.self_attention ~hidden_size:cfg.hidden_size
      ~num_heads:cfg.num_attention_heads ~num_kv_heads:cfg.num_key_value_heads
      ~head_dim:(cfg.hidden_size / cfg.num_attention_heads)
      ~rope_theta:cfg.rope_theta ?window ~dropout:cfg.attention_dropout
      ~weight_init:(weight_init cfg) ()
  in
  let post_attention_layernorm =
    Layer.rms_norm ~dim:cfg.hidden_size ~eps:cfg.rms_norm_eps ()
  in
  let mlp =
    Core.Ffn.gated ~hidden_size:cfg.hidden_size
      ~intermediate_size:cfg.intermediate_size ~activation:Core.Ffn.Silu
      ~weight_init:(weight_init cfg) ()
  in
  let names =
    [ "input_layernorm"; "self_attn"; "post_attention_layernorm"; "mlp" ]
  in
  {
    Layer.init =
      (fun ~dtype ->
        Core.Layer_util.init_children dtype
          [
            ("input_layernorm", input_layernorm.Layer.init);
            ("self_attn", self_attn.Layer.init);
            ("post_attention_layernorm", post_attention_layernorm.Layer.init);
            ("mlp", mlp.Layer.init);
          ]);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        let x =
          Core.Layer_util.require_same_float_dtype ~ctx:"Llama.decoder_block"
            dtype x
        in
        let normalized, input_norm_state =
          Core.Layer_util.apply_child ~ctx:"Llama.decoder_block" input_layernorm
            ~name:"input_layernorm" ~params ~state ~dtype ~training
            ?call_ctx:ctx x
        in
        let attended, attention_state =
          Core.Layer_util.apply_child ~ctx:"Llama.decoder_block" self_attn
            ~name:"self_attn" ~params ~state ~dtype ~training ?call_ctx:ctx
            normalized
        in
        let x = Nx.add x attended in
        let normalized, post_norm_state =
          Core.Layer_util.apply_child ~ctx:"Llama.decoder_block"
            post_attention_layernorm ~name:"post_attention_layernorm" ~params
            ~state ~dtype ~training ?call_ctx:ctx x
        in
        let transformed, mlp_state =
          Core.Layer_util.apply_child ~ctx:"Llama.decoder_block" mlp ~name:"mlp"
            ~params ~state ~dtype ~training ?call_ctx:ctx normalized
        in
        let state =
          Core.Layer_util.merge_state ~names
            [ input_norm_state; attention_state; post_norm_state; mlp_state ]
        in
        (Nx.add x transformed, state));
  }

let decoder_block cfg () = decoder_block_with_window cfg ()

let lm_head cfg =
  Core.Projection.linear ~in_features:cfg.hidden_size
    ~out_features:cfg.vocab_size ~weight_init:(weight_init cfg) ()

let cached_decoder_block ?window cfg ~params ~state ~dtype ~position ~valid
    ~key_cache ~value_cache x =
  let ctx = "Llama.cached_decoder_block" in
  let input_layernorm =
    Layer.rms_norm ~dim:cfg.hidden_size ~eps:cfg.rms_norm_eps ()
  in
  let post_attention_layernorm =
    Layer.rms_norm ~dim:cfg.hidden_size ~eps:cfg.rms_norm_eps ()
  in
  let mlp =
    Core.Ffn.gated ~hidden_size:cfg.hidden_size
      ~intermediate_size:cfg.intermediate_size ~activation:Core.Ffn.Silu ()
  in
  let normalized, input_norm_state =
    Core.Layer_util.apply_child ~ctx input_layernorm ~name:"input_layernorm"
      ~params ~state ~dtype ~training:false x
  in
  let attention_params, attention_input_state =
    Core.Layer_util.child_vars ~ctx ~params ~state "self_attn"
  in
  let attended, attention_state, key_cache, value_cache =
    Core.Dense_attention.cached_self_attention ~hidden_size:cfg.hidden_size
      ~num_heads:cfg.num_attention_heads ~num_kv_heads:cfg.num_key_value_heads
      ~head_dim:(cfg.hidden_size / cfg.num_attention_heads)
      ~rope_theta:cfg.rope_theta ?window ~dropout:cfg.attention_dropout
      ~params:attention_params ~state:attention_input_state ~dtype
      ~training:false ~position ~valid ~key_cache ~value_cache normalized
  in
  let x = Nx.add x attended in
  let normalized, post_norm_state =
    Core.Layer_util.apply_child ~ctx post_attention_layernorm
      ~name:"post_attention_layernorm" ~params ~state ~dtype ~training:false x
  in
  let transformed, mlp_state =
    Core.Layer_util.apply_child ~ctx mlp ~name:"mlp" ~params ~state ~dtype
      ~training:false normalized
  in
  let state =
    Core.Layer_util.merge_state
      ~names:
        [ "input_layernorm"; "self_attn"; "post_attention_layernorm"; "mlp" ]
      [ input_norm_state; attention_state; post_norm_state; mlp_state ]
  in
  (Nx.add x transformed, state, key_cache, value_cache)

let init_model_vars ?window ~with_lm_head cfg dtype =
  let embeddings_vars =
    Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
        (token_embedding cfg).Layer.init ~dtype)
  in
  let block = decoder_block_with_window ?window cfg () in
  let layer_vars =
    List.init cfg.num_hidden_layers (fun _ ->
        Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () -> block.Layer.init ~dtype))
  in
  let norm_vars =
    Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
        (final_norm cfg).Layer.init ~dtype)
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

let decode (type layout input_layout) ?window ~(cfg : config) ~params ~state
    ~(dtype : (float, layout) Nx.dtype) ~training ?ctx
    (input_ids : (int32, input_layout) Nx.t) =
  let params_root = Core.Layer_util.fields ~ctx:"Llama.decode.params" params in
  let state_root = Core.Layer_util.fields ~ctx:"Llama.decode.state" state in
  let param name =
    Core.Layer_util.find ~ctx:"Llama.decode.params" name params_root
  in
  let child_state name =
    Core.Layer_util.find ~ctx:"Llama.decode.state" name state_root
  in
  let embeddings = token_embedding cfg in
  let hidden, embeddings_state =
    embeddings.Layer.apply ~params:(param "embed_tokens")
      ~state:(child_state "embed_tokens")
      ~dtype ~training ?ctx input_ids
  in
  let block = decoder_block_with_window ?window cfg () in
  let layer_params =
    Ptree.List.items_exn ~ctx:"Llama.decode.params.layers" (param "layers")
  in
  let layer_states =
    Ptree.List.items_exn ~ctx:"Llama.decode.state.layers" (child_state "layers")
  in
  if List.length layer_params <> cfg.num_hidden_layers then
    invalid_argf "Llama.decode: expected %d parameter sets, got %d"
      cfg.num_hidden_layers (List.length layer_params);
  if List.length layer_states <> cfg.num_hidden_layers then
    invalid_argf "Llama.decode: expected %d layer states, got %d"
      cfg.num_hidden_layers (List.length layer_states);
  let hidden, layer_states =
    List.fold_left2
      (fun (hidden, states) params state ->
        let hidden, state =
          block.Layer.apply ~params ~state ~dtype ~training ?ctx hidden
        in
        (hidden, state :: states))
      (hidden, []) layer_params layer_states
  in
  let norm = final_norm cfg in
  let hidden, norm_state =
    norm.Layer.apply ~params:(param "norm") ~state:(child_state "norm") ~dtype
      ~training ?ctx hidden
  in
  let state =
    Ptree.dict
      [
        ("embed_tokens", embeddings_state);
        ("layers", Ptree.list (List.rev layer_states));
        ("norm", norm_state);
      ]
  in
  (hidden, state)

let decoder_with_window ?window cfg () =
  {
    Layer.init =
      (fun ~dtype -> init_model_vars ?window ~with_lm_head:false cfg dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx input_ids ->
        decode ?window ~cfg ~params ~state ~dtype ~training ?ctx input_ids);
  }

let decoder cfg () = decoder_with_window cfg ()

let for_causal_lm_with_window ?window cfg () =
  let use_head = not cfg.tie_word_embeddings in
  {
    Layer.init =
      (fun ~dtype -> init_model_vars ?window ~with_lm_head:use_head cfg dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx input_ids ->
        let original_state = state in
        let hidden, decoder_state =
          decode ?window ~cfg ~params ~state ~dtype ~training ?ctx input_ids
        in
        if cfg.tie_word_embeddings then
          let root =
            Core.Layer_util.fields ~ctx:"Llama.for_causal_lm.params" params
          in
          let embeddings =
            Core.Layer_util.find ~ctx:"Llama.for_causal_lm.params"
              "embed_tokens" root
            |> Core.Layer_util.fields
                 ~ctx:"Llama.for_causal_lm.params.embed_tokens"
          in
          let weight = Core.Layer_util.get embeddings ~name:"weight" dtype in
          (Nx.matmul hidden (Nx.transpose weight ~axes:[ 1; 0 ]), decoder_state)
        else
          let param_root =
            Core.Layer_util.fields ~ctx:"Llama.for_causal_lm.params" params
          in
          let state_root =
            Core.Layer_util.fields ~ctx:"Llama.for_causal_lm.state"
              original_state
          in
          let head = lm_head cfg in
          let logits, head_state =
            head.Layer.apply
              ~params:
                (Core.Layer_util.find ~ctx:"Llama.for_causal_lm.params"
                   "lm_head" param_root)
              ~state:
                (Core.Layer_util.find ~ctx:"Llama.for_causal_lm.state" "lm_head"
                   state_root)
              ~dtype ~training ?ctx hidden
          in
          let decoder_fields =
            Core.Layer_util.fields ~ctx:"Llama.for_causal_lm.decoder_state"
              decoder_state
          in
          (logits, Ptree.dict (decoder_fields @ [ ("lm_head", head_state) ])));
  }

let for_causal_lm cfg () = for_causal_lm_with_window cfg ()

let cached_causal_lm ?window ~cfg ~params ~state ~dtype ?attention_mask cache
    input_ids =
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 then
    invalid_argf
      "Llama.cached_causal_lm: expected input IDs with shape [batch; seq], got \
       [%s]"
      (String.concat "; " (List.map string_of_int (Array.to_list shape)));
  let batch = shape.(0) in
  let seq = shape.(1) in
  if batch <> cache.Dense_cache.batch_size then
    invalid_argf "Llama.cached_causal_lm: expected batch size %d, got %d"
      cache.batch_size batch;
  if seq <= 0 then
    invalid_argf "Llama.cached_causal_lm: sequence length must be positive";
  if cache.length + seq > cache.max_length then
    invalid_argf
      "Llama.cached_causal_lm: cache capacity %d exceeded by position %d + %d"
      cache.max_length cache.length seq;
  if Array.length cache.keys <> cfg.num_hidden_layers then
    invalid_argf "Llama.cached_causal_lm: expected %d key caches, got %d"
      cfg.num_hidden_layers (Array.length cache.keys);
  if Array.length cache.values <> cfg.num_hidden_layers then
    invalid_argf "Llama.cached_causal_lm: expected %d value caches, got %d"
      cfg.num_hidden_layers
      (Array.length cache.values);
  let token_valid =
    match attention_mask with
    | None -> Nx.ones Nx.bool [| batch; seq |]
    | Some mask ->
        if Nx.shape mask <> [| batch; seq |] then
          invalid_argf
            "Llama.cached_causal_lm: attention mask must have shape [%d; %d]"
            batch seq;
        mask
  in
  let valid = Dense_cache.append_valid cache token_valid seq in
  let params_root =
    Core.Layer_util.fields ~ctx:"Llama.cached_causal_lm.params" params
  in
  let state_root =
    Core.Layer_util.fields ~ctx:"Llama.cached_causal_lm.state" state
  in
  let param name =
    Core.Layer_util.find ~ctx:"Llama.cached_causal_lm.params" name params_root
  in
  let child_state name =
    Core.Layer_util.find ~ctx:"Llama.cached_causal_lm.state" name state_root
  in
  let embeddings = token_embedding cfg in
  let hidden, embeddings_state =
    embeddings.Layer.apply ~params:(param "embed_tokens")
      ~state:(child_state "embed_tokens")
      ~dtype ~training:false input_ids
  in
  let layer_params =
    Ptree.List.items_exn ~ctx:"Llama.cached_causal_lm.params.layers"
      (param "layers")
    |> Array.of_list
  in
  let layer_states =
    Ptree.List.items_exn ~ctx:"Llama.cached_causal_lm.state.layers"
      (child_state "layers")
    |> Array.of_list
  in
  if Array.length layer_params <> cfg.num_hidden_layers then
    invalid_argf "Llama.cached_causal_lm: expected %d parameter sets, got %d"
      cfg.num_hidden_layers
      (Array.length layer_params);
  if Array.length layer_states <> cfg.num_hidden_layers then
    invalid_argf "Llama.cached_causal_lm: expected %d layer states, got %d"
      cfg.num_hidden_layers
      (Array.length layer_states);
  let keys = Array.copy cache.keys in
  let values = Array.copy cache.values in
  let output_states = Array.make cfg.num_hidden_layers Ptree.empty in
  let rec apply_layers index hidden =
    if index = cfg.num_hidden_layers then hidden
    else
      let hidden, layer_state, key_cache, value_cache =
        cached_decoder_block ?window cfg ~params:layer_params.(index)
          ~state:layer_states.(index) ~dtype ~position:cache.position ~valid
          ~key_cache:keys.(index) ~value_cache:values.(index) hidden
      in
      keys.(index) <- key_cache;
      values.(index) <- value_cache;
      output_states.(index) <- layer_state;
      apply_layers (index + 1) hidden
  in
  let hidden = apply_layers 0 hidden in
  let norm = final_norm cfg in
  let hidden, norm_state =
    norm.Layer.apply ~params:(param "norm") ~state:(child_state "norm") ~dtype
      ~training:false hidden
  in
  let decoder_state =
    [
      ("embed_tokens", embeddings_state);
      ("layers", Ptree.list (Array.to_list output_states));
      ("norm", norm_state);
    ]
  in
  let logits, state =
    if cfg.tie_word_embeddings then
      let embeddings =
        Core.Layer_util.fields ~ctx:"Llama.cached_causal_lm.embed_tokens"
          (param "embed_tokens")
      in
      let weight = Core.Layer_util.get embeddings ~name:"weight" dtype in
      ( Nx.matmul hidden (Nx.transpose weight ~axes:[ 1; 0 ]),
        Ptree.dict decoder_state )
    else
      let head = lm_head cfg in
      let logits, head_state =
        head.Layer.apply ~params:(param "lm_head")
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
  (logits, state, cache)

let prefill_with_window ?window cfg vars cache ?attention_mask input_ids =
  let logits, _, cache =
    cached_causal_lm ?window ~cfg ~params:(Layer.params vars)
      ~state:(Layer.state vars) ~dtype:(Layer.dtype vars) ?attention_mask cache
      input_ids
  in
  (logits, cache)

let prefill cfg vars cache ?attention_mask input_ids =
  prefill_with_window cfg vars cache ?attention_mask input_ids

let decode_step_with_window ?window cfg vars cache input_ids =
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 || shape.(1) <> 1 then
    invalid_argf
      "Llama.decode_step: expected input IDs with shape [batch; 1], got [%s]"
      (String.concat "; " (List.map string_of_int (Array.to_list shape)));
  prefill_with_window ?window cfg vars cache input_ids

let decode_step cfg vars cache input_ids =
  decode_step_with_window cfg vars cache input_ids

module Pjrt = struct
  type 'layout t = 'layout Dense_pjrt.t

  let compile_with_window ?window ?(device_id = 0) cfg vars =
    let dtype = Layer.dtype vars in
    let params = Layer.params vars in
    let state = Layer.state vars in
    Dense_pjrt.compile ~device_id ~layer_count:cfg.num_hidden_layers ~dtype
      (fun ~attention_mask cache input_ids ->
        let logits, _, cache =
          cached_causal_lm ?window ~cfg ~params ~state ~dtype ~attention_mask
            cache input_ids
        in
        (logits, cache))

  let compile ?device_id cfg vars = compile_with_window ?device_id cfg vars
  let prefill = Dense_pjrt.prefill
  let decode_step = Dense_pjrt.decode_step

  module Resident = struct
    type 'layout cache = 'layout Dense_pjrt.resident

    let of_host = Dense_pjrt.resident_of_host
    let length = Dense_pjrt.resident_length
    let prefill = Dense_pjrt.resident_prefill
    let decode_step = Dense_pjrt.resident_decode_step
  end
end

let map_hf_weights ~cfg ~dtype tensors =
  let module Hf = Core.Hf in
  let weights = Hf.weights tensors in
  let vector name size =
    Hf.tensor weights ~name ~shape:[| size |] |> Hf.cast dtype
  in
  let matrix name ~rows ~cols = Hf.matrix weights dtype ~name ~rows ~cols in
  let projection name ~out_features =
    Ptree.dict
      [ ("weight", matrix name ~rows:out_features ~cols:cfg.hidden_size) ]
  in
  let layer index =
    let prefix = Printf.sprintf "model.layers.%d" index in
    Ptree.dict
      [
        ( "input_layernorm",
          Ptree.dict
            [
              ( "scale",
                vector (prefix ^ ".input_layernorm.weight") cfg.hidden_size );
            ] );
        ( "self_attn",
          Ptree.dict
            [
              ( "q_proj",
                projection
                  (prefix ^ ".self_attn.q_proj.weight")
                  ~out_features:
                    (cfg.num_attention_heads
                    * (cfg.hidden_size / cfg.num_attention_heads)) );
              ( "k_proj",
                projection
                  (prefix ^ ".self_attn.k_proj.weight")
                  ~out_features:
                    (cfg.num_key_value_heads
                    * (cfg.hidden_size / cfg.num_attention_heads)) );
              ( "v_proj",
                projection
                  (prefix ^ ".self_attn.v_proj.weight")
                  ~out_features:
                    (cfg.num_key_value_heads
                    * (cfg.hidden_size / cfg.num_attention_heads)) );
              ( "o_proj",
                Ptree.dict
                  [
                    ( "weight",
                      matrix
                        (prefix ^ ".self_attn.o_proj.weight")
                        ~rows:cfg.hidden_size
                        ~cols:
                          (cfg.num_attention_heads
                          * (cfg.hidden_size / cfg.num_attention_heads)) );
                  ] );
            ] );
        ( "post_attention_layernorm",
          Ptree.dict
            [
              ( "scale",
                vector
                  (prefix ^ ".post_attention_layernorm.weight")
                  cfg.hidden_size );
            ] );
        ( "mlp",
          Ptree.dict
            [
              ( "gate_proj",
                projection
                  (prefix ^ ".mlp.gate_proj.weight")
                  ~out_features:cfg.intermediate_size );
              ( "up_proj",
                projection
                  (prefix ^ ".mlp.up_proj.weight")
                  ~out_features:cfg.intermediate_size );
              ( "down_proj",
                Ptree.dict
                  [
                    ( "weight",
                      matrix
                        (prefix ^ ".mlp.down_proj.weight")
                        ~rows:cfg.hidden_size ~cols:cfg.intermediate_size );
                  ] );
            ] );
      ]
  in
  let params =
    [
      ( "embed_tokens",
        Ptree.dict
          [
            ( "weight",
              Hf.tensor weights ~name:"model.embed_tokens.weight"
                ~shape:[| cfg.vocab_size; cfg.hidden_size |]
              |> Hf.cast dtype );
          ] );
      ("layers", Ptree.list (List.init cfg.num_hidden_layers layer));
      ( "norm",
        Ptree.dict [ ("scale", vector "model.norm.weight" cfg.hidden_size) ] );
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
      (cfg.tie_word_embeddings && name = "lm_head.weight")
      || String.ends_with ~suffix:".rotary_emb.inv_freq" name);
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
  let decoder_block_with_window = decoder_block_with_window
  let decoder_with_window = decoder_with_window
  let for_causal_lm_with_window = for_causal_lm_with_window
  let prefill_with_window = prefill_with_window
  let decode_step_with_window = decode_step_with_window
  let pjrt_compile_with_window = Pjrt.compile_with_window
  let map_hf_weights = map_hf_weights
end
