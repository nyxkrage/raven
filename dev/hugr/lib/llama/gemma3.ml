(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun
module Config = Gemma3_config

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
  sliding_window_pattern : int;
  rms_norm_eps : float;
  rope_theta : float;
  rope_local_base_freq : float;
  attention_dropout : float;
  attention_logit_softcapping : float option;
  final_logit_softcapping : float option;
  query_pre_attn_scalar : float;
  initializer_range : float;
  tie_word_embeddings : bool;
}

let config = Config.make
let position_ids_key = Gemma2.position_ids_key
let attention_mask_key = Gemma2.attention_mask_key

let gemma2_config cfg =
  Gemma2.Config.make ~vocab_size:cfg.vocab_size ~hidden_size:cfg.hidden_size
    ~intermediate_size:cfg.intermediate_size
    ~num_hidden_layers:cfg.num_hidden_layers
    ~num_attention_heads:cfg.num_attention_heads
    ~num_key_value_heads:cfg.num_key_value_heads ~head_dim:cfg.head_dim
    ~max_position_embeddings:cfg.max_position_embeddings
    ~sliding_window:cfg.sliding_window ~rms_norm_eps:cfg.rms_norm_eps
    ~rope_theta:cfg.rope_theta ~attention_dropout:cfg.attention_dropout
    ~attention_logit_softcapping:
      (Option.value cfg.attention_logit_softcapping ~default:1.0)
    ~final_logit_softcapping:
      (Option.value cfg.final_logit_softcapping ~default:1.0)
    ~query_pre_attn_scalar:cfg.query_pre_attn_scalar
    ~initializer_range:cfg.initializer_range
    ~tie_word_embeddings:cfg.tie_word_embeddings ()

let is_global cfg layer_index =
  (layer_index + 1) mod cfg.sliding_window_pattern = 0

let profile cfg =
  Gemma2.Internal.profile
    ~window:(fun layer_index ->
      if is_global cfg layer_index then None else Some cfg.sliding_window)
    ~rope_theta:(fun layer_index ->
      if is_global cfg layer_index then cfg.rope_theta
      else cfg.rope_local_base_freq)
    ~qk_norm_eps:(Some cfg.rms_norm_eps)
    ~attention_logit_softcapping:cfg.attention_logit_softcapping
    ~final_logit_softcapping:cfg.final_logit_softcapping

module Cache = struct
  type 'layout t = 'layout Gemma2.Cache.t

  let create cfg ~batch_size ~max_length ~dtype =
    Gemma2.Cache.create (gemma2_config cfg) ~batch_size ~max_length ~dtype

  let batch_size = Gemma2.Cache.batch_size
  let max_length = Gemma2.Cache.max_length
  let length = Gemma2.Cache.length
end

let decoder_block cfg ~layer_index () =
  Gemma2.Internal.decoder_block_with_profile (gemma2_config cfg) (profile cfg)
    ~layer_index ()

let decoder cfg () =
  Gemma2.Internal.decoder_with_profile (gemma2_config cfg) (profile cfg) ()

let for_causal_lm cfg () =
  Gemma2.Internal.for_causal_lm_with_profile (gemma2_config cfg) (profile cfg)
    ()

let prefill cfg vars cache ?attention_mask input_ids =
  Gemma2.Internal.prefill_with_profile (gemma2_config cfg) (profile cfg) vars
    cache ?attention_mask input_ids

let decode_step cfg vars cache input_ids =
  Gemma2.Internal.decode_step_with_profile (gemma2_config cfg) (profile cfg)
    vars cache input_ids

module Pjrt = struct
  type 'layout t = 'layout Gemma2.Pjrt.t

  let compile ?device_id cfg vars =
    Gemma2.Internal.pjrt_compile_with_profile ?device_id (gemma2_config cfg)
      (profile cfg) vars

  let prefill = Gemma2.Pjrt.prefill
  let decode_step = Gemma2.Pjrt.decode_step
end

let from_pretrained ?token ?cache_dir ?offline ?revision ~model_id ~dtype () =
  let json =
    Kaun_hf.load_config ?token ?cache_dir ?offline ?revision ~model_id ()
  in
  let cfg = Config.of_json json in
  let weights =
    Kaun_hf.load_weights ?token ?cache_dir ?offline ?revision ~model_id ()
  in
  let model_prefix =
    if
      List.exists
        (fun (name, _) -> name = "model.language_model.embed_tokens.weight")
        weights
    then "model.language_model"
    else "model"
  in
  let params =
    Gemma2.Internal.map_hf_weights ~qk_norm:true ~model_prefix
      ~allow_unconsumed:(fun name ->
        model_prefix = "model.language_model"
        && (String.starts_with ~prefix:"model.vision_tower." name
           || String.starts_with ~prefix:"model.multi_modal_projector." name))
      ~cfg:(gemma2_config cfg) ~dtype weights
  in
  (cfg, params)
