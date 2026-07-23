(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun
module Llama_config = Config
module Config = Mistral_config

type config = Mistral_config.t = {
  vocab_size : int;
  hidden_size : int;
  intermediate_size : int;
  num_hidden_layers : int;
  num_attention_heads : int;
  num_key_value_heads : int;
  max_position_embeddings : int;
  sliding_window : int;
  rms_norm_eps : float;
  rope_theta : float;
  attention_dropout : float;
  initializer_range : float;
  tie_word_embeddings : bool;
}

let config = Mistral_config.make
let position_ids_key = Llama.position_ids_key
let attention_mask_key = Llama.attention_mask_key

let llama_config cfg =
  Llama_config.make ~vocab_size:cfg.vocab_size ~hidden_size:cfg.hidden_size
    ~intermediate_size:cfg.intermediate_size
    ~num_hidden_layers:cfg.num_hidden_layers
    ~num_attention_heads:cfg.num_attention_heads
    ~num_key_value_heads:cfg.num_key_value_heads
    ~max_position_embeddings:cfg.max_position_embeddings
    ~rms_norm_eps:cfg.rms_norm_eps ~rope_theta:cfg.rope_theta
    ~attention_dropout:cfg.attention_dropout
    ~initializer_range:cfg.initializer_range
    ~tie_word_embeddings:cfg.tie_word_embeddings ()

module Cache = struct
  type 'layout t = 'layout Llama.Cache.t

  let create cfg ~batch_size ~max_length ~dtype =
    Llama.Cache.create (llama_config cfg) ~batch_size ~max_length ~dtype

  let batch_size = Llama.Cache.batch_size
  let max_length = Llama.Cache.max_length
  let length = Llama.Cache.length
end

let decoder_block cfg () =
  Llama.Internal.decoder_block_with_window ~window:cfg.sliding_window
    (llama_config cfg) ()

let decoder cfg () =
  Llama.Internal.decoder_with_window ~window:cfg.sliding_window
    (llama_config cfg) ()

let for_causal_lm cfg () =
  Llama.Internal.for_causal_lm_with_window ~window:cfg.sliding_window
    (llama_config cfg) ()

let prefill cfg vars cache ?attention_mask input_ids =
  Llama.Internal.prefill_with_window ~window:cfg.sliding_window
    (llama_config cfg) vars cache ?attention_mask input_ids

let decode_step cfg vars cache input_ids =
  Llama.Internal.decode_step_with_window ~window:cfg.sliding_window
    (llama_config cfg) vars cache input_ids

module Pjrt = struct
  type 'layout t = 'layout Llama.Pjrt.t

  let compile ?device_id cfg vars =
    Llama.Internal.pjrt_compile_with_window ~window:cfg.sliding_window
      ?device_id (llama_config cfg) vars

  let prefill = Llama.Pjrt.prefill
  let decode_step = Llama.Pjrt.decode_step
end

let from_pretrained ?token ?cache_dir ?offline ?revision ~model_id ~dtype () =
  let json =
    Kaun_hf.load_config ?token ?cache_dir ?offline ?revision ~model_id ()
  in
  let cfg = Mistral_config.of_json json in
  let weights =
    Kaun_hf.load_weights ?token ?cache_dir ?offline ?revision ~model_id ()
  in
  let params =
    Llama.Internal.map_hf_weights ~cfg:(llama_config cfg) ~dtype weights
  in
  (cfg, params)
