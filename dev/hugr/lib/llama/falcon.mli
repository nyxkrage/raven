(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Falcon parallel decoder and causal language model. *)

open Kaun
module Config = Falcon_config

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

val config :
  vocab_size:int ->
  hidden_size:int ->
  ?ffn_hidden_size:int ->
  num_hidden_layers:int ->
  num_attention_heads:int ->
  ?num_key_value_heads:int ->
  ?max_position_embeddings:int ->
  ?layer_norm_epsilon:float ->
  ?rope_theta:float ->
  ?hidden_dropout:float ->
  ?attention_dropout:float ->
  ?initializer_range:float ->
  ?bias:bool ->
  ?new_decoder_architecture:bool ->
  ?multi_query:bool ->
  ?num_ln_in_parallel_attn:int ->
  ?tie_word_embeddings:bool ->
  unit ->
  config

val position_ids_key : string
val attention_mask_key : string
val decoder_block : config -> unit -> (float, float) Layer.t
val decoder : config -> unit -> (int32, float) Layer.t
val for_causal_lm : config -> unit -> (int32, float) Layer.t

module Cache : sig
  type 'layout t

  val create :
    config ->
    batch_size:int ->
    max_length:int ->
    dtype:(float, 'layout) Nx.dtype ->
    'layout t

  val batch_size : 'layout t -> int
  val max_length : 'layout t -> int
  val length : 'layout t -> int
end

val prefill :
  config ->
  'layout Layer.vars ->
  'layout Cache.t ->
  ?attention_mask:Nx.bool_t ->
  Nx.int32_t ->
  (float, 'layout) Nx.t * 'layout Cache.t

val decode_step :
  config ->
  'layout Layer.vars ->
  'layout Cache.t ->
  Nx.int32_t ->
  (float, 'layout) Nx.t * 'layout Cache.t

module Pjrt : sig
  type 'layout t

  val compile : ?device_id:int -> config -> 'layout Layer.vars -> 'layout t

  val prefill :
    'layout t ->
    'layout Cache.t ->
    ?attention_mask:Nx.bool_t ->
    Nx.int32_t ->
    (float, 'layout) Nx.t * 'layout Cache.t

  val decode_step :
    'layout t ->
    'layout Cache.t ->
    Nx.int32_t ->
    (float, 'layout) Nx.t * 'layout Cache.t
end

val from_pretrained :
  ?token:string ->
  ?cache_dir:string ->
  ?offline:bool ->
  ?revision:Kaun_hf.revision ->
  model_id:string ->
  dtype:(float, 'layout) Nx.dtype ->
  unit ->
  config * Ptree.t
