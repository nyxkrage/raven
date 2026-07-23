(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gemma 2 decoder and causal language model. *)

open Kaun
module Config = Gemma2_config

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

val config :
  vocab_size:int ->
  hidden_size:int ->
  intermediate_size:int ->
  num_hidden_layers:int ->
  num_attention_heads:int ->
  num_key_value_heads:int ->
  head_dim:int ->
  ?max_position_embeddings:int ->
  ?sliding_window:int ->
  ?rms_norm_eps:float ->
  ?rope_theta:float ->
  ?attention_dropout:float ->
  ?attention_logit_softcapping:float ->
  ?final_logit_softcapping:float ->
  ?query_pre_attn_scalar:float ->
  ?initializer_range:float ->
  ?tie_word_embeddings:bool ->
  unit ->
  config

val position_ids_key : string
val attention_mask_key : string
val decoder_block : config -> layer_index:int -> unit -> (float, float) Layer.t
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

module Internal : sig
  type profile

  val profile :
    window:(int -> int option) ->
    rope_theta:(int -> float) ->
    qk_norm_eps:float option ->
    attention_logit_softcapping:float option ->
    final_logit_softcapping:float option ->
    profile

  val decoder_block_with_profile :
    config -> profile -> layer_index:int -> unit -> (float, float) Layer.t

  val decoder_with_profile : config -> profile -> unit -> (int32, float) Layer.t

  val for_causal_lm_with_profile :
    config -> profile -> unit -> (int32, float) Layer.t

  val prefill_with_profile :
    config ->
    profile ->
    'layout Layer.vars ->
    'layout Cache.t ->
    ?attention_mask:Nx.bool_t ->
    Nx.int32_t ->
    (float, 'layout) Nx.t * 'layout Cache.t

  val decode_step_with_profile :
    config ->
    profile ->
    'layout Layer.vars ->
    'layout Cache.t ->
    Nx.int32_t ->
    (float, 'layout) Nx.t * 'layout Cache.t

  val pjrt_compile_with_profile :
    ?device_id:int -> config -> profile -> 'layout Layer.vars -> 'layout Pjrt.t

  val map_hf_weights :
    ?qk_norm:bool ->
    ?model_prefix:string ->
    ?allow_unconsumed:(string -> bool) ->
    cfg:config ->
    dtype:(float, 'layout) Nx.dtype ->
    (string * Ptree.tensor) list ->
    Ptree.t
end
[@@ocaml.doc "@hidden"]

val from_pretrained :
  ?token:string ->
  ?cache_dir:string ->
  ?offline:bool ->
  ?revision:Kaun_hf.revision ->
  model_id:string ->
  dtype:(float, 'layout) Nx.dtype ->
  unit ->
  config * Ptree.t
