(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Mistral decoder and causal language model. *)

open Kaun

module Config = Mistral_config
(** Mistral model configuration. *)

type config = Config.t = {
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
(** The type for Mistral configurations. *)

val config :
  vocab_size:int ->
  hidden_size:int ->
  intermediate_size:int ->
  num_hidden_layers:int ->
  num_attention_heads:int ->
  ?num_key_value_heads:int ->
  ?max_position_embeddings:int ->
  ?sliding_window:int ->
  ?rms_norm_eps:float ->
  ?rope_theta:float ->
  ?attention_dropout:float ->
  ?initializer_range:float ->
  ?tie_word_embeddings:bool ->
  unit ->
  config
(** Alias for {!Config.make}. *)

val position_ids_key : string
val attention_mask_key : string

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

val decoder_block : config -> unit -> (float, float) Layer.t
val decoder : config -> unit -> (int32, float) Layer.t
val for_causal_lm : config -> unit -> (int32, float) Layer.t

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
(** [from_pretrained ~model_id ~dtype ()] loads a standard-RoPE Hugging Face
    Mistral model. *)
