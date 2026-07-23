(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** LLaMA decoder and causal language model. *)

open Kaun

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
(** The type for LLaMA configurations. *)

val config :
  vocab_size:int ->
  hidden_size:int ->
  intermediate_size:int ->
  num_hidden_layers:int ->
  num_attention_heads:int ->
  ?num_key_value_heads:int ->
  ?max_position_embeddings:int ->
  ?rms_norm_eps:float ->
  ?rope_theta:float ->
  ?attention_dropout:float ->
  ?initializer_range:float ->
  ?tie_word_embeddings:bool ->
  unit ->
  config
(** Alias for {!Config.make}. *)

val position_ids_key : string
(** [position_ids_key] is ["position_ids"], the {!Context} key for explicit
    int32 position IDs with shape [[seq]] or [[batch; seq]]. *)

val attention_mask_key : string
(** [attention_mask_key] is ["attention_mask"], the {!Context} key for a bool or
    integer padding mask with shape [[batch; seq]]. *)

module Cache : sig
  type 'layout t
  (** A fixed-capacity per-layer key/value cache. Cache values are explicit and
      request-owned; model variables remain immutable and shareable. *)

  val create :
    config ->
    batch_size:int ->
    max_length:int ->
    dtype:(float, 'layout) Nx.dtype ->
    'layout t
  (** [create cfg ~batch_size ~max_length ~dtype] is an empty cache. *)

  val batch_size : 'layout t -> int
  (** [batch_size cache] is the fixed batch size. *)

  val max_length : 'layout t -> int
  (** [max_length cache] is the fixed token capacity. *)

  val length : 'layout t -> int
  (** [length cache] is the number of positions appended so far. *)
end

val decoder_block : config -> unit -> (float, float) Layer.t
(** [decoder_block cfg ()] is one sequential pre-RMSNorm LLaMA block. *)

val decoder : config -> unit -> (int32, float) Layer.t
(** [decoder cfg ()] is the LLaMA decoder.

    Input IDs have shape [[batch; seq]]. The result contains hidden states with
    shape [[batch; seq; hidden_size]]. *)

val for_causal_lm : config -> unit -> (int32, float) Layer.t
(** [for_causal_lm cfg ()] is the LLaMA decoder followed by its language-model
    head. The head reuses token embeddings when [cfg.tie_word_embeddings] is
    [true]. *)

val prefill :
  config ->
  'layout Layer.vars ->
  'layout Cache.t ->
  ?attention_mask:Nx.bool_t ->
  Nx.int32_t ->
  (float, 'layout) Nx.t * 'layout Cache.t
(** [prefill cfg vars cache ?attention_mask input_ids] appends a prompt or
    prompt chunk to [cache] and returns logits for every appended token.
    [input_ids] and [attention_mask] have shape [[batch; seq]]. *)

val decode_step :
  config ->
  'layout Layer.vars ->
  'layout Cache.t ->
  Nx.int32_t ->
  (float, 'layout) Nx.t * 'layout Cache.t
(** [decode_step cfg vars cache input_ids] appends one token per batch and
    returns logits with shape [[batch; 1; vocab_size]]. *)

module Pjrt : sig
  type 'layout t
  (** A PJRT CUDA runner with compiled programs cached by input signature. *)

  val compile : ?device_id:int -> config -> 'layout Layer.vars -> 'layout t
  (** [compile ?device_id cfg vars] prepares cached inference on a CUDA device.
      The first call for each prompt shape compiles that shape; one-token decode
      calls reuse a single compiled program. *)

  val prefill :
    'layout t ->
    'layout Cache.t ->
    ?attention_mask:Nx.bool_t ->
    Nx.int32_t ->
    (float, 'layout) Nx.t * 'layout Cache.t
  (** [prefill runner cache ?attention_mask input_ids] appends a prompt using
      PJRT CUDA. *)

  val decode_step :
    'layout t ->
    'layout Cache.t ->
    Nx.int32_t ->
    (float, 'layout) Nx.t * 'layout Cache.t
  (** [decode_step runner cache input_ids] appends one token using PJRT CUDA. *)
end

module Internal : sig
  val decoder_block_with_window :
    ?window:int -> config -> unit -> (float, float) Layer.t

  val decoder_with_window :
    ?window:int -> config -> unit -> (int32, float) Layer.t

  val for_causal_lm_with_window :
    ?window:int -> config -> unit -> (int32, float) Layer.t

  val prefill_with_window :
    ?window:int ->
    config ->
    'layout Layer.vars ->
    'layout Cache.t ->
    ?attention_mask:Nx.bool_t ->
    Nx.int32_t ->
    (float, 'layout) Nx.t * 'layout Cache.t

  val decode_step_with_window :
    ?window:int ->
    config ->
    'layout Layer.vars ->
    'layout Cache.t ->
    Nx.int32_t ->
    (float, 'layout) Nx.t * 'layout Cache.t

  val pjrt_compile_with_window :
    ?window:int ->
    ?device_id:int ->
    config ->
    'layout Layer.vars ->
    'layout Pjrt.t

  val map_hf_weights :
    cfg:config ->
    dtype:(float, 'layout) Nx.dtype ->
    (string * Kaun.Ptree.tensor) list ->
    Kaun.Ptree.t
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
(** [from_pretrained ~model_id ~dtype ()] loads a Hugging Face LLaMA-family
    configuration using standard RoPE and maps its weights into Hugr's parameter
    tree. *)
