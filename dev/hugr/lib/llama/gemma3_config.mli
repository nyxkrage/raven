(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gemma 3 text model configuration. *)

type t = {
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

val make :
  vocab_size:int ->
  hidden_size:int ->
  intermediate_size:int ->
  num_hidden_layers:int ->
  num_attention_heads:int ->
  num_key_value_heads:int ->
  head_dim:int ->
  ?max_position_embeddings:int ->
  ?sliding_window:int ->
  ?sliding_window_pattern:int ->
  ?rms_norm_eps:float ->
  ?rope_theta:float ->
  ?rope_local_base_freq:float ->
  ?attention_dropout:float ->
  ?attention_logit_softcapping:float ->
  ?final_logit_softcapping:float ->
  ?query_pre_attn_scalar:float ->
  ?initializer_range:float ->
  ?tie_word_embeddings:bool ->
  unit ->
  t

val of_json : Jsont.json -> t
