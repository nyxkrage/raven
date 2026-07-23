(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Falcon parallel-decoder configuration. *)

type t = {
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

val make :
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
  t

val of_json : Jsont.json -> t
