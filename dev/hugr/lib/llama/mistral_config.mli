(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Mistral model configuration. *)

type t = {
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

val make :
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
  t
(** [make ~vocab_size ~hidden_size ~intermediate_size ~num_hidden_layers
     ~num_attention_heads ()] is a Mistral configuration. *)

val of_json : Jsont.json -> t
(** [of_json json] decodes a standard-RoPE Hugging Face Mistral configuration.
*)
