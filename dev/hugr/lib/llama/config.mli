(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** LLaMA model configuration. *)

type t = {
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

val make :
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
  t
(** [make ~vocab_size ~hidden_size ~intermediate_size ~num_hidden_layers
     ~num_attention_heads ()] is a LLaMA configuration.

    [num_key_value_heads] defaults to [num_attention_heads].
    [max_position_embeddings] defaults to [2048], [rms_norm_eps] to [1e-6],
    [rope_theta] to [10000], [attention_dropout] to [0], [initializer_range] to
    [0.02], and [tie_word_embeddings] to [false].

    Raises [Invalid_argument] when dimensions, head counts, or numerical
    parameters are invalid. *)

val of_json : Jsont.json -> t
(** [of_json json] decodes a standard-RoPE Hugging Face LLaMA configuration.

    Raises [Invalid_argument] for unsupported activation, projection-bias,
    explicit-head-dimension, or RoPE-scaling variants. *)
