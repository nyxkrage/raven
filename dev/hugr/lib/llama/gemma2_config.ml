(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

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
  rms_norm_eps : float;
  rope_theta : float;
  attention_dropout : float;
  attention_logit_softcapping : float;
  final_logit_softcapping : float;
  query_pre_attn_scalar : float;
  initializer_range : float;
  tie_word_embeddings : bool;
}

let positive name value =
  if value <= 0 then
    invalid_argf "Gemma2.Config.make: %s must be positive, got %d" name value

let positive_float name value =
  if value <= 0.0 then
    invalid_argf "Gemma2.Config.make: %s must be positive, got %g" name value

let make ~vocab_size ~hidden_size ~intermediate_size ~num_hidden_layers
    ~num_attention_heads ~num_key_value_heads ~head_dim
    ?(max_position_embeddings = 8192) ?(sliding_window = 4096)
    ?(rms_norm_eps = 1e-6) ?(rope_theta = 10000.0) ?(attention_dropout = 0.0)
    ?(attention_logit_softcapping = 50.0) ?(final_logit_softcapping = 30.0)
    ?query_pre_attn_scalar ?(initializer_range = 0.02)
    ?(tie_word_embeddings = true) () =
  List.iter
    (fun (name, value) -> positive name value)
    [
      ("vocab_size", vocab_size);
      ("hidden_size", hidden_size);
      ("intermediate_size", intermediate_size);
      ("num_hidden_layers", num_hidden_layers);
      ("num_attention_heads", num_attention_heads);
      ("num_key_value_heads", num_key_value_heads);
      ("head_dim", head_dim);
      ("max_position_embeddings", max_position_embeddings);
      ("sliding_window", sliding_window);
    ];
  if num_attention_heads mod num_key_value_heads <> 0 then
    invalid_argf
      "Gemma2.Config.make: num_attention_heads (%d) must be divisible by \
       num_key_value_heads (%d)"
      num_attention_heads num_key_value_heads;
  List.iter
    (fun (name, value) -> positive_float name value)
    [
      ("rms_norm_eps", rms_norm_eps);
      ("rope_theta", rope_theta);
      ("attention_logit_softcapping", attention_logit_softcapping);
      ("final_logit_softcapping", final_logit_softcapping);
      ("initializer_range", initializer_range);
    ];
  if attention_dropout < 0.0 || attention_dropout >= 1.0 then
    invalid_argf
      "Gemma2.Config.make: expected 0.0 <= attention_dropout < 1.0, got %g"
      attention_dropout;
  let query_pre_attn_scalar =
    Option.value query_pre_attn_scalar ~default:(float_of_int head_dim)
  in
  positive_float "query_pre_attn_scalar" query_pre_attn_scalar;
  {
    vocab_size;
    hidden_size;
    intermediate_size;
    num_hidden_layers;
    num_attention_heads;
    num_key_value_heads;
    head_dim;
    max_position_embeddings;
    sliding_window;
    rms_norm_eps;
    rope_theta;
    attention_dropout;
    attention_logit_softcapping;
    final_logit_softcapping;
    query_pre_attn_scalar;
    initializer_range;
    tie_word_embeddings;
  }

let of_json json =
  let module Hf = Hugr_core.Hf in
  let ctx = "Gemma2.Config.of_json" in
  (match Hf.string_opt "model_type" json with
  | None | Some "gemma2" -> ()
  | Some model_type ->
      invalid_argf "%s: expected model_type \"gemma2\", got %S" ctx model_type);
  (match Hf.string_opt "hidden_activation" json with
  | None | Some "gelu_pytorch_tanh" -> ()
  | Some activation ->
      invalid_argf
        "%s: expected hidden_activation \"gelu_pytorch_tanh\", got %S" ctx
        activation);
  make
    ~vocab_size:(Hf.int_exn ~ctx "vocab_size" json)
    ~hidden_size:(Hf.int_exn ~ctx "hidden_size" json)
    ~intermediate_size:(Hf.int_exn ~ctx "intermediate_size" json)
    ~num_hidden_layers:(Hf.int_exn ~ctx "num_hidden_layers" json)
    ~num_attention_heads:(Hf.int_exn ~ctx "num_attention_heads" json)
    ~num_key_value_heads:(Hf.int_exn ~ctx "num_key_value_heads" json)
    ~head_dim:(Hf.int_exn ~ctx "head_dim" json)
    ?max_position_embeddings:(Hf.int_opt "max_position_embeddings" json)
    ?sliding_window:(Hf.int_opt "sliding_window" json)
    ?rms_norm_eps:(Hf.float_opt "rms_norm_eps" json)
    ?rope_theta:(Hf.float_opt "rope_theta" json)
    ?attention_dropout:(Hf.float_opt "attention_dropout" json)
    ?attention_logit_softcapping:(Hf.float_opt "attn_logit_softcapping" json)
    ?final_logit_softcapping:(Hf.float_opt "final_logit_softcapping" json)
    ?query_pre_attn_scalar:(Hf.float_opt "query_pre_attn_scalar" json)
    ?initializer_range:(Hf.float_opt "initializer_range" json)
    ?tie_word_embeddings:(Hf.bool_opt "tie_word_embeddings" json)
    ()
