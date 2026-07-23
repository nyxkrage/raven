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
  max_position_embeddings : int;
  rms_norm_eps : float;
  rope_theta : float;
  attention_dropout : float;
  initializer_range : float;
  tie_word_embeddings : bool;
}

let positive name value =
  if value <= 0 then
    invalid_argf "Llama.Config.make: %s must be positive, got %d" name value

let positive_float name value =
  if value <= 0.0 then
    invalid_argf "Llama.Config.make: %s must be positive, got %g" name value

let make ~vocab_size ~hidden_size ~intermediate_size ~num_hidden_layers
    ~num_attention_heads ?(num_key_value_heads = num_attention_heads)
    ?(max_position_embeddings = 2048) ?(rms_norm_eps = 1e-6)
    ?(rope_theta = 10000.0) ?(attention_dropout = 0.0)
    ?(initializer_range = 0.02) ?(tie_word_embeddings = false) () =
  positive "vocab_size" vocab_size;
  positive "hidden_size" hidden_size;
  positive "intermediate_size" intermediate_size;
  positive "num_hidden_layers" num_hidden_layers;
  positive "num_attention_heads" num_attention_heads;
  positive "num_key_value_heads" num_key_value_heads;
  positive "max_position_embeddings" max_position_embeddings;
  positive_float "rms_norm_eps" rms_norm_eps;
  positive_float "rope_theta" rope_theta;
  positive_float "initializer_range" initializer_range;
  if hidden_size mod num_attention_heads <> 0 then
    invalid_argf
      "Llama.Config.make: hidden_size (%d) must be divisible by \
       num_attention_heads (%d)"
      hidden_size num_attention_heads;
  if num_attention_heads mod num_key_value_heads <> 0 then
    invalid_argf
      "Llama.Config.make: num_attention_heads (%d) must be divisible by \
       num_key_value_heads (%d)"
      num_attention_heads num_key_value_heads;
  let head_dim = hidden_size / num_attention_heads in
  if head_dim mod 2 <> 0 then
    invalid_argf "Llama.Config.make: attention head dimension must be even";
  if attention_dropout < 0.0 || attention_dropout >= 1.0 then
    invalid_argf
      "Llama.Config.make: expected 0.0 <= attention_dropout < 1.0, got %g"
      attention_dropout;
  {
    vocab_size;
    hidden_size;
    intermediate_size;
    num_hidden_layers;
    num_attention_heads;
    num_key_value_heads;
    max_position_embeddings;
    rms_norm_eps;
    rope_theta;
    attention_dropout;
    initializer_range;
    tie_word_embeddings;
  }

let of_json json =
  let module Hf = Hugr_core.Hf in
  let ctx = "Llama.Config.of_json" in
  (match Hf.string_opt "model_type" json with
  | None | Some "llama" -> ()
  | Some model_type ->
      invalid_argf "%s: expected model_type \"llama\", got %S" ctx model_type);
  (match Hf.string_opt "hidden_act" json with
  | None | Some "silu" -> ()
  | Some hidden_act ->
      invalid_argf "%s: expected hidden_act \"silu\", got %S" ctx hidden_act);
  (match Hf.member "rope_scaling" json with
  | None | Some (Jsont.Null _) -> ()
  | Some _ ->
      invalid_argf
        "%s: rope_scaling is not supported yet; standard RoPE is required" ctx);
  if Option.value (Hf.bool_opt "attention_bias" json) ~default:false then
    invalid_argf "%s: attention_bias=true is not supported" ctx;
  if Option.value (Hf.bool_opt "mlp_bias" json) ~default:false then
    invalid_argf "%s: mlp_bias=true is not supported" ctx;
  let hidden_size = Hf.int_exn ~ctx "hidden_size" json in
  let num_attention_heads = Hf.int_exn ~ctx "num_attention_heads" json in
  (match Hf.int_opt "head_dim" json with
  | None -> ()
  | Some head_dim ->
      let expected = hidden_size / num_attention_heads in
      if head_dim <> expected then
        invalid_argf "%s: head_dim=%d differs from hidden_size / heads = %d" ctx
          head_dim expected);
  make
    ~vocab_size:(Hf.int_exn ~ctx "vocab_size" json)
    ~hidden_size
    ~intermediate_size:(Hf.int_exn ~ctx "intermediate_size" json)
    ~num_hidden_layers:(Hf.int_exn ~ctx "num_hidden_layers" json)
    ~num_attention_heads
    ?num_key_value_heads:(Hf.int_opt "num_key_value_heads" json)
    ?max_position_embeddings:(Hf.int_opt "max_position_embeddings" json)
    ?rms_norm_eps:(Hf.float_opt "rms_norm_eps" json)
    ?rope_theta:(Hf.float_opt "rope_theta" json)
    ?attention_dropout:(Hf.float_opt "attention_dropout" json)
    ?initializer_range:(Hf.float_opt "initializer_range" json)
    ?tie_word_embeddings:(Hf.bool_opt "tie_word_embeddings" json)
    ()
