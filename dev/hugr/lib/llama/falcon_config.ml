(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

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

let make ~vocab_size ~hidden_size ?ffn_hidden_size ~num_hidden_layers
    ~num_attention_heads ?num_key_value_heads ?(max_position_embeddings = 2048)
    ?(layer_norm_epsilon = 1e-5) ?(rope_theta = 10000.0) ?(hidden_dropout = 0.0)
    ?(attention_dropout = 0.0) ?(initializer_range = 0.02) ?(bias = false)
    ?(new_decoder_architecture = false) ?(multi_query = true)
    ?num_ln_in_parallel_attn ?(tie_word_embeddings = true) () =
  let ffn_hidden_size =
    Option.value ffn_hidden_size ~default:(4 * hidden_size)
  in
  let num_key_value_heads =
    match (new_decoder_architecture, multi_query, num_key_value_heads) with
    | true, _, Some value -> value
    | true, _, None -> num_attention_heads
    | false, true, Some 1 | false, true, None -> 1
    | false, true, Some value ->
        invalid_argf
          "Falcon.Config.make: multi-query attention requires one KV head, got \
           %d"
          value
    | false, false, Some value when value = num_attention_heads -> value
    | false, false, None -> num_attention_heads
    | false, false, Some value ->
        invalid_argf
          "Falcon.Config.make: classic multi-head Falcon requires %d KV heads, \
           got %d"
          num_attention_heads value
  in
  let num_ln_in_parallel_attn =
    Option.value num_ln_in_parallel_attn
      ~default:(if new_decoder_architecture then 2 else 1)
  in
  List.iter
    (fun (name, value) ->
      if value <= 0 then
        invalid_argf "Falcon.Config.make: %s must be positive, got %d" name
          value)
    [
      ("vocab_size", vocab_size);
      ("hidden_size", hidden_size);
      ("ffn_hidden_size", ffn_hidden_size);
      ("num_hidden_layers", num_hidden_layers);
      ("num_attention_heads", num_attention_heads);
      ("num_key_value_heads", num_key_value_heads);
      ("max_position_embeddings", max_position_embeddings);
    ];
  if hidden_size mod num_attention_heads <> 0 then
    invalid_argf
      "Falcon.Config.make: hidden_size (%d) must be divisible by \
       num_attention_heads (%d)"
      hidden_size num_attention_heads;
  if num_attention_heads mod num_key_value_heads <> 0 then
    invalid_argf
      "Falcon.Config.make: num_attention_heads (%d) must be divisible by \
       num_key_value_heads (%d)"
      num_attention_heads num_key_value_heads;
  let head_dim = hidden_size / num_attention_heads in
  if head_dim mod 2 <> 0 then
    invalid_argf "Falcon.Config.make: attention head dimension must be even";
  if num_ln_in_parallel_attn <> 1 && num_ln_in_parallel_attn <> 2 then
    invalid_argf
      "Falcon.Config.make: num_ln_in_parallel_attn must be 1 or 2, got %d"
      num_ln_in_parallel_attn;
  if num_ln_in_parallel_attn = 2 && not new_decoder_architecture then
    invalid_argf
      "Falcon.Config.make: two parallel layer norms require \
       new_decoder_architecture=true";
  List.iter
    (fun (name, value) ->
      if value <= 0.0 then
        invalid_argf "Falcon.Config.make: %s must be positive, got %g" name
          value)
    [
      ("layer_norm_epsilon", layer_norm_epsilon);
      ("rope_theta", rope_theta);
      ("initializer_range", initializer_range);
    ];
  List.iter
    (fun (name, value) ->
      if value < 0.0 || value >= 1.0 then
        invalid_argf "Falcon.Config.make: expected 0 <= %s < 1, got %g" name
          value)
    [
      ("hidden_dropout", hidden_dropout);
      ("attention_dropout", attention_dropout);
    ];
  {
    vocab_size;
    hidden_size;
    ffn_hidden_size;
    num_hidden_layers;
    num_attention_heads;
    num_key_value_heads;
    max_position_embeddings;
    layer_norm_epsilon;
    rope_theta;
    hidden_dropout;
    attention_dropout;
    initializer_range;
    bias;
    new_decoder_architecture;
    multi_query;
    num_ln_in_parallel_attn;
    tie_word_embeddings;
  }

let of_json json =
  let module Hf = Hugr_core.Hf in
  let ctx = "Falcon.Config.of_json" in
  (match Hf.string_opt "model_type" json with
  | None | Some "falcon" -> ()
  | Some model_type ->
      invalid_argf "%s: expected model_type \"falcon\", got %S" ctx model_type);
  if Option.value (Hf.bool_opt "alibi" json) ~default:false then
    invalid_argf "%s: ALiBi Falcon variants are not supported" ctx;
  if not (Option.value (Hf.bool_opt "parallel_attn" json) ~default:true) then
    invalid_argf "%s: sequential-attention Falcon variants are not supported"
      ctx;
  (match Hf.string_opt "activation" json with
  | None | Some "gelu" -> ()
  | Some activation ->
      invalid_argf "%s: expected activation \"gelu\", got %S" ctx activation);
  (match Hf.member "rope_scaling" json with
  | None | Some (Jsont.Null _) -> ()
  | Some _ -> invalid_argf "%s: RoPE scaling is not supported" ctx);
  make
    ~vocab_size:(Hf.int_exn ~ctx "vocab_size" json)
    ~hidden_size:(Hf.int_exn ~ctx "hidden_size" json)
    ?ffn_hidden_size:(Hf.int_opt "ffn_hidden_size" json)
    ~num_hidden_layers:(Hf.int_exn ~ctx "num_hidden_layers" json)
    ~num_attention_heads:(Hf.int_exn ~ctx "num_attention_heads" json)
    ?num_key_value_heads:(Hf.int_opt "num_kv_heads" json)
    ?max_position_embeddings:(Hf.int_opt "max_position_embeddings" json)
    ?layer_norm_epsilon:(Hf.float_opt "layer_norm_epsilon" json)
    ?rope_theta:(Hf.float_opt "rope_theta" json)
    ?hidden_dropout:(Hf.float_opt "hidden_dropout" json)
    ?attention_dropout:(Hf.float_opt "attention_dropout" json)
    ?initializer_range:(Hf.float_opt "initializer_range" json)
    ?bias:(Hf.bool_opt "bias" json)
    ?new_decoder_architecture:(Hf.bool_opt "new_decoder_architecture" json)
    ?multi_query:(Hf.bool_opt "multi_query" json)
    ?num_ln_in_parallel_attn:(Hf.int_opt "num_ln_in_parallel_attn" json)
    ?tie_word_embeddings:(Hf.bool_opt "tie_word_embeddings" json)
    ()
