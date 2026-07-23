(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Gemma3 = Hugr.Gemma3
module Layer = Kaun.Layer
module Ptree = Kaun.Ptree

let dtype = Nx.float32

let tiny () =
  Gemma3.config ~vocab_size:64 ~hidden_size:24 ~intermediate_size:32
    ~num_hidden_layers:3 ~num_attention_heads:4 ~num_key_value_heads:2
    ~head_dim:6 ~max_position_embeddings:32 ~sliding_window:2
    ~sliding_window_pattern:3 ()

let input values =
  Array.map Int32.of_int values
  |> Nx.create Nx.int32 [| 1; Array.length values |]

let max_difference expected actual =
  Nx.sub expected actual |> Nx.abs |> Nx.max |> Nx.item []

let test_qk_norm_parameters () =
  Nx.Rng.run ~seed:71 @@ fun () ->
  let model = Gemma3.for_causal_lm (tiny ()) () in
  let params = Layer.init model ~dtype |> Layer.params in
  let paths = Ptree.flatten_with_paths params |> List.map fst in
  is_true ~msg:"query norm parameter"
    (List.mem "layers.0.self_attn.q_norm.weight" paths);
  is_true ~msg:"key norm parameter"
    (List.mem "layers.0.self_attn.k_norm.weight" paths)

let test_local_global_pattern () =
  Nx.Rng.run ~seed:73 @@ fun () ->
  let cfg = tiny () in
  let hidden = Nx.randn dtype [| 1; 4; 24 |] in
  let changed =
    let first = Nx.add_s (Nx.slice [ Nx.A; Nx.R (0, 1); Nx.A ] hidden) 1.0 in
    Nx.concatenate ~axis:1
      [ first; Nx.slice [ Nx.A; Nx.R (1, 4); Nx.A ] hidden ]
  in
  let compare layer_index =
    let block = Gemma3.decoder_block cfg ~layer_index () in
    let vars = Layer.init block ~dtype in
    let original, _ = Layer.apply block vars ~training:false hidden in
    let altered, _ = Layer.apply block vars ~training:false changed in
    max_difference
      (Nx.slice [ Nx.A; Nx.I 3; Nx.A ] original)
      (Nx.slice [ Nx.A; Nx.I 3; Nx.A ] altered)
  in
  let local_difference = compare 1 in
  let global_difference = compare 2 in
  is_true
    ~msg:(Printf.sprintf "local layer distant error is %g" local_difference)
    (local_difference < 1e-5);
  is_true
    ~msg:(Printf.sprintf "global layer distant effect is %g" global_difference)
    (global_difference > 1e-5)

let test_cached_decode () =
  Nx.Rng.run ~seed:79 @@ fun () ->
  let cfg = tiny () in
  let model = Gemma3.for_causal_lm cfg () in
  let vars = Layer.init model ~dtype in
  let ids = input [| 1; 2; 3; 4 |] in
  let expected, _ = Layer.apply model vars ~training:false ids in
  let cache = Gemma3.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
  let rec decode position cache outputs =
    if position = 4 then List.rev outputs
    else
      let token = Nx.slice [ Nx.A; Nx.R (position, position + 1) ] ids in
      let logits, cache = Gemma3.decode_step cfg vars cache token in
      decode (position + 1) cache (logits :: outputs)
  in
  let actual = Nx.concatenate ~axis:1 (decode 0 cache []) in
  let difference = max_difference expected actual in
  is_true
    ~msg:(Printf.sprintf "cached Gemma 3 max error is %g" difference)
    (difference < 1e-5)

let test_cached_pjrt_cuda () =
  if Rune.Backend.available Pjrt_cuda then
    try
      Nx.Rng.run ~seed:83 @@ fun () ->
      let cfg = tiny () in
      let model = Gemma3.for_causal_lm cfg () in
      let vars = Layer.init model ~dtype in
      let ids = input [| 1; 2; 3 |] in
      let expected, _ = Layer.apply model vars ~training:false ids in
      let runner = Gemma3.Pjrt.compile cfg vars in
      let cache = Gemma3.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
      let prompt = Nx.slice [ Nx.A; Nx.R (0, 2) ] ids in
      let prompt_logits, cache = Gemma3.Pjrt.prefill runner cache prompt in
      let token = Nx.slice [ Nx.A; Nx.R (2, 3) ] ids in
      let token_logits, _ = Gemma3.Pjrt.decode_step runner cache token in
      let actual = Nx.concatenate ~axis:1 [ prompt_logits; token_logits ] in
      let difference = max_difference expected actual in
      is_true
        ~msg:(Printf.sprintf "PJRT cached Gemma 3 max error is %g" difference)
        (difference < 5e-4)
    with Rune_pjrt.Error.Error error ->
      failwith ("Gemma 3 cached PJRT CUDA: " ^ Rune_pjrt.Error.to_string error)

let () =
  run "Hugr.Gemma3"
    [
      group "model"
        [
          test "QK norm parameters" test_qk_norm_parameters;
          test "local/global pattern" test_local_global_pattern;
          test "cached decode" test_cached_decode;
          test "cached PJRT CUDA" test_cached_pjrt_cuda;
        ];
    ]
