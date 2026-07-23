(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Gemma2 = Hugr.Gemma2
module Layer = Kaun.Layer

let dtype = Nx.float32

let tiny () =
  Gemma2.config ~vocab_size:64 ~hidden_size:32 ~intermediate_size:48
    ~num_hidden_layers:2 ~num_attention_heads:4 ~num_key_value_heads:2
    ~head_dim:6 ~max_position_embeddings:32 ~sliding_window:2
    ~final_logit_softcapping:3.0 ()

let input values =
  Array.map Int32.of_int values
  |> Nx.create Nx.int32 [| 1; Array.length values |]

let max_difference expected actual =
  Nx.sub expected actual |> Nx.abs |> Nx.max |> Nx.item []

let test_shape_and_softcap () =
  Nx.Rng.run ~seed:53 @@ fun () ->
  let cfg = tiny () in
  let model = Gemma2.for_causal_lm cfg () in
  let vars = Layer.init model ~dtype in
  let logits, _ =
    Layer.apply model vars ~training:false (input [| 1; 2; 3; 4 |])
  in
  equal ~msg:"logit shape" (list int) [ 1; 4; 64 ]
    (Array.to_list (Nx.shape logits));
  let largest = Nx.abs logits |> Nx.max |> Nx.item [] in
  is_true
    ~msg:(Printf.sprintf "softcapped logits reached %g" largest)
    (largest <= 3.0)

let test_alternating_attention () =
  Nx.Rng.run ~seed:59 @@ fun () ->
  let cfg = tiny () in
  let hidden = Nx.randn dtype [| 1; 4; 32 |] in
  let changed =
    let first = Nx.add_s (Nx.slice [ Nx.A; Nx.R (0, 1); Nx.A ] hidden) 1.0 in
    Nx.concatenate ~axis:1
      [ first; Nx.slice [ Nx.A; Nx.R (1, 4); Nx.A ] hidden ]
  in
  let compare layer_index =
    let block = Gemma2.decoder_block cfg ~layer_index () in
    let vars = Layer.init block ~dtype in
    let original, _ = Layer.apply block vars ~training:false hidden in
    let altered, _ = Layer.apply block vars ~training:false changed in
    max_difference
      (Nx.slice [ Nx.A; Nx.I 3; Nx.A ] original)
      (Nx.slice [ Nx.A; Nx.I 3; Nx.A ] altered)
  in
  let local_difference = compare 0 in
  let global_difference = compare 1 in
  is_true
    ~msg:(Printf.sprintf "local layer distant error is %g" local_difference)
    (local_difference < 1e-5);
  is_true
    ~msg:(Printf.sprintf "global layer distant effect is %g" global_difference)
    (global_difference > 1e-5)

let test_cached_decode () =
  Nx.Rng.run ~seed:61 @@ fun () ->
  let cfg = tiny () in
  let model = Gemma2.for_causal_lm cfg () in
  let vars = Layer.init model ~dtype in
  let ids = input [| 1; 2; 3; 4 |] in
  let expected, _ = Layer.apply model vars ~training:false ids in
  let cache = Gemma2.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
  let rec decode position cache outputs =
    if position = 4 then List.rev outputs
    else
      let token = Nx.slice [ Nx.A; Nx.R (position, position + 1) ] ids in
      let logits, cache = Gemma2.decode_step cfg vars cache token in
      decode (position + 1) cache (logits :: outputs)
  in
  let actual = Nx.concatenate ~axis:1 (decode 0 cache []) in
  let difference = max_difference expected actual in
  is_true
    ~msg:(Printf.sprintf "cached Gemma 2 max error is %g" difference)
    (difference < 1e-5)

let test_cached_pjrt_cuda () =
  if Rune.Backend.available Pjrt_cuda then
    try
      Nx.Rng.run ~seed:67 @@ fun () ->
      let cfg = tiny () in
      let model = Gemma2.for_causal_lm cfg () in
      let vars = Layer.init model ~dtype in
      let ids = input [| 1; 2; 3 |] in
      let expected, _ = Layer.apply model vars ~training:false ids in
      let runner = Gemma2.Pjrt.compile cfg vars in
      let cache = Gemma2.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
      let prompt = Nx.slice [ Nx.A; Nx.R (0, 2) ] ids in
      let prompt_logits, cache = Gemma2.Pjrt.prefill runner cache prompt in
      let token = Nx.slice [ Nx.A; Nx.R (2, 3) ] ids in
      let token_logits, _ = Gemma2.Pjrt.decode_step runner cache token in
      let actual = Nx.concatenate ~axis:1 [ prompt_logits; token_logits ] in
      let difference = max_difference expected actual in
      is_true
        ~msg:(Printf.sprintf "PJRT cached Gemma 2 max error is %g" difference)
        (difference < 5e-4)
    with Rune_pjrt.Error.Error error ->
      failwith ("Gemma 2 cached PJRT CUDA: " ^ Rune_pjrt.Error.to_string error)

let () =
  run "Hugr.Gemma2"
    [
      group "model"
        [
          test "shape and softcap" test_shape_and_softcap;
          test "alternating attention" test_alternating_attention;
          test "cached decode" test_cached_decode;
          test "cached PJRT CUDA" test_cached_pjrt_cuda;
        ];
    ]
