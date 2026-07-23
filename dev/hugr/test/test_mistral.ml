(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Layer = Kaun.Layer
module Mistral = Hugr.Mistral

let dtype = Nx.float32

let tiny () =
  Mistral.config ~vocab_size:64 ~hidden_size:32 ~intermediate_size:48
    ~num_hidden_layers:2 ~num_attention_heads:4 ~num_key_value_heads:2
    ~max_position_embeddings:32 ~sliding_window:2 ()

let input values =
  Array.map Int32.of_int values
  |> Nx.create Nx.int32 [| 1; Array.length values |]

let max_difference expected actual =
  Nx.sub expected actual |> Nx.abs |> Nx.max |> Nx.item []

let test_sliding_window () =
  Nx.Rng.run ~seed:41 @@ fun () ->
  let cfg = tiny () in
  let model = Mistral.for_causal_lm cfg () in
  let vars = Layer.init model ~dtype in
  let original, _ =
    Layer.apply model vars ~training:false (input [| 1; 2; 3; 4 |])
  in
  let distant, _ =
    Layer.apply model vars ~training:false (input [| 9; 2; 3; 4 |])
  in
  let nearby, _ =
    Layer.apply model vars ~training:false (input [| 1; 2; 9; 4 |])
  in
  let last tensor = Nx.slice [ Nx.A; Nx.I 3; Nx.A ] tensor in
  let distant_difference = max_difference (last original) (last distant) in
  let nearby_difference = max_difference (last original) (last nearby) in
  is_true
    ~msg:
      (Printf.sprintf "out-of-window token changed logits by %g"
         distant_difference)
    (distant_difference < 1e-5);
  is_true
    ~msg:
      (Printf.sprintf "in-window token changed logits by only %g"
         nearby_difference)
    (nearby_difference > 1e-5)

let test_cached_decode () =
  Nx.Rng.run ~seed:43 @@ fun () ->
  let cfg = tiny () in
  let model = Mistral.for_causal_lm cfg () in
  let vars = Layer.init model ~dtype in
  let ids = input [| 1; 2; 3; 4 |] in
  let expected, _ = Layer.apply model vars ~training:false ids in
  let cache = Mistral.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
  let rec decode position cache outputs =
    if position = 4 then (List.rev outputs, cache)
    else
      let token = Nx.slice [ Nx.A; Nx.R (position, position + 1) ] ids in
      let logits, cache = Mistral.decode_step cfg vars cache token in
      decode (position + 1) cache (logits :: outputs)
  in
  let outputs, cache = decode 0 cache [] in
  let actual = Nx.concatenate ~axis:1 outputs in
  let difference = max_difference expected actual in
  equal ~msg:"cache length" int 4 (Mistral.Cache.length cache);
  is_true
    ~msg:(Printf.sprintf "cached Mistral max error is %g" difference)
    (difference < 1e-5)

let test_cached_pjrt_cuda () =
  if Rune.Backend.available Pjrt_cuda then
    try
      Nx.Rng.run ~seed:47 @@ fun () ->
      let cfg = tiny () in
      let model = Mistral.for_causal_lm cfg () in
      let vars = Layer.init model ~dtype in
      let ids = input [| 1; 2; 3 |] in
      let expected, _ = Layer.apply model vars ~training:false ids in
      let runner = Mistral.Pjrt.compile cfg vars in
      let cache = Mistral.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
      let prompt = Nx.slice [ Nx.A; Nx.R (0, 2) ] ids in
      let prompt_logits, cache = Mistral.Pjrt.prefill runner cache prompt in
      let token = Nx.slice [ Nx.A; Nx.R (2, 3) ] ids in
      let token_logits, _ = Mistral.Pjrt.decode_step runner cache token in
      let actual = Nx.concatenate ~axis:1 [ prompt_logits; token_logits ] in
      let difference = max_difference expected actual in
      is_true
        ~msg:(Printf.sprintf "PJRT cached Mistral max error is %g" difference)
        (difference < 5e-4)
    with Rune_pjrt.Error.Error error ->
      failwith ("Mistral cached PJRT CUDA: " ^ Rune_pjrt.Error.to_string error)

let () =
  run "Hugr.Mistral"
    [
      group "model"
        [
          test "sliding window" test_sliding_window;
          test "cached decode" test_cached_decode;
          test "cached PJRT CUDA" test_cached_pjrt_cuda;
        ];
    ]
