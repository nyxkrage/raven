(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Falcon = Hugr.Falcon
module Layer = Kaun.Layer
module Ptree = Kaun.Ptree

let dtype = Nx.float32

let tiny_mqa () =
  Falcon.config ~vocab_size:64 ~hidden_size:24 ~ffn_hidden_size:40
    ~num_hidden_layers:2 ~num_attention_heads:4 ~max_position_embeddings:32 ()

let tiny_gqa () =
  Falcon.config ~vocab_size:64 ~hidden_size:24 ~ffn_hidden_size:40
    ~num_hidden_layers:2 ~num_attention_heads:4 ~num_key_value_heads:2
    ~max_position_embeddings:32 ~new_decoder_architecture:true ()

let input values =
  Array.map Int32.of_int values
  |> Nx.create Nx.int32 [| 1; Array.length values |]

let max_difference expected actual =
  Nx.sub expected actual |> Nx.abs |> Nx.max |> Nx.item []

let test_attention_variants () =
  equal ~msg:"MQA KV heads" int 1 (tiny_mqa ()).num_key_value_heads;
  equal ~msg:"GQA KV heads" int 2 (tiny_gqa ()).num_key_value_heads;
  Nx.Rng.run ~seed:89 @@ fun () ->
  let model = Falcon.for_causal_lm (tiny_gqa ()) () in
  let vars = Layer.init model ~dtype in
  let paths = Layer.params vars |> Ptree.flatten_with_paths |> List.map fst in
  is_true ~msg:"separate attention norm" (List.mem "h.0.ln_attn.gamma" paths);
  is_true ~msg:"separate MLP norm" (List.mem "h.0.ln_mlp.gamma" paths);
  let logits, _ =
    Layer.apply model vars ~training:false (input [| 1; 2; 3 |])
  in
  equal ~msg:"GQA logit shape" (list int) [ 1; 3; 64 ]
    (Array.to_list (Nx.shape logits))

let check_cached cfg seed =
  Nx.Rng.run ~seed @@ fun () ->
  let model = Falcon.for_causal_lm cfg () in
  let vars = Layer.init model ~dtype in
  let ids = input [| 1; 2; 3; 4 |] in
  let expected, _ = Layer.apply model vars ~training:false ids in
  let cache = Falcon.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
  let rec decode position cache outputs =
    if position = 4 then List.rev outputs
    else
      let token = Nx.slice [ Nx.A; Nx.R (position, position + 1) ] ids in
      let logits, cache = Falcon.decode_step cfg vars cache token in
      decode (position + 1) cache (logits :: outputs)
  in
  Nx.concatenate ~axis:1 (decode 0 cache []) |> max_difference expected

let test_cached_decode () =
  let mqa_difference = check_cached (tiny_mqa ()) 97 in
  let gqa_difference = check_cached (tiny_gqa ()) 101 in
  is_true
    ~msg:(Printf.sprintf "cached Falcon MQA max error is %g" mqa_difference)
    (mqa_difference < 1e-5);
  is_true
    ~msg:(Printf.sprintf "cached Falcon GQA max error is %g" gqa_difference)
    (gqa_difference < 1e-5)

let test_cached_pjrt_cuda () =
  if Rune.Backend.available Pjrt_cuda then
    try
      Nx.Rng.run ~seed:103 @@ fun () ->
      let cfg = tiny_mqa () in
      let model = Falcon.for_causal_lm cfg () in
      let vars = Layer.init model ~dtype in
      let ids = input [| 1; 2; 3 |] in
      let expected, _ = Layer.apply model vars ~training:false ids in
      let runner = Falcon.Pjrt.compile cfg vars in
      let cache = Falcon.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
      let prompt = Nx.slice [ Nx.A; Nx.R (0, 2) ] ids in
      let prompt_logits, cache = Falcon.Pjrt.prefill runner cache prompt in
      let token = Nx.slice [ Nx.A; Nx.R (2, 3) ] ids in
      let token_logits, _ = Falcon.Pjrt.decode_step runner cache token in
      let actual = Nx.concatenate ~axis:1 [ prompt_logits; token_logits ] in
      let difference = max_difference expected actual in
      is_true
        ~msg:(Printf.sprintf "PJRT cached Falcon max error is %g" difference)
        (difference < 5e-4)
    with Rune_pjrt.Error.Error error ->
      failwith ("Falcon cached PJRT CUDA: " ^ Rune_pjrt.Error.to_string error)

let () =
  run "Hugr.Falcon"
    [
      group "model"
        [
          test "attention variants" test_attention_variants;
          test "cached decode" test_cached_decode;
          test "cached PJRT CUDA" test_cached_pjrt_cuda;
        ];
    ]
