(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Layer = Kaun.Layer
module Llama = Hugr.Llama
module Ptree = Kaun.Ptree

let dtype = Nx.float32

let tiny ?(tie_word_embeddings = false) () =
  Llama.config ~vocab_size:64 ~hidden_size:32 ~intermediate_size:48
    ~num_hidden_layers:2 ~num_attention_heads:4 ~num_key_value_heads:2
    ~max_position_embeddings:32 ~tie_word_embeddings ()

let input values =
  Array.map Int32.of_int values
  |> Nx.create Nx.int32 [| 1; Array.length values |]

let path_shape params path =
  match List.assoc_opt path (Ptree.flatten_with_paths params) with
  | Some tensor -> Array.to_list (Ptree.Tensor.shape tensor)
  | None -> failwith (Printf.sprintf "missing parameter path %S" path)

let test_config_defaults () =
  let cfg =
    Llama.config ~vocab_size:128 ~hidden_size:64 ~intermediate_size:96
      ~num_hidden_layers:3 ~num_attention_heads:8 ()
  in
  equal ~msg:"KV heads default to query heads" int 8 cfg.num_key_value_heads;
  equal ~msg:"default maximum positions" int 2048 cfg.max_position_embeddings;
  equal ~msg:"default embeddings are untied" bool false cfg.tie_word_embeddings

let test_config_validation () =
  raises_match
    (fun error -> match error with Invalid_argument _ -> true | _ -> false)
    (fun () ->
      ignore
        (Llama.config ~vocab_size:64 ~hidden_size:30 ~intermediate_size:48
           ~num_hidden_layers:2 ~num_attention_heads:4 ()));
  raises_match
    (fun error -> match error with Invalid_argument _ -> true | _ -> false)
    (fun () ->
      ignore
        (Llama.config ~vocab_size:64 ~hidden_size:32 ~intermediate_size:48
           ~num_hidden_layers:2 ~num_attention_heads:4 ~num_key_value_heads:3 ()))

let test_parameter_shapes () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let model = Llama.for_causal_lm (tiny ()) () in
  let vars = Layer.init model ~dtype in
  let params = Layer.params vars in
  equal ~msg:"token embedding" (list int) [ 64; 32 ]
    (path_shape params "embed_tokens.weight");
  equal ~msg:"query projection" (list int) [ 32; 32 ]
    (path_shape params "layers.0.self_attn.q_proj.weight");
  equal ~msg:"key projection" (list int) [ 32; 16 ]
    (path_shape params "layers.0.self_attn.k_proj.weight");
  equal ~msg:"SwiGLU gate" (list int) [ 32; 48 ]
    (path_shape params "layers.0.mlp.gate_proj.weight");
  equal ~msg:"LM head" (list int) [ 32; 64 ]
    (path_shape params "lm_head.weight")

let test_decoder_shape () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let cfg = tiny () in
  let model = Llama.decoder cfg () in
  let vars = Layer.init model ~dtype in
  let ids =
    Nx.create Nx.int32 [| 2; 4 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l |]
  in
  let hidden, _ = Layer.apply model vars ~training:false ids in
  equal ~msg:"decoder output shape" (list int) [ 2; 4; 32 ]
    (Array.to_list (Nx.shape hidden))

let test_causal_lm_shape_and_values () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let model = Llama.for_causal_lm (tiny ()) () in
  let vars = Layer.init model ~dtype in
  let logits, _ =
    Layer.apply model vars ~training:false (input [| 1; 2; 3; 4 |])
  in
  equal ~msg:"logit shape" (list int) [ 1; 4; 64 ]
    (Array.to_list (Nx.shape logits));
  is_false ~msg:"logits are finite" (Nx.item [] (Nx.any (Nx.isnan logits)))

let test_causal_mask () =
  Nx.Rng.run ~seed:7 @@ fun () ->
  let model = Llama.for_causal_lm (tiny ()) () in
  let vars = Layer.init model ~dtype in
  let first, _ = Layer.apply model vars ~training:false (input [| 1; 2; 3 |]) in
  let second, _ =
    Layer.apply model vars ~training:false (input [| 1; 2; 4 |])
  in
  let prefix tensor = Nx.slice [ Nx.A; Nx.R (0, 2); Nx.A ] tensor in
  let difference =
    Nx.sub (prefix first) (prefix second) |> Nx.abs |> Nx.sum |> Nx.item []
  in
  is_true ~msg:"future tokens do not affect earlier logits" (difference < 1e-5)

let test_padding_mask () =
  Nx.Rng.run ~seed:11 @@ fun () ->
  let model = Llama.for_causal_lm (tiny ()) () in
  let vars = Layer.init model ~dtype in
  let attention_mask = Nx.create Nx.int32 [| 1; 3 |] [| 1l; 0l; 1l |] in
  let ctx =
    Kaun.Context.empty
    |> Kaun.Context.set ~name:Llama.attention_mask_key (Ptree.P attention_mask)
  in
  let first, _ =
    Layer.apply model vars ~training:false ~ctx (input [| 1; 2; 3 |])
  in
  let second, _ =
    Layer.apply model vars ~training:false ~ctx (input [| 1; 9; 3 |])
  in
  let last tensor = Nx.slice [ Nx.A; Nx.I 2; Nx.A ] tensor in
  let difference =
    Nx.sub (last first) (last second) |> Nx.abs |> Nx.sum |> Nx.item []
  in
  is_true ~msg:"masked keys do not affect visible tokens" (difference < 1e-5)

let test_explicit_position_ids () =
  Nx.Rng.run ~seed:13 @@ fun () ->
  let model = Llama.for_causal_lm (tiny ()) () in
  let vars = Layer.init model ~dtype in
  let ids = input [| 1; 2; 3 |] in
  let ordinary, _ = Layer.apply model vars ~training:false ids in
  let position_ids = Nx.create Nx.int32 [| 1; 3 |] [| 0l; 2l; 4l |] in
  let ctx =
    Kaun.Context.empty
    |> Kaun.Context.set ~name:Llama.position_ids_key (Ptree.P position_ids)
  in
  let spaced, _ = Layer.apply model vars ~training:false ~ctx ids in
  let difference = Nx.sub ordinary spaced |> Nx.abs |> Nx.sum |> Nx.item [] in
  is_true ~msg:"explicit positions change rotary attention" (difference > 1e-5)

let test_tied_embeddings () =
  Nx.Rng.run ~seed:17 @@ fun () ->
  let model = Llama.for_causal_lm (tiny ~tie_word_embeddings:true ()) () in
  let vars = Layer.init model ~dtype in
  let paths = Ptree.flatten_with_paths (Layer.params vars) |> List.map fst in
  is_false ~msg:"tied model has no separate LM head"
    (List.exists
       (fun path -> String.length path >= 7 && String.sub path 0 7 = "lm_head")
       paths);
  let logits, _ =
    Layer.apply model vars ~training:false (input [| 1; 2; 3 |])
  in
  equal ~msg:"tied model logit shape" (list int) [ 1; 3; 64 ]
    (Array.to_list (Nx.shape logits))

let test_gradients () =
  Nx.Rng.run ~seed:19 @@ fun () ->
  let model = Llama.for_causal_lm (tiny ()) () in
  let vars = Layer.init model ~dtype in
  let state = Layer.state vars in
  let ids = input [| 1; 2; 3 |] in
  let loss params =
    model.Layer.apply ~params ~state ~dtype ~training:false ids |> fst |> Nx.sum
  in
  let gradients = Kaun.Grad.grad loss (Layer.params vars) in
  let total = ref 0.0 in
  let finite = ref true in
  Ptree.iter
    (fun tensor ->
      let gradient = Ptree.Tensor.to_typed_exn dtype tensor in
      if Nx.item [] (Nx.any (Nx.isnan gradient)) then finite := false;
      total := !total +. Nx.item [] (Nx.sum (Nx.abs gradient)))
    gradients;
  is_true ~msg:"gradients are finite" !finite;
  is_true ~msg:"gradient tree contains nonzero values" (!total > 0.0)

let max_difference expected actual =
  Nx.sub expected actual |> Nx.abs |> Nx.max |> Nx.item []

let test_cached_prefill () =
  Nx.Rng.run ~seed:23 @@ fun () ->
  let cfg = tiny () in
  let model = Llama.for_causal_lm cfg () in
  let vars = Layer.init model ~dtype in
  let ids = input [| 1; 2; 3; 4 |] in
  let expected, _ = Layer.apply model vars ~training:false ids in
  let cache = Llama.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
  let actual, cache = Llama.prefill cfg vars cache ids in
  let difference = max_difference expected actual in
  equal ~msg:"cache length" int 4 (Llama.Cache.length cache);
  is_true
    ~msg:(Printf.sprintf "cached prefill max error is %g" difference)
    (difference < 1e-5)

let test_cached_decode_steps () =
  Nx.Rng.run ~seed:29 @@ fun () ->
  let cfg = tiny () in
  let model = Llama.for_causal_lm cfg () in
  let vars = Layer.init model ~dtype in
  let ids = input [| 1; 2; 3; 4 |] in
  let expected, _ = Layer.apply model vars ~training:false ids in
  let cache = Llama.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
  let rec decode position cache outputs =
    if position = 4 then (List.rev outputs, cache)
    else
      let token = Nx.slice [ Nx.A; Nx.R (position, position + 1) ] ids in
      let logits, cache = Llama.decode_step cfg vars cache token in
      decode (position + 1) cache (logits :: outputs)
  in
  let outputs, cache = decode 0 cache [] in
  let actual = Nx.concatenate ~axis:1 outputs in
  let difference = max_difference expected actual in
  equal ~msg:"decode cache length" int 4 (Llama.Cache.length cache);
  is_true
    ~msg:(Printf.sprintf "token decode max error is %g" difference)
    (difference < 1e-5)

let test_cached_padding_mask () =
  Nx.Rng.run ~seed:31 @@ fun () ->
  let cfg = tiny () in
  let model = Llama.for_causal_lm cfg () in
  let vars = Layer.init model ~dtype in
  let ids = input [| 1; 2; 3 |] in
  let attention_mask = Nx.create Nx.bool [| 1; 3 |] [| true; false; true |] in
  let ctx =
    Kaun.Context.empty
    |> Kaun.Context.set ~name:Llama.attention_mask_key (Ptree.P attention_mask)
  in
  let expected, _ = Layer.apply model vars ~training:false ~ctx ids in
  let cache = Llama.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
  let actual, _ = Llama.prefill cfg vars cache ~attention_mask ids in
  let difference = max_difference expected actual in
  is_true
    ~msg:(Printf.sprintf "cached masked prefill max error is %g" difference)
    (difference < 1e-5)

let test_pjrt_cuda () =
  if Rune.Backend.available Pjrt_cuda then
    try
      Nx.Rng.run ~seed:23 @@ fun () ->
      let model = Llama.for_causal_lm (tiny ()) () in
      let vars = Layer.init model ~dtype in
      let params = Layer.params vars in
      let state = Layer.state vars in
      let forward input_ids =
        model.Layer.apply ~params ~state ~dtype ~training:false input_ids |> fst
      in
      let compiled = Rune.jit ~device:(Rune.Backend.device Pjrt_cuda) forward in
      let ids = input [| 1; 2; 3 |] in
      let expected = forward ids in
      ignore (compiled ids);
      ignore (compiled ids);
      let actual = compiled ids in
      let difference =
        Nx.sub expected actual |> Nx.abs |> Nx.max |> Nx.item []
      in
      is_true
        ~msg:(Printf.sprintf "PJRT CUDA max error is %g" difference)
        (difference < 5e-4)
    with Rune_pjrt.Error.Error error ->
      failwith ("LLaMA PJRT CUDA: " ^ Rune_pjrt.Error.to_string error)

let test_cached_pjrt_cuda () =
  if Rune.Backend.available Pjrt_cuda then
    try
      Nx.Rng.run ~seed:37 @@ fun () ->
      let cfg = tiny () in
      let model = Llama.for_causal_lm cfg () in
      let vars = Layer.init model ~dtype in
      let ids = input [| 1; 2; 3; 4 |] in
      let expected, _ = Layer.apply model vars ~training:false ids in
      let runner = Llama.Pjrt.compile cfg vars in
      let cache = Llama.Cache.create cfg ~batch_size:1 ~max_length:8 ~dtype in
      let prompt = Nx.slice [ Nx.A; Nx.R (0, 2) ] ids in
      let prompt_logits, cache = Llama.Pjrt.prefill runner cache prompt in
      let third = Nx.slice [ Nx.A; Nx.R (2, 3) ] ids in
      let third_logits, cache = Llama.Pjrt.decode_step runner cache third in
      let fourth = Nx.slice [ Nx.A; Nx.R (3, 4) ] ids in
      let fourth_logits, cache = Llama.Pjrt.decode_step runner cache fourth in
      let actual =
        Nx.concatenate ~axis:1 [ prompt_logits; third_logits; fourth_logits ]
      in
      let difference = max_difference expected actual in
      equal ~msg:"PJRT cache length" int 4 (Llama.Cache.length cache);
      is_true
        ~msg:(Printf.sprintf "PJRT cached decode max error is %g" difference)
        (difference < 5e-4)
    with Rune_pjrt.Error.Error error ->
      failwith ("LLaMA cached PJRT CUDA: " ^ Rune_pjrt.Error.to_string error)

let () =
  run "Hugr.Llama"
    [
      group "config"
        [
          test "defaults" test_config_defaults;
          test "validation" test_config_validation;
        ];
      group "model"
        [
          test "parameter shapes" test_parameter_shapes;
          test "decoder shape" test_decoder_shape;
          test "causal LM shape and values" test_causal_lm_shape_and_values;
          test "causal mask" test_causal_mask;
          test "padding mask" test_padding_mask;
          test "explicit position IDs" test_explicit_position_ids;
          test "tied embeddings" test_tied_embeddings;
          test "gradients" test_gradients;
          test "cached prefill" test_cached_prefill;
          test "cached decode steps" test_cached_decode_steps;
          test "cached padding mask" test_cached_padding_mask;
          test "PJRT CUDA" test_pjrt_cuda;
          test "cached PJRT CUDA" test_cached_pjrt_cuda;
        ];
    ]
