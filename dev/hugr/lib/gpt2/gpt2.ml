(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Gpt2_attention = Attention
open Kaun

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type config = Config.t = {
  vocab_size : int;
  n_positions : int;
  n_embd : int;
  n_layer : int;
  n_head : int;
  n_inner : int;
  resid_pdrop : float;
  embd_pdrop : float;
  attn_pdrop : float;
  layer_norm_eps : float;
}

let config = Config.make

let embeddings cfg =
  Embedding.token_position ~vocab_size:cfg.vocab_size
    ~max_positions:cfg.n_positions ~embed_dim:cfg.n_embd ~dropout:cfg.embd_pdrop
    ()

let final_norm cfg = Norm.layer_norm ~dim:cfg.n_embd ~eps:cfg.layer_norm_eps ()

let decoder_block cfg () =
  if cfg.resid_pdrop < 0.0 || cfg.resid_pdrop >= 1.0 then
    invalid_argf "Gpt2.decoder_block: expected 0.0 <= resid_pdrop < 1.0, got %g"
      cfg.resid_pdrop;
  let ln1 = Norm.layer_norm ~dim:cfg.n_embd ~eps:cfg.layer_norm_eps () in
  let attention =
    Gpt2_attention.causal_self_attention ~embed_dim:cfg.n_embd
      ~num_heads:cfg.n_head ~dropout:cfg.attn_pdrop ()
  in
  let ln2 = Norm.layer_norm ~dim:cfg.n_embd ~eps:cfg.layer_norm_eps () in
  let mlp = Feed_forward.mlp ~embed_dim:cfg.n_embd ~hidden_dim:cfg.n_inner () in
  let names = [ "ln1"; "attention"; "ln2"; "mlp" ] in
  {
    Layer.init =
      (fun ~dtype ->
        Layer_util.init_children dtype
          [
            ("ln1", ln1.Layer.init);
            ("attention", attention.Layer.init);
            ("ln2", ln2.Layer.init);
            ("mlp", mlp.Layer.init);
          ]);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        let x =
          Layer_util.require_same_float_dtype ~ctx:"Gpt2.decoder_block" dtype x
        in
        let x_norm, ln1_state =
          Layer_util.apply_child ~ctx:"Gpt2.decoder_block" ln1 ~name:"ln1"
            ~params ~state ~dtype ~training ?call_ctx:ctx x
        in
        let attn, attn_state =
          Layer_util.apply_child ~ctx:"Gpt2.decoder_block" attention
            ~name:"attention" ~params ~state ~dtype ~training ?call_ctx:ctx
            x_norm
        in
        let attn =
          if training && cfg.resid_pdrop > 0.0 then
            Fn.dropout ~rate:cfg.resid_pdrop attn
          else attn
        in
        let x = Nx.add x attn in
        let x_norm, ln2_state =
          Layer_util.apply_child ~ctx:"Gpt2.decoder_block" ln2 ~name:"ln2"
            ~params ~state ~dtype ~training ?call_ctx:ctx x
        in
        let y, mlp_state =
          Layer_util.apply_child ~ctx:"Gpt2.decoder_block" mlp ~name:"mlp"
            ~params ~state ~dtype ~training ?call_ctx:ctx x_norm
        in
        let y =
          if training && cfg.resid_pdrop > 0.0 then
            Fn.dropout ~rate:cfg.resid_pdrop y
          else y
        in
        let state =
          Layer_util.merge_state ~names
            [ ln1_state; attn_state; ln2_state; mlp_state ]
        in
        (Nx.add x y, state));
  }

let init_decoder_params ~cfg ~dtype =
  let embeddings_vars =
    Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
        (embeddings cfg).Layer.init ~dtype)
  in
  let block_layer = decoder_block cfg () in
  let layer_vars =
    List.init cfg.n_layer (fun _ ->
        Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
            block_layer.Layer.init ~dtype))
  in
  let ln_f_vars =
    Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () ->
        (final_norm cfg).Layer.init ~dtype)
  in
  let params =
    Ptree.dict
      [
        ("embeddings", Layer.params embeddings_vars);
        ("layers", Ptree.list (List.map Layer.params layer_vars));
        ("ln_f", Layer.params ln_f_vars);
      ]
  in
  let state =
    Ptree.dict
      [
        ("embeddings", Layer.state embeddings_vars);
        ("layers", Ptree.list (List.map Layer.state layer_vars));
        ("ln_f", Layer.state ln_f_vars);
      ]
  in
  Layer.make_vars ~params ~state ~dtype

let decode (type l in_elt) ~(cfg : config) ~params ~state
    ~(dtype : (float, l) Nx.dtype) ~training ?ctx
    (input_ids : (int32, in_elt) Nx.t) : (float, l) Nx.t * Ptree.t =
  let root = Layer_util.fields ~ctx:"Gpt2.decode.params" params in
  let state_root = Layer_util.fields ~ctx:"Gpt2.decode.state" state in
  let get_param name = Layer_util.find ~ctx:"Gpt2.decode.params" name root in
  let get_state name =
    Layer_util.find ~ctx:"Gpt2.decode.state" name state_root
  in

  let embeddings_layer = embeddings cfg in
  let x, embeddings_state =
    embeddings_layer.Layer.apply ~params:(get_param "embeddings")
      ~state:(get_state "embeddings") ~dtype ~training ?ctx input_ids
  in

  let block_layer = decoder_block cfg () in
  let layer_params =
    Ptree.List.items_exn ~ctx:"Gpt2.decode.params.layers" (get_param "layers")
  in
  let layer_state =
    Ptree.List.items_exn ~ctx:"Gpt2.decode.state.layers" (get_state "layers")
  in
  if List.length layer_params <> cfg.n_layer then
    invalid_argf "Gpt2.decode: expected %d layer parameter sets, got %d"
      cfg.n_layer (List.length layer_params);
  if List.length layer_state <> cfg.n_layer then
    invalid_argf "Gpt2.decode: expected %d layer states, got %d" cfg.n_layer
      (List.length layer_state);
  let x, layer_states =
    List.fold_left2
      (fun (h, states) params state ->
        let y, state =
          block_layer.Layer.apply ~params ~state ~dtype ~training ?ctx h
        in
        (y, state :: states))
      (x, []) layer_params layer_state
  in
  let layer_states = List.rev layer_states in

  let ln_f = final_norm cfg in
  let y, ln_f_state =
    ln_f.Layer.apply ~params:(get_param "ln_f") ~state:(get_state "ln_f") ~dtype
      ~training ?ctx x
  in
  let state =
    Ptree.dict
      [
        ("embeddings", embeddings_state);
        ("layers", Ptree.list layer_states);
        ("ln_f", ln_f_state);
      ]
  in
  (y, state)

let decoder (cfg : config) () : (int32, float) Layer.t =
  {
    Layer.init = (fun ~dtype -> init_decoder_params ~cfg ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        decode ~cfg ~params ~state ~dtype ~training ?ctx x);
  }

let for_causal_lm (cfg : config) () : (int32, float) Layer.t =
  {
    Layer.init = (fun ~dtype -> init_decoder_params ~cfg ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        let hidden, state =
          decode ~cfg ~params ~state ~dtype ~training ?ctx x
        in
        let root = Layer_util.fields ~ctx:"Gpt2.lm_head.params" params in
        let embeddings =
          Layer_util.find ~ctx:"Gpt2.lm_head.params" "embeddings" root
        in
        let fields =
          Layer_util.fields ~ctx:"Gpt2.lm_head.params.embeddings" embeddings
        in
        let wte = Layer_util.get fields ~name:"wte" dtype in
        let logits = Nx.matmul hidden (Nx.transpose wte ~axes:[ 1; 0 ]) in
        (logits, state));
  }

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

let json_to_int = function
  | Jsont.Number (f, _) -> int_of_float f
  | _ -> failwith "expected int"

let json_to_int_option = function
  | Jsont.Number (f, _) -> Some (int_of_float f)
  | _ -> None

let json_to_float_option = function Jsont.Number (f, _) -> Some f | _ -> None

let parse_config json =
  let n_embd = json |> json_mem "n_embd" |> json_to_int in
  config
    ~vocab_size:(json |> json_mem "vocab_size" |> json_to_int)
    ~n_embd
    ~n_layer:(json |> json_mem "n_layer" |> json_to_int)
    ~n_head:(json |> json_mem "n_head" |> json_to_int)
    ?n_positions:(json |> json_mem "n_positions" |> json_to_int_option)
    ?n_inner:(json |> json_mem "n_inner" |> json_to_int_option)
    ?resid_pdrop:(json |> json_mem "resid_pdrop" |> json_to_float_option)
    ?embd_pdrop:(json |> json_mem "embd_pdrop" |> json_to_float_option)
    ?attn_pdrop:(json |> json_mem "attn_pdrop" |> json_to_float_option)
    ?layer_norm_eps:
      (json |> json_mem "layer_norm_epsilon" |> json_to_float_option)
    ()

let cast_tensor dtype (Ptree.P t) = Ptree.P (Nx.cast dtype t)

let map_hf_weights ~cfg ~dtype hf_weights =
  let tbl = Hashtbl.create (List.length hf_weights) in
  List.iter (fun (name, tensor) -> Hashtbl.add tbl name tensor) hf_weights;
  let hf name =
    match Hashtbl.find_opt tbl name with
    | Some t -> cast_tensor dtype t
    | None -> invalid_argf "from_pretrained: missing HF weight %S" name
  in
  let hf_t name = Ptree.Tensor (hf name) in
  let layer i =
    let p s = Printf.sprintf "h.%d.%s" i s in
    Ptree.dict
      [
        ( "ln1",
          Ptree.dict
            [
              ("gamma", hf_t (p "ln_1.weight")); ("beta", hf_t (p "ln_1.bias"));
            ] );
        ( "attention",
          Ptree.dict
            [
              ("qkv_weight", hf_t (p "attn.c_attn.weight"));
              ("qkv_bias", hf_t (p "attn.c_attn.bias"));
              ("o_weight", hf_t (p "attn.c_proj.weight"));
              ("o_bias", hf_t (p "attn.c_proj.bias"));
            ] );
        ( "ln2",
          Ptree.dict
            [
              ("gamma", hf_t (p "ln_2.weight")); ("beta", hf_t (p "ln_2.bias"));
            ] );
        ( "mlp",
          Ptree.dict
            [
              ("up_weight", hf_t (p "mlp.c_fc.weight"));
              ("up_bias", hf_t (p "mlp.c_fc.bias"));
              ("down_weight", hf_t (p "mlp.c_proj.weight"));
              ("down_bias", hf_t (p "mlp.c_proj.bias"));
            ] );
      ]
  in
  Ptree.dict
    [
      ( "embeddings",
        Ptree.dict [ ("wte", hf_t "wte.weight"); ("wpe", hf_t "wpe.weight") ] );
      ("layers", Ptree.list (List.init cfg.n_layer layer));
      ( "ln_f",
        Ptree.dict [ ("gamma", hf_t "ln_f.weight"); ("beta", hf_t "ln_f.bias") ]
      );
    ]

let from_pretrained ?(model_id = "gpt2") () =
  let json = Kaun_hf.load_config ~model_id () in
  let cfg = parse_config json in
  let hf_weights = Kaun_hf.load_weights ~model_id () in
  let params = map_hf_weights ~cfg ~dtype:Nx.float32 hf_weights in
  (cfg, params)
