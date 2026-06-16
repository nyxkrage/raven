(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let causal_self_attention ~embed_dim ~num_heads ?(dropout = 0.0) () =
  if embed_dim mod num_heads <> 0 then
    invalid_argf
      "Gpt2.Attention.causal_self_attention: embed_dim (%d) not divisible by \
       num_heads (%d)"
      embed_dim num_heads;
  if dropout < 0.0 || dropout >= 1.0 then
    invalid_argf
      "Gpt2.Attention.causal_self_attention: expected 0.0 <= dropout < 1.0, \
       got %g"
      dropout;
  let head_dim = embed_dim / num_heads in
  let weight_init = Kaun.Init.normal ~stddev:0.02 () in
  {
    Kaun.Layer.init =
      (fun ~dtype ->
        Kaun.Layer.make_vars
          ~params:
            (Kaun.Ptree.dict
               [
                 ( "qkv_weight",
                   Kaun.Ptree.tensor
                     (weight_init.f [| embed_dim; 3 * embed_dim |] dtype) );
                 ( "qkv_bias",
                   Kaun.Ptree.tensor (Nx.zeros dtype [| 3 * embed_dim |]) );
                 ( "o_weight",
                   Kaun.Ptree.tensor
                     (weight_init.f [| embed_dim; embed_dim |] dtype) );
                 ("o_bias", Kaun.Ptree.tensor (Nx.zeros dtype [| embed_dim |]));
               ])
          ~state:Kaun.Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore ctx;
        let x =
          Layer_util.require_same_float_dtype
            ~ctx:"Gpt2.Attention.causal_self_attention" dtype x
        in
        let shape = Nx.shape x in
        let batch = shape.(0) in
        let seq = shape.(1) in
        let fields =
          Layer_util.fields ~ctx:"Gpt2.Attention.causal_self_attention.params"
            params
        in
        let qkv_w = Layer_util.get fields ~name:"qkv_weight" dtype in
        let qkv_b = Layer_util.get fields ~name:"qkv_bias" dtype in
        let qkv = Nx.add (Nx.matmul x qkv_w) qkv_b in
        let qkv_parts = Nx.split ~axis:(-1) 3 qkv in
        let q = List.nth qkv_parts 0 in
        let k = List.nth qkv_parts 1 in
        let v = List.nth qkv_parts 2 in
        let split_heads t =
          Nx.reshape [| batch; seq; num_heads; head_dim |] t
          |> Nx.transpose ~axes:[ 0; 2; 1; 3 ]
        in
        let q = split_heads q in
        let k = split_heads k in
        let v = split_heads v in
        let dropout_rate =
          if training && dropout > 0.0 then Some dropout else None
        in
        let attn =
          Kaun.Fn.dot_product_attention ~is_causal:true ?dropout_rate q k v
        in
        let merged =
          Nx.transpose attn ~axes:[ 0; 2; 1; 3 ]
          |> Nx.contiguous
          |> Nx.reshape [| batch; seq; embed_dim |]
        in
        let o_w = Layer_util.get fields ~name:"o_weight" dtype in
        let o_b = Layer_util.get fields ~name:"o_bias" dtype in
        (Nx.add (Nx.matmul merged o_w) o_b, state));
  }
