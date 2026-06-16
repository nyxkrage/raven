(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let mlp ~embed_dim ~hidden_dim () =
  let weight_init = Kaun.Init.normal ~stddev:0.02 () in
  {
    Kaun.Layer.init =
      (fun ~dtype ->
        Kaun.Layer.make_vars
          ~params:
            (Kaun.Ptree.dict
               [
                 ( "up_weight",
                   Kaun.Ptree.tensor
                     (weight_init.f [| embed_dim; hidden_dim |] dtype) );
                 ( "up_bias",
                   Kaun.Ptree.tensor (Nx.zeros dtype [| hidden_dim |]) );
                 ( "down_weight",
                   Kaun.Ptree.tensor
                     (weight_init.f [| hidden_dim; embed_dim |] dtype) );
                 ( "down_bias",
                   Kaun.Ptree.tensor (Nx.zeros dtype [| embed_dim |]) );
               ])
          ~state:Kaun.Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (training, ctx);
        let x =
          Layer_util.require_same_float_dtype ~ctx:"Gpt2.Feed_forward.mlp" dtype
            x
        in
        let fields = Layer_util.fields ~ctx:"Gpt2.Feed_forward.mlp" params in
        let up_w = Layer_util.get fields ~name:"up_weight" dtype in
        let up_b = Layer_util.get fields ~name:"up_bias" dtype in
        let down_w = Layer_util.get fields ~name:"down_weight" dtype in
        let down_b = Layer_util.get fields ~name:"down_bias" dtype in
        let y =
          Nx.add (Nx.matmul x up_w) up_b |> Kaun.Activation.gelu_approx
        in
        (Nx.add (Nx.matmul y down_w) down_b, state));
  }
