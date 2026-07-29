(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let token_position ~vocab_size ~max_positions ~embed_dim ?(dropout = 0.0) () =
  if dropout < 0.0 || dropout >= 1.0 then
    invalid_argf
      "Gpt2.Embedding.token_position: expected 0.0 <= dropout < 1.0, got %g"
      dropout;
  let weight_init = Kaun.Init.normal ~stddev:0.02 () in
  {
    Kaun.Layer.init =
      (fun ~dtype ->
        Kaun.Layer.make_vars
          ~params:
            (Kaun.Ptree.dict
               [
                 ( "wte",
                   Kaun.Ptree.tensor
                     (weight_init.f [| vocab_size; embed_dim |] dtype) );
                 ( "wpe",
                   Kaun.Ptree.tensor
                     (weight_init.f [| max_positions; embed_dim |] dtype) );
               ])
          ~state:Kaun.Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx input_ids ->
        ignore ctx;
        let input_ids = Nx.cast Nx.int32 input_ids in
        let shape = Nx.shape input_ids in
        let batch = shape.(0) in
        let seq = shape.(1) in
        if seq > max_positions then
          invalid_argf
            "Gpt2.Embedding.token_position: seq_len=%d exceeds max_positions=%d"
            seq max_positions;
        let fields =
          Layer_util.fields ~ctx:"Gpt2.Embedding.token_position.params" params
        in
        let wte = Layer_util.get fields ~name:"wte" dtype in
        let wpe = Layer_util.get fields ~name:"wpe" dtype in
        let position_ids =
          Nx.arange_f Nx.float32 0.0 (float_of_int seq) 1.0
          |> Nx.cast Nx.int32
          |> Nx.reshape [| 1; seq |]
          |> Nx.broadcast_to [| batch; seq |]
          |> Nx.contiguous
        in
        let tok = Kaun.Fn.embedding ~scale:false ~embedding:wte input_ids in
        let pos = Kaun.Fn.embedding ~scale:false ~embedding:wpe position_ids in
        let x = Nx.add tok pos in
        let x =
          if training && dropout > 0.0 then Kaun.Fn.dropout ~rate:dropout x
          else x
        in
        (x, state));
  }
