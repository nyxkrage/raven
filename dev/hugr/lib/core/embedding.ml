(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let token ~vocab_size ~hidden_size ?(scale = false) ?weight_init () =
  if vocab_size <= 0 then
    invalid_argf "Embedding.token: vocab_size must be positive, got %d"
      vocab_size;
  if hidden_size <= 0 then
    invalid_argf "Embedding.token: hidden_size must be positive, got %d"
      hidden_size;
  let weight_init =
    Option.value weight_init ~default:(Kaun.Init.normal ~stddev:0.02 ())
  in
  {
    Kaun.Layer.init =
      (fun ~dtype ->
        Kaun.Layer.make_vars
          ~params:
            (Kaun.Ptree.dict
               [
                 ( "weight",
                   Kaun.Ptree.tensor
                     (weight_init.f [| vocab_size; hidden_size |] dtype) );
               ])
          ~state:Kaun.Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx input_ids ->
        ignore (training, ctx);
        let fields =
          Kaun.Ptree.Dict.fields_exn ~ctx:"Embedding.token.params" params
        in
        let weight =
          Kaun.Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype
        in
        let input_ids = Nx.cast Nx.int32 input_ids in
        (Kaun.Fn.embedding ~scale ~embedding:weight input_ids, state));
  }
