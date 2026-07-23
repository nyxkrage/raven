(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let linear ~in_features ~out_features ?(bias = false) ?weight_init () =
  if in_features <= 0 then
    invalid_argf "Projection.linear: in_features must be positive, got %d"
      in_features;
  if out_features <= 0 then
    invalid_argf "Projection.linear: out_features must be positive, got %d"
      out_features;
  let weight_init =
    Option.value weight_init ~default:(Kaun.Init.glorot_uniform ())
  in
  {
    Kaun.Layer.init =
      (fun ~dtype ->
        let fields =
          [
            ( "weight",
              Kaun.Ptree.tensor
                (weight_init.f [| in_features; out_features |] dtype) );
          ]
        in
        let fields =
          if bias then
            fields
            @ [
                ("bias", Kaun.Ptree.tensor (Nx.zeros dtype [| out_features |]));
              ]
          else fields
        in
        Kaun.Layer.make_vars ~params:(Kaun.Ptree.dict fields)
          ~state:Kaun.Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (training, ctx);
        let x =
          Layer_util.require_same_float_dtype ~ctx:"Projection.linear" dtype x
        in
        let fields =
          Kaun.Ptree.Dict.fields_exn ~ctx:"Projection.linear.params" params
        in
        let weight =
          Kaun.Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype
        in
        let y = Nx.matmul x weight in
        let y =
          match Kaun.Ptree.Dict.find "bias" fields with
          | None -> y
          | Some bias ->
              let bias =
                Kaun.Ptree.as_tensor_exn ~ctx:"Projection.linear.params.bias"
                  bias
                |> Kaun.Ptree.Tensor.to_typed_exn dtype
              in
              Nx.add y bias
        in
        (y, state));
  }
