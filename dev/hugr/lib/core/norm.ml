(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let gemma_rms ~dim ~eps () =
  if dim <= 0 then
    invalid_arg
      (Printf.sprintf "Norm.gemma_rms: dim must be positive, got %d" dim);
  if eps <= 0.0 then
    invalid_arg
      (Printf.sprintf "Norm.gemma_rms: eps must be positive, got %g" eps);
  {
    Kaun.Layer.init =
      (fun ~dtype ->
        Kaun.Layer.make_vars
          ~params:
            (Kaun.Ptree.dict
               [ ("weight", Kaun.Ptree.tensor (Nx.zeros dtype [| dim |])) ])
          ~state:Kaun.Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (training, ctx);
        let x =
          Layer_util.require_same_float_dtype ~ctx:"Norm.gemma_rms" dtype x
        in
        let fields =
          Kaun.Ptree.Dict.fields_exn ~ctx:"Norm.gemma_rms.params" params
        in
        let weight =
          Kaun.Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype
        in
        let scale = Nx.add weight (Nx.scalar dtype 1.0) in
        (Kaun.Fn.rms_norm ~gamma:scale ~epsilon:eps x, state));
  }
