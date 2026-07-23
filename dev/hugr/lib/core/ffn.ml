(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type activation = Silu | Gelu | Gelu_approx

let activate = function
  | Silu -> Kaun.Activation.silu
  | Gelu -> Kaun.Activation.gelu
  | Gelu_approx -> Kaun.Activation.gelu_approx

let gated ~hidden_size ~intermediate_size ~activation ?(bias = false)
    ?weight_init () =
  let projection ~in_features ~out_features =
    Projection.linear ~in_features ~out_features ~bias ?weight_init ()
  in
  let gate_proj =
    projection ~in_features:hidden_size ~out_features:intermediate_size
  in
  let up_proj =
    projection ~in_features:hidden_size ~out_features:intermediate_size
  in
  let down_proj =
    projection ~in_features:intermediate_size ~out_features:hidden_size
  in
  let names = [ "gate_proj"; "up_proj"; "down_proj" ] in
  {
    Kaun.Layer.init =
      (fun ~dtype ->
        Layer_util.init_children dtype
          [
            ("gate_proj", gate_proj.Kaun.Layer.init);
            ("up_proj", up_proj.Kaun.Layer.init);
            ("down_proj", down_proj.Kaun.Layer.init);
          ]);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        let x = Layer_util.require_same_float_dtype ~ctx:"Ffn.gated" dtype x in
        let apply projection name input =
          Layer_util.apply_child ~ctx:"Ffn.gated" projection ~name ~params
            ~state ~dtype ~training ?call_ctx:ctx input
        in
        let gate, gate_state = apply gate_proj "gate_proj" x in
        let up, up_state = apply up_proj "up_proj" x in
        let hidden = Nx.mul (activate activation gate) up in
        let output, down_state = apply down_proj "down_proj" hidden in
        let state =
          Layer_util.merge_state ~names [ gate_state; up_state; down_state ]
        in
        (output, state));
  }
