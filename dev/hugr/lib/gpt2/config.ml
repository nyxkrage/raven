(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = {
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

let make ~vocab_size ~n_embd ~n_layer ~n_head ?(n_positions = 1024)
    ?(n_inner = 4 * n_embd) ?(resid_pdrop = 0.1) ?(embd_pdrop = 0.1)
    ?(attn_pdrop = 0.1) ?(layer_norm_eps = 1e-5) () =
  if n_embd mod n_head <> 0 then
    invalid_argf "Gpt2.Config.make: n_embd (%d) not divisible by n_head (%d)"
      n_embd n_head;
  if n_layer < 0 then
    invalid_argf "Gpt2.Config.make: n_layer must be non-negative, got %d"
      n_layer;
  {
    vocab_size;
    n_positions;
    n_embd;
    n_layer;
    n_head;
    n_inner;
    resid_pdrop;
    embd_pdrop;
    attn_pdrop;
    layer_norm_eps;
  }
