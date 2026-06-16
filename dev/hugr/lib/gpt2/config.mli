(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPT-2 model configuration. *)

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
(** The type for GPT-2 configurations. *)

val make :
  vocab_size:int ->
  n_embd:int ->
  n_layer:int ->
  n_head:int ->
  ?n_positions:int ->
  ?n_inner:int ->
  ?resid_pdrop:float ->
  ?embd_pdrop:float ->
  ?attn_pdrop:float ->
  ?layer_norm_eps:float ->
  unit ->
  t
(** [make ~vocab_size ~n_embd ~n_layer ~n_head ()] is a GPT-2 configuration.

    [n_positions] defaults to [1024]. [n_inner] defaults to [4 * n_embd].
    Dropout rates default to [0.1]. [layer_norm_eps] defaults to [1e-5].

    Raises [Invalid_argument] if [n_embd] is not divisible by [n_head]. *)
