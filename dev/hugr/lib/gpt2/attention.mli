(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPT-2 attention layers. *)

val causal_self_attention :
  embed_dim:int ->
  num_heads:int ->
  ?dropout:float ->
  unit ->
  (float, float) Kaun.Layer.t
(** [causal_self_attention ~embed_dim ~num_heads ()] is GPT-2-style causal
    self-attention with a combined QKV projection. *)
