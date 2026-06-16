(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPT-2 feed-forward layers. *)

val mlp : embed_dim:int -> hidden_dim:int -> unit -> (float, float) Kaun.Layer.t
(** [mlp ~embed_dim ~hidden_dim ()] is GPT-2's GELU MLP. *)
