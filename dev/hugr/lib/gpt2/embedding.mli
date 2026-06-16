(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPT-2 embedding layers. *)

val token_position :
  vocab_size:int ->
  max_positions:int ->
  embed_dim:int ->
  ?dropout:float ->
  unit ->
  (int32, float) Kaun.Layer.t
(** [token_position ~vocab_size ~max_positions ~embed_dim ()] adds token and
    learned position embeddings. *)
