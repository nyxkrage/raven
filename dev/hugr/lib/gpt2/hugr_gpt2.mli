(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPT-2 decoder and language model head. *)

module Config = Config
(** GPT-2 model configuration. *)

module Attention = Attention
(** GPT-2 attention layers. *)

module Embedding = Embedding
(** GPT-2 embedding layers. *)

module Feed_forward = Feed_forward
(** GPT-2 feed-forward layers. *)

module Norm = Norm
(** GPT-2 normalization layers. *)

include module type of Gpt2
