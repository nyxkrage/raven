(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** LLaMA decoder and causal language model. *)

module Config = Config
(** LLaMA model configuration. *)

include module type of Llama with module Internal := Llama.Internal

module Mistral = Mistral
(** Mistral decoder and causal language model. *)

module Gemma2 : module type of Gemma2 with module Internal := Gemma2.Internal
(** Gemma 2 decoder and causal language model. *)

module Gemma3 = Gemma3
(** Gemma 3 text decoder and causal language model. *)

module Falcon = Falcon
(** Falcon parallel decoder and causal language model. *)
