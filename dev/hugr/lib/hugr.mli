(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Full model architectures for Raven.

    Hugr is the experimental home for complete model definitions: decoder-only
    language models, multimodal stacks, mixture-of-experts architectures, and
    pretrained weight loaders built on Kaun, Rune, and Nx.

    Architecture modules expose Kaun layers so they compose with the rest of the
    Raven training and inference stack. *)

module Gpt2 = Hugr_gpt2
(** GPT-2 decoder and language model head. *)

module Llama = Hugr_llama
(** LLaMA decoder and causal language model head. *)

module Mistral = Hugr_llama.Mistral
(** Mistral decoder and causal language model head. *)

module Gemma2 = Hugr_llama.Gemma2
(** Gemma 2 decoder and causal language model head. *)

module Gemma3 = Hugr_llama.Gemma3
(** Gemma 3 text decoder and causal language model head. *)

module Falcon = Hugr_llama.Falcon
(** Falcon parallel decoder and causal language model head. *)
