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
