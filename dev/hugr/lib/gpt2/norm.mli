(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPT-2 normalization layers. *)

val layer_norm : dim:int -> ?eps:float -> unit -> (float, float) Kaun.Layer.t
(** [layer_norm ~dim ?eps ()] is affine layer normalization. *)
