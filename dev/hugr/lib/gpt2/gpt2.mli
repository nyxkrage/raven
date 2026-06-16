(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Primary GPT-2 modeling code. *)

open Kaun

type config = Config.t = {
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

val config :
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
  config
(** Alias for {!Config.make}. *)

val decoder_block : config -> unit -> (float, float) Layer.t
(** [decoder_block cfg ()] is one GPT-2 pre-norm decoder block. *)

val decoder : config -> unit -> (int32, float) Layer.t
(** [decoder cfg ()] is the GPT-2 transformer decoder. *)

val for_causal_lm : config -> unit -> (int32, float) Layer.t
(** [for_causal_lm cfg ()] is decoder + tied LM head. *)

val from_pretrained : ?model_id:string -> unit -> config * Ptree.t
(** [from_pretrained ?model_id ()] downloads [model_id] from HuggingFace and
    returns [(cfg, decoder_params)]. [model_id] defaults to ["gpt2"]. *)
