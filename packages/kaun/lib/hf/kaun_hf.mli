(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HuggingFace Hub integration.

    Download pretrained model weights and configuration files from the
    {{:https://huggingface.co}HuggingFace Hub}. Supports single-file and sharded
    SafeTensors checkpoints, caching, authentication, and offline mode.

    {[
      let config =
        Kaun_hf.load_config ~model_id:"bert-base-uncased" ()
      in
      let weights =
        Kaun_hf.load_weights ~model_id:"bert-base-uncased" ()
      in
      (* weights : (string * Kaun.Ptree.tensor) list *)
    ]} *)

(** {1:types Types} *)

(** The type for repository revisions. *)
type revision =
  | Main  (** The default branch. *)
  | Rev of string  (** A tag, branch name, or commit hash. *)

(** {1:downloading Downloading} *)

val download_file :
  ?token:string ->
  ?cache_dir:string ->
  ?offline:bool ->
  ?revision:revision ->
  model_id:string ->
  filename:string ->
  unit ->
  string
(** [download_file ~model_id ~filename ()] is the local path to [filename] from
    the repository [model_id].

    The file is downloaded to the cache on first access and served from there on
    subsequent calls.

    [token] is a HuggingFace API token for private repositories. Defaults to the
    value of [HF_TOKEN].

    [cache_dir] defaults to the regular HuggingFace Hub cache: [HF_HUB_CACHE],
    then [HUGGINGFACE_HUB_CACHE], then [{HF_HOME}/hub], then
    [{XDG_CACHE_HOME}/huggingface/hub].

    Files are stored using the Hub cache layout
    [models--org--repo/{refs,snapshots,blobs}] so Kaun shares downloads with
    [huggingface_hub] and Transformers.

    [offline] defaults to [true] when [HF_HUB_OFFLINE] or [TRANSFORMERS_OFFLINE]
    is set, and [false] otherwise. When [true], only cached files are returned.

    [revision] defaults to {!Main}.

    When the [xet] package is available, Xet is tried first so Xet-backed
    repositories on the Hub download through the native Xet path. If Xet is
    unavailable or the file is not served via Xet, the function falls back to a
    regular HTTP download into the Hub cache layout.

    Raises [Failure] if the download fails or the file is not cached in offline
    mode. *)

(** {1:loading Loading} *)

val load_config :
  ?token:string ->
  ?cache_dir:string ->
  ?offline:bool ->
  ?revision:revision ->
  model_id:string ->
  unit ->
  Jsont.json
(** [load_config ~model_id ()] is the parsed [config.json] from [model_id].

    Parameters are the same as {!download_file}.

    Raises [Failure] on download or JSON parse errors. *)

val load_weights :
  ?token:string ->
  ?cache_dir:string ->
  ?offline:bool ->
  ?revision:revision ->
  model_id:string ->
  unit ->
  (string * Kaun.Ptree.tensor) list
(** [load_weights ~model_id ()] is the list of [(name, tensor)] pairs from
    [model_id]'s SafeTensors checkpoint.

    Handles sharded checkpoints transparently: when
    [model.safetensors.index.json] is present, all referenced shards are
    downloaded and merged. Falls back to [model.safetensors] when no index
    exists.

    Tensor names are the raw keys from the SafeTensors file (e.g.
    ["bert.encoder.layer.0.attention.self.query.weight"]). Model code is
    responsible for mapping these to its own parameter structure.

    Parameters are the same as {!download_file}.

    Raises [Failure] if no SafeTensors files are found, or on download/parse
    errors. *)

(** {1:cache Cache management} *)

val clear_cache : ?cache_dir:string -> ?model_id:string -> unit -> unit
(** [clear_cache ()] removes the selected HuggingFace Hub cache directory.

    Without [model_id], this removes the whole [cache_dir], which defaults to
    the shared HuggingFace Hub cache. When [model_id] is given, only that
    model's cache folder is removed. *)
