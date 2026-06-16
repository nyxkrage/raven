(** Experimental OCaml bindings for Hugging Face Xet.

    The implementation delegates to the vendored [huggingface/xet-core] Rust
    crates through a small C ABI shim. *)

val version : unit -> string
(** [version ()] returns the OCaml FFI shim version. *)

type file_info = {
  hash : string;
  file_size : int64 option;
  sha256 : string option;
}
(** Metadata computed by xet-core for one file. *)

type repo_type = Model | Dataset | Space  (** Hugging Face repository kind. *)

val hash_files_raw : string list -> string
(** [hash_files_raw paths] computes Xet file hashes locally and returns the
    upstream JSON representation of a [XetFileInfo list].

    This calls xet-core's content-defined chunking and Merkle hashing code; no
    CAS server or Hugging Face credentials are required. *)

module File_download : sig
  type t
  (** Handle for a background file download. Mirrors Python's [XetFileDownload].
  *)

  val status : t -> string
  (** [status t] returns ["running"], ["finalizing"], ["completed"], or
      ["user_cancelled"]. Errors are raised as [Failure]. *)

  val cancel : t -> unit
  (** [cancel t] cancels this file download. *)
end

module File_download_group : sig
  type t
  (** Group of related file downloads. Mirrors Python's [XetFileDownloadGroup].
  *)

  val start_download_file : t -> file_info -> string -> File_download.t
  (** [start_download_file t file_info dest_path] queues [file_info] for
      download to [dest_path] and returns immediately with a download handle. *)

  val wait_to_finish : t -> string
  (** Wait for all downloads in the group to complete and return a compact JSON
      report. *)

  val abort : t -> unit
  (** Cancel all active downloads in this group. *)

  val progress : t -> string
  (** Return aggregate group progress as compact JSON. *)
end

module Session : sig
  type t
  (** Xet runtime session. Mirrors Python's [XetSession]. *)

  val create : unit -> t
  (** Create a session using xet-core's default config and environment
      overrides. *)

  val status : t -> string
  val abort : t -> unit

  val new_file_download_group :
    ?endpoint:string ->
    ?token:string ->
    ?token_expiry_unix_secs:int64 ->
    ?token_refresh_url:string ->
    ?token_refresh_headers:(string * string) list ->
    ?custom_headers:(string * string) list ->
    t ->
    File_download_group.t
  (** Create a file download group.

      The optional arguments intentionally mirror Python's
      [XetSession.new_file_download_group]: [endpoint], [token] with
      [token_expiry_unix_secs], [token_refresh_url] with
      [token_refresh_headers], and [custom_headers]. *)
end

val download_hf_file :
  ?repo_type:repo_type ->
  ?revision:string ->
  ?endpoint:string ->
  ?token:string ->
  repo_id:string ->
  filename:string ->
  destination:string ->
  unit ->
  string
(** Convenience helper for Hugging Face Hub files stored with Xet.

    It resolves Hub Xet metadata for [repo_id]/[filename], creates a download
    group with the returned token refresh route, downloads to [destination], and
    returns compact JSON metadata. This is a helper on top of the session/group
    API rather than the primary shape. *)
