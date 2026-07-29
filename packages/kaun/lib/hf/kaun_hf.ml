(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Types *)

type revision = Main | Rev of string

(* Error messages *)

let err_no_curl = "curl not found on PATH"
let err_download url = Printf.sprintf "Failed to download %s" url

let err_offline model_id filename =
  Printf.sprintf "Not cached in HuggingFace Hub cache (offline): %s/%s" model_id
    filename

let err_metadata url = Printf.sprintf "Failed to fetch metadata for %s" url

let err_missing_header url name =
  Printf.sprintf "Missing %s header in metadata response for %s" name url

let err_no_safetensors model_id =
  Printf.sprintf "No safetensors found for %s" model_id

let err_missing_tensor model_id name path =
  Printf.sprintf "%s: tensor %S missing in shard %s" model_id name path

let err_empty_weight_map = "Empty weight_map in index file"
let err_missing_weight_map = "Missing weight_map in index file"

let err_incomplete_shards =
  "Incomplete shard loading: not all weight_map tensors were found"

(* HuggingFace Hub cache *)

let home_dir () =
  match Sys.getenv_opt "HOME" with Some d when d <> "" -> d | _ -> "."

let expand_user path =
  let len = String.length path in
  if path = "~" then home_dir ()
  else if len >= 2 && path.[0] = '~' && path.[1] = '/' then
    Filename.concat (home_dir ()) (String.sub path 2 (len - 2))
  else path

let default_cache_dir () =
  match Sys.getenv_opt "HF_HUB_CACHE" with
  | Some d when d <> "" -> expand_user d
  | _ -> (
      match Sys.getenv_opt "HUGGINGFACE_HUB_CACHE" with
      | Some d when d <> "" -> expand_user d
      | _ ->
          let hf_home =
            match Sys.getenv_opt "HF_HOME" with
            | Some d when d <> "" -> expand_user d
            | _ ->
                let xdg =
                  match Sys.getenv_opt "XDG_CACHE_HOME" with
                  | Some d when d <> "" -> expand_user d
                  | _ -> Filename.concat (home_dir ()) ".cache"
                in
                Filename.concat xdg "huggingface"
          in
          Filename.concat hf_home "hub")

(* Filesystem *)

let rec mkdir_p path =
  if path = "" || path = "." || path = Filename.dir_sep then ()
  else if not (Sys.file_exists path) then begin
    mkdir_p (Filename.dirname path);
    try Unix.mkdir path 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ()
  end

let rec rm_rf path =
  if Sys.is_directory path then begin
    let entries = Sys.readdir path in
    Array.iter (fun e -> rm_rf (Filename.concat path e)) entries;
    Unix.rmdir path
  end
  else Sys.remove path

(* HTTP via curl *)

let curl_available =
  lazy (Unix.system "command -v curl >/dev/null 2>&1" = Unix.WEXITED 0)

let check_curl () = if not (Lazy.force curl_available) then failwith err_no_curl

let header_flags headers =
  List.map
    (fun (k, v) -> Printf.sprintf "-H %s" (Filename.quote (k ^ ": " ^ v)))
    headers
  |> String.concat " "

let curl_download ~headers ~url ~dest () =
  check_curl ();
  mkdir_p (Filename.dirname dest);
  let hdr = header_flags headers in
  let cmd =
    Printf.sprintf "curl -L --fail -s %s -o %s %s" hdr (Filename.quote dest)
      (Filename.quote url)
  in
  match Unix.system cmd with
  | Unix.WEXITED 0 -> ()
  | _ ->
      (try Sys.remove dest with Sys_error _ -> ());
      failwith (err_download url)

let curl_head ~headers ~url () =
  check_curl ();
  let path = Filename.temp_file "kaun-hf-headers" ".txt" in
  let hdr = header_flags headers in
  let cmd =
    Printf.sprintf "curl --head --fail -s %s -D %s -o /dev/null %s" hdr
      (Filename.quote path) (Filename.quote url)
  in
  match Unix.system cmd with
  | Unix.WEXITED 0 -> path
  | _ ->
      (try Sys.remove path with Sys_error _ -> ());
      failwith (err_metadata url)

(* Hub URL and cache paths *)

let revision_string = function Main -> "main" | Rev r -> r

let hub_url ~model_id ~revision ~filename =
  Printf.sprintf "https://huggingface.co/%s/resolve/%s/%s" model_id
    (revision_string revision) filename

let auth_headers = function
  | Some t -> [ ("Authorization", "Bearer " ^ t) ]
  | None -> []

let split_on_char sep s =
  let rec loop acc start i =
    if i = String.length s then List.rev (String.sub s start (i - start) :: acc)
    else if s.[i] = sep then
      loop (String.sub s start (i - start) :: acc) (i + 1) (i + 1)
    else loop acc start (i + 1)
  in
  loop [] 0 0

let repo_folder_name ~model_id =
  "models--" ^ String.concat "--" (split_on_char '/' model_id)

let storage_folder ~cache_dir ~model_id =
  Filename.concat cache_dir (repo_folder_name ~model_id)

let relative_filename filename =
  let parts = split_on_char '/' filename in
  if List.exists (fun p -> p = "" || p = "." || p = "..") parts then
    invalid_arg ("invalid HuggingFace filename: " ^ filename);
  List.fold_left Filename.concat "" parts

let snapshot_path ~storage ~commit_hash ~filename =
  Filename.concat
    (Filename.concat (Filename.concat storage "snapshots") commit_hash)
    (relative_filename filename)

let blob_path ~storage ~etag =
  Filename.concat (Filename.concat storage "blobs") etag

let ref_path ~storage ~revision =
  Filename.concat (Filename.concat storage "refs") revision

let is_hex c =
  match c with '0' .. '9' | 'a' .. 'f' | 'A' .. 'F' -> true | _ -> false

let is_commit_hash s = String.length s = 40 && String.for_all is_hex s

let read_file path =
  let ic = open_in path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let write_file path contents =
  mkdir_p (Filename.dirname path);
  let oc = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc contents)

let trim s =
  let is_space = function ' ' | '\t' | '\r' | '\n' -> true | _ -> false in
  let len = String.length s in
  let i = ref 0 in
  while !i < len && is_space s.[!i] do
    incr i
  done;
  let j = ref (len - 1) in
  while !j >= !i && is_space s.[!j] do
    decr j
  done;
  if !j < !i then "" else String.sub s !i (!j - !i + 1)

let lowercase_ascii s =
  String.map
    (fun c -> match c with 'A' .. 'Z' -> Char.chr (Char.code c + 32) | _ -> c)
    s

let env_truthy name =
  match Sys.getenv_opt name with
  | None | Some "" -> false
  | Some v -> (
      match lowercase_ascii (trim v) with
      | "1" | "on" | "true" | "yes" -> true
      | _ -> false)

let default_offline () =
  env_truthy "HF_HUB_OFFLINE" || env_truthy "TRANSFORMERS_OFFLINE"

let normalize_etag s =
  let s = trim s in
  let s =
    if String.length s >= 2 && String.sub s 0 2 = "W/" then
      String.sub s 2 (String.length s - 2)
    else s
  in
  let len = String.length s in
  if len >= 2 && s.[0] = '"' && s.[len - 1] = '"' then String.sub s 1 (len - 2)
  else s

let parse_headers path =
  let ic = open_in path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
      let headers = Hashtbl.create 16 in
      (try
         while true do
           let line = input_line ic in
           match String.index_opt line ':' with
           | None -> ()
           | Some i ->
               let name = lowercase_ascii (trim (String.sub line 0 i)) in
               let value =
                 trim (String.sub line (i + 1) (String.length line - i - 1))
               in
               Hashtbl.replace headers name value
         done
       with End_of_file -> ());
      headers)

let header headers name = Hashtbl.find_opt headers (lowercase_ascii name)

type metadata = { commit_hash : string; etag : string; size : int option }

let fetch_metadata ~headers ~url =
  let headers_path = curl_head ~headers ~url () in
  Fun.protect
    ~finally:(fun () -> try Sys.remove headers_path with Sys_error _ -> ())
    (fun () ->
      let headers = parse_headers headers_path in
      let commit_hash =
        match header headers "x-repo-commit" with
        | Some v -> v
        | None -> failwith (err_missing_header url "X-Repo-Commit")
      in
      let etag =
        match
          Option.bind
            (match header headers "x-linked-etag" with
            | Some _ as v -> v
            | None -> header headers "etag")
            (fun v ->
              let v = normalize_etag v in
              if v = "" then None else Some v)
        with
        | Some v -> v
        | None -> failwith (err_missing_header url "ETag")
      in
      let size =
        match
          match header headers "x-linked-size" with
          | Some _ as v -> v
          | None -> header headers "content-length"
        with
        | Some v -> int_of_string_opt v
        | None -> None
      in
      { commit_hash; etag; size })

let cached_file ~storage ~revision ~filename =
  let revision = revision_string revision in
  let commit_hash =
    if is_commit_hash revision then Some revision
    else
      let ref_path = ref_path ~storage ~revision in
      if Sys.file_exists ref_path then Some (trim (read_file ref_path))
      else None
  in
  Option.bind commit_hash (fun commit_hash ->
      let path = snapshot_path ~storage ~commit_hash ~filename in
      if Sys.file_exists path then Some path else None)

let relative_blob_link ~filename ~etag =
  let parts = split_on_char '/' filename in
  let depth = max 0 (List.length parts - 1) in
  let rec parent_prefix n acc =
    if n = 0 then acc else parent_prefix (n - 1) (Filename.concat ".." acc)
  in
  parent_prefix depth
    (Filename.concat ".." (Filename.concat ".." (Filename.concat "blobs" etag)))

let copy_file ~src ~dst =
  let ic = open_in_bin src in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
      let oc = open_out_bin dst in
      Fun.protect
        ~finally:(fun () -> close_out oc)
        (fun () ->
          let buf = Bytes.create 65536 in
          let rec loop () =
            let n = input ic buf 0 (Bytes.length buf) in
            if n > 0 then begin
              output oc buf 0 n;
              loop ()
            end
          in
          loop ()))

let create_pointer ~blob ~pointer ~filename ~etag =
  mkdir_p (Filename.dirname pointer);
  if Sys.file_exists pointer then ()
  else
    try
      let rel = relative_blob_link ~filename ~etag in
      Unix.symlink rel pointer
    with Unix.Unix_error _ -> copy_file ~src:blob ~dst:pointer

let update_ref ~storage ~revision ~commit_hash =
  let revision = revision_string revision in
  if not (is_commit_hash revision) then
    write_file (ref_path ~storage ~revision) commit_hash

(* Xet download *)

let download_file_via_xet ?token ?(revision = Main) ~model_id ~filename
    ~destination () =
  Kaun_hf_xet.download_hf_file ?token ~model_id ~filename
    ~revision:(revision_string revision) ~destination ()

let try_xet_download ?token ~revision ~model_id ~filename ~destination () =
  if not (Kaun_hf_xet.available ()) then None
  else
    try
      let _json =
        download_file_via_xet ?token ~revision ~model_id ~filename ~destination
          ()
      in
      if Sys.file_exists destination then Some destination else None
    with Failure _ -> None

(* Downloading *)

let download_file ?token ?cache_dir ?offline ?(revision = Main) ~model_id
    ~filename () =
  let token =
    match token with Some _ as t -> t | None -> Sys.getenv_opt "HF_TOKEN"
  in
  let cache_dir = Option.value cache_dir ~default:(default_cache_dir ()) in
  let offline = Option.value offline ~default:(default_offline ()) in
  let storage = storage_folder ~cache_dir ~model_id in
  match cached_file ~storage ~revision ~filename with
  | Some path -> path
  | None when offline -> failwith (err_offline model_id filename)
  | None -> begin
      let xet_destination =
        snapshot_path ~storage ~commit_hash:"xet" ~filename
      in
      match
        try_xet_download ?token ~revision ~model_id ~filename
          ~destination:xet_destination ()
      with
      | Some path -> path
      | None -> begin
          let url = hub_url ~model_id ~revision ~filename in
          let headers = auth_headers token in
          let metadata = fetch_metadata ~headers ~url in
          let blob = blob_path ~storage ~etag:metadata.etag in
          let pointer =
            snapshot_path ~storage ~commit_hash:metadata.commit_hash ~filename
          in
          mkdir_p (Filename.dirname blob);
          mkdir_p (Filename.dirname pointer);
          if not (Sys.file_exists blob) then (
            let incomplete = blob ^ ".incomplete" in
            try
              curl_download ~headers ~url ~dest:incomplete ();
              (match metadata.size with
              | Some expected ->
                  let actual = (Unix.stat incomplete).st_size in
                  if actual <> expected then
                    failwith
                      (Printf.sprintf "Downloaded %s has size %d, expected %d"
                         url actual expected)
              | None -> ());
              Sys.rename incomplete blob
            with e ->
              (try Sys.remove incomplete with Sys_error _ -> ());
              raise e);
          update_ref ~storage ~revision ~commit_hash:metadata.commit_hash;
          create_pointer ~blob ~pointer ~filename ~etag:metadata.etag;
          pointer
        end
    end

(* JSON helpers *)

let read_json_file path =
  let ic = open_in path in
  let s =
    Fun.protect
      ~finally:(fun () -> close_in ic)
      (fun () -> really_input_string ic (in_channel_length ic))
  in
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> v
  | Error e -> failwith e

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

(* Tensor conversion *)

let to_ptree_tensor (Nx_io.P nx) = Kaun.Ptree.P nx

(* Loading *)

let load_entries ?allowed_names path =
  let archive = Nx_io.load_safetensors path in
  match allowed_names with
  | None ->
      Hashtbl.fold
        (fun name packed acc -> (name, to_ptree_tensor packed) :: acc)
        archive []
  | Some names ->
      List.map
        (fun name ->
          match Hashtbl.find_opt archive name with
          | Some packed -> (name, to_ptree_tensor packed)
          | None -> failwith (err_missing_tensor "" name path))
        names

let try_download f =
  try Some (f ()) with Failure _ -> None | Sys_error _ -> None

let load_sharded ~download index_filename =
  match try_download (fun () -> download index_filename) with
  | None -> None
  | Some index_path ->
      let json = read_json_file index_path in
      let weight_map =
        match json_mem "weight_map" json with
        | Jsont.Object (entries, _) ->
            List.map
              (fun ((tensor_name, _), shard_json) ->
                match shard_json with
                | Jsont.String (shard, _) -> (tensor_name, shard)
                | _ -> failwith err_missing_weight_map)
              entries
        | _ -> failwith err_missing_weight_map
      in
      if weight_map = [] then failwith err_empty_weight_map;
      (* Group tensors by shard filename, preserving file order *)
      let shards_by_file = Hashtbl.create 8 in
      let file_order = ref [] in
      List.iter
        (fun (tensor_name, shard_filename) ->
          match Hashtbl.find_opt shards_by_file shard_filename with
          | Some tensors ->
              Hashtbl.replace shards_by_file shard_filename
                (tensor_name :: tensors)
          | None ->
              Hashtbl.add shards_by_file shard_filename [ tensor_name ];
              file_order := shard_filename :: !file_order)
        weight_map;
      let file_order = List.rev !file_order in
      let seen = Hashtbl.create (List.length weight_map) in
      let entries =
        List.fold_left
          (fun acc shard_filename ->
            let shard_path = download shard_filename in
            let tensors =
              match Hashtbl.find_opt shards_by_file shard_filename with
              | Some names -> List.rev names
              | None -> []
            in
            let new_entries = load_entries ~allowed_names:tensors shard_path in
            List.iter
              (fun (name, _) -> Hashtbl.replace seen name ())
              new_entries;
            List.rev_append new_entries acc)
          [] file_order
      in
      if Hashtbl.length seen <> List.length weight_map then
        failwith err_incomplete_shards;
      Some (List.rev entries)

let load_single ~download filename =
  match try_download (fun () -> download filename) with
  | None -> None
  | Some path -> Some (load_entries path)

let load_config ?token ?cache_dir ?offline ?revision ~model_id () =
  let path =
    download_file ?token ?cache_dir ?offline ?revision ~model_id
      ~filename:"config.json" ()
  in
  read_json_file path

let load_weights ?token ?cache_dir ?offline ?revision ~model_id () =
  let download filename =
    download_file ?token ?cache_dir ?offline ?revision ~model_id ~filename ()
  in
  match load_sharded ~download "model.safetensors.index.json" with
  | Some entries -> entries
  | None -> (
      match load_single ~download "model.safetensors" with
      | Some entries -> entries
      | None -> failwith (err_no_safetensors model_id))

(* Cache management *)

let clear_cache ?cache_dir ?model_id () =
  let cache_dir = Option.value cache_dir ~default:(default_cache_dir ()) in
  match model_id with
  | Some id ->
      let path = storage_folder ~cache_dir ~model_id:id in
      if Sys.file_exists path then rm_rf path
  | None -> if Sys.file_exists cache_dir then rm_rf cache_dir
