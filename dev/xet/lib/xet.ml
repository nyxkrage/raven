external version : unit -> string = "caml_xet_version"
external hash_files_json : string -> string = "caml_xet_hash_files_json"

external download_hf_file_json : string -> string
  = "caml_xet_download_hf_file_json"

type session
type file_download_group
type file_download

external session_create : unit -> session = "caml_xet_session_create"
external session_status : session -> string = "caml_xet_session_status"
external session_abort : session -> unit = "caml_xet_session_abort"

external session_new_file_download_group :
  session -> string -> file_download_group
  = "caml_xet_session_new_file_download_group"

external file_download_group_start_download_file :
  file_download_group -> string -> string -> file_download
  = "caml_xet_file_download_group_start_download_file"

external file_download_group_wait_to_finish_json : file_download_group -> string
  = "caml_xet_file_download_group_wait_to_finish_json"

external file_download_group_abort : file_download_group -> unit
  = "caml_xet_file_download_group_abort"

external file_download_group_progress_json : file_download_group -> string
  = "caml_xet_file_download_group_progress_json"

external file_download_status : file_download -> string
  = "caml_xet_file_download_status"

external file_download_cancel : file_download -> unit
  = "caml_xet_file_download_cancel"

type file_info = {
  hash : string;
  file_size : int64 option;
  sha256 : string option;
}

type repo_type = Model | Dataset | Space

let version = version

let json_string s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  String.iter
    (function
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | '\b' -> Buffer.add_string b "\\b"
      | '\012' -> Buffer.add_string b "\\f"
      | '\n' -> Buffer.add_string b "\\n"
      | '\r' -> Buffer.add_string b "\\r"
      | '\t' -> Buffer.add_string b "\\t"
      | c when Char.code c < 0x20 ->
          Buffer.add_string b (Printf.sprintf "\\u%04x" (Char.code c))
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"';
  Buffer.contents b

let json_field name value = Printf.sprintf "%s:%s" (json_string name) value

let json_object fields =
  Printf.sprintf "{%s}" (fields |> List.filter_map Fun.id |> String.concat ",")

let json_optional_string_field name = function
  | None -> None
  | Some value -> Some (json_field name (json_string value))

let json_optional_int64_field name = function
  | None -> None
  | Some value -> Some (json_field name (Int64.to_string value))

let json_string_map fields =
  fields
  |> List.map (fun (name, value) -> json_field name (json_string value))
  |> String.concat "," |> Printf.sprintf "{%s}"

let repo_type_to_string = function
  | Model -> "model"
  | Dataset -> "dataset"
  | Space -> "space"

let file_info_to_json file_info =
  json_object
    [
      Some (json_field "hash" (json_string file_info.hash));
      json_optional_int64_field "file_size" file_info.file_size;
      json_optional_string_field "sha256" file_info.sha256;
    ]

let hash_files_raw paths =
  let json_paths =
    paths |> List.map json_string |> String.concat "," |> Printf.sprintf "[%s]"
  in
  hash_files_json json_paths

module File_download = struct
  type t = file_download

  let status = file_download_status
  let cancel = file_download_cancel
end

module File_download_group = struct
  type t = file_download_group

  let start_download_file t file_info dest_path =
    file_download_group_start_download_file t
      (file_info_to_json file_info)
      dest_path

  let wait_to_finish = file_download_group_wait_to_finish_json
  let abort = file_download_group_abort
  let progress = file_download_group_progress_json
end

module Session = struct
  type t = session

  let create = session_create
  let status = session_status
  let abort = session_abort

  let new_file_download_group ?endpoint ?token ?token_expiry_unix_secs
      ?token_refresh_url ?(token_refresh_headers = []) ?(custom_headers = []) t
      =
    let config =
      json_object
        [
          json_optional_string_field "endpoint" endpoint;
          json_optional_string_field "token" token;
          json_optional_int64_field "token_expiry_unix_secs"
            token_expiry_unix_secs;
          json_optional_string_field "token_refresh_url" token_refresh_url;
          Some
            (json_field "token_refresh_headers"
               (json_string_map token_refresh_headers));
          Some (json_field "custom_headers" (json_string_map custom_headers));
        ]
    in
    session_new_file_download_group t config
end

let download_hf_file ?(repo_type = Model) ?revision ?endpoint ?token ~repo_id
    ~filename ~destination () =
  let fields =
    [
      Some (json_field "repo_id" (json_string repo_id));
      Some (json_field "filename" (json_string filename));
      Some (json_field "destination" (json_string destination));
      Some
        (json_field "repo_type" (json_string (repo_type_to_string repo_type)));
      json_optional_string_field "revision" revision;
      json_optional_string_field "endpoint" endpoint;
      json_optional_string_field "token" token;
    ]
  in
  download_hf_file_json (json_object fields)
