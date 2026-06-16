(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let mkdir_p path =
  let rec loop path =
    if path = "" || path = "." || path = Filename.dir_sep then ()
    else if not (Sys.file_exists path) then begin
      loop (Filename.dirname path);
      try Unix.mkdir path 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ()
    end
  in
  loop path

let write_file path contents =
  mkdir_p (Filename.dirname path);
  let oc = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc contents)

let rec rm_rf path =
  if Sys.file_exists path then
    if Sys.is_directory path then begin
      let entries = Sys.readdir path in
      Array.iter (fun e -> rm_rf (Filename.concat path e)) entries;
      Unix.rmdir path
    end
    else Sys.remove path

let test_offline_uses_hub_snapshot_layout () =
  let cache_dir = Filename.temp_file "kaun-hf-cache-test-" "" in
  Sys.remove cache_dir;
  Unix.mkdir cache_dir 0o755;
  Fun.protect
    ~finally:(fun () -> rm_rf cache_dir)
    (fun () ->
      let commit = "0123456789abcdef0123456789abcdef01234567" in
      let storage = Filename.concat cache_dir "models--org--repo" in
      let snapshot_file =
        Filename.concat
          (Filename.concat (Filename.concat storage "snapshots") commit)
          "config.json"
      in
      write_file
        (Filename.concat (Filename.concat storage "refs") "main")
        commit;
      write_file snapshot_file "{}";
      let path =
        Kaun_hf.download_file ~cache_dir ~offline:true ~model_id:"org/repo"
          ~filename:"config.json" ()
      in
      equal ~msg:"snapshot path" string snapshot_file path)

let () =
  run "Kaun_hf cache layout"
    [
      test "offline uses HuggingFace Hub snapshot layout"
        test_offline_uses_hub_snapshot_layout;
    ]
