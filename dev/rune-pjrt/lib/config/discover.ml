(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module C = Configurator.V1

let file_exists path =
  try Sys.file_exists path with Sys_error _ -> false

let rec find_repo_root dir =
  let vendor_xla = Filename.concat dir "vendor/xla" in
  if file_exists vendor_xla then Some dir
  else
    let parent = Filename.dirname dir in
    if String.equal parent dir then None else find_repo_root parent

let () =
  C.main ~name:"rune_pjrt_discover" (fun c ->
      let vendor_xla =
        match find_repo_root (Sys.getcwd ()) with
        | Some root -> Filename.concat root "vendor/xla"
        | None -> Filename.concat (Sys.getcwd ()) "vendor/xla"
      in
      let include_dir = vendor_xla in
      let c_flags =
        if file_exists vendor_xla then
          [ "-DRUNE_PJRT_VENDOR_XLA=1"; "-I" ^ include_dir ]
        else []
      in
      C.Flags.write_sexp "c_flags.sexp" c_flags;
      C.Flags.write_sexp "c_library_flags.sexp" [ "-ldl" ])
