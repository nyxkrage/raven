(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module C = Configurator.V1

let file_exists path = try Sys.file_exists path with Sys_error _ -> false
let pjrt_header root = Filename.concat root "xla/pjrt/c/pjrt_c_api.h"
let has_pjrt_headers root = file_exists (pjrt_header root)

let rec find_repo_root dir =
  let xla = Filename.concat dir "vendor/xla" in
  if has_pjrt_headers xla then Some xla
  else
    let parent = Filename.dirname dir in
    if String.equal parent dir then None else find_repo_root parent

let bundled_headers_root () =
  match Sys.getenv_opt "DUNE_SOURCEROOT" with
  | Some root -> Filename.concat root "dev/rune-pjrt/vendor"
  | None ->
      let project = Filename.dirname (Sys.getcwd ()) in
      Filename.concat project "vendor"

let () =
  C.main ~name:"rune_pjrt_discover" (fun c ->
      let include_dir =
        match Sys.getenv_opt "RUNE_PJRT_XLA_SOURCE" with
        | Some path when has_pjrt_headers path -> path
        | Some path ->
            C.die "RUNE_PJRT_XLA_SOURCE=%s does not contain %s" path
              (pjrt_header path)
        | None -> (
            match find_repo_root (Sys.getcwd ()) with
            | Some root -> root
            | None ->
                let bundled = bundled_headers_root () in
                if has_pjrt_headers bundled then bundled
                else C.die "bundled PJRT C headers are missing from %s" bundled)
      in
      let c_flags = [ "-DRUNE_PJRT_VENDOR_XLA=1"; "-I" ^ include_dir ] in
      C.Flags.write_sexp "c_flags.sexp" c_flags;
      C.Flags.write_sexp "c_library_flags.sexp" [ "-ldl" ])
