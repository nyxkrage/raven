(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type packed = Tensor : ('a, 'b) Nx.t -> packed

module Internal = struct
  type handler = { library : string; symbol : string }
  type identity = { library : string; library_digest : string; target : string }

  let identity (handler : handler) =
    let library = Unix.realpath handler.library in
    let raw_library_digest = Digest.file library in
    let library_digest = Digest.to_hex raw_library_digest in
    let digest =
      Digest.string (raw_library_digest ^ "\000" ^ handler.symbol)
      |> Digest.to_hex
    in
    { library; library_digest; target = "raven_cuda_" ^ digest }

  let target handler = (identity handler).target

  type ('a, 'b) request = {
    handler : handler;
    inputs : packed list;
    fallback : unit -> ('a, 'b) Nx.t;
  }

  type ('a, 'b) decision = Use_kernel of ('a, 'b) Nx.t | Use_fallback
  type _ Effect.t += Call : ('a, 'b) request -> ('a, 'b) decision Effect.t
end

module Kernel = struct
  type t = { fwd : Internal.handler option; bwd : Internal.handler option }

  let handler library symbol = Internal.{ library; symbol }

  let executable_directory =
    lazy
      (let executable =
         try Unix.realpath "/proc/self/exe"
         with Unix.Unix_error _ -> (
           let executable =
             if Filename.is_relative Sys.executable_name then
               Filename.concat (Sys.getcwd ()) Sys.executable_name
             else Sys.executable_name
           in
           try Unix.realpath executable with Unix.Unix_error _ -> executable)
       in
       Filename.dirname executable)

  let resolve_library library =
    if Filename.is_relative library then
      Filename.concat (Lazy.force executable_directory) library
    else library

  let create ~library ?fwd ?bwd () =
    if String.trim library = "" then
      invalid_arg "Rune_pjrt.Ffi.Kernel.create: library must not be empty";
    let validate name =
      if String.trim name = "" then
        invalid_arg
          "Rune_pjrt.Ffi.Kernel.create: handler symbol must not be empty";
      name
    in
    let library = resolve_library library in
    let fwd = Option.map (fun name -> handler library (validate name)) fwd in
    let bwd = Option.map (fun name -> handler library (validate name)) bwd in
    if Option.is_none fwd && Option.is_none bwd then
      invalid_arg
        "Rune_pjrt.Ffi.Kernel.create: at least one of fwd or bwd is required";
    { fwd; bwd }
end

let call handler ~inputs ~fallback =
  match handler with
  | None -> fallback ()
  | Some handler -> (
      let decision =
        try Effect.perform (Internal.Call { handler; inputs; fallback })
        with Effect.Unhandled (Internal.Call _) -> Internal.Use_fallback
      in
      match decision with
      | Internal.Use_kernel output -> output
      | Internal.Use_fallback -> fallback ())

let call_fwd (kernel : Kernel.t) ~inputs ~fallback =
  call kernel.fwd ~inputs ~fallback

let call_bwd (kernel : Kernel.t) ~inputs ~fallback =
  call kernel.bwd ~inputs ~fallback
