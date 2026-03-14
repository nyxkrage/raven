(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Unsupported_effect of string
  | Unsupported_op of string
  | Unsupported_program of string
  | Runtime_unavailable of string

exception Error of t

let raise err = Stdlib.raise (Error err)

let to_string = function
  | Unsupported_effect name ->
      Printf.sprintf "unsupported effect while tracing: %s" name
  | Unsupported_op name -> Printf.sprintf "unsupported PJRT op: %s" name
  | Unsupported_program msg -> Printf.sprintf "unsupported PJRT program: %s" msg
  | Runtime_unavailable msg -> Printf.sprintf "PJRT runtime unavailable: %s" msg
