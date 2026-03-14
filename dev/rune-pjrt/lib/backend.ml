(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = [ `Cpu | `Cuda ]

let to_string : t -> string = function `Cpu -> "cpu" | `Cuda -> "cuda"
