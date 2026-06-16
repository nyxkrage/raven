(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Tolk_cpu | Pjrt_cpu | Pjrt_cuda

let all = [ Tolk_cpu; Pjrt_cpu; Pjrt_cuda ]
let default = Pjrt_cuda

let to_string = function
  | Tolk_cpu -> "tolk-cpu"
  | Pjrt_cpu -> "pjrt-cpu"
  | Pjrt_cuda -> "pjrt-cuda"

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let of_string value =
  match String.lowercase_ascii value with
  | "tolk-cpu" | "tolk_cpu" | "tolk" -> Tolk_cpu
  | "pjrt-cpu" | "pjrt_cpu" -> Pjrt_cpu
  | "pjrt-cuda" | "pjrt_cuda" | "cuda" -> Pjrt_cuda
  | value ->
      invalid_argf
        "backend must be \"tolk-cpu\", \"pjrt-cpu\", or \"pjrt-cuda\", got %S"
        value

let of_env ?(var = "RUNE_JIT_BACKEND") ?(default = default) () =
  match Sys.getenv_opt var with
  | None -> default
  | Some value -> of_string value

let int_of_env name ~default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value -> (
      match int_of_string_opt value with
      | Some value -> value
      | None -> invalid_argf "%s must be an integer, got %S" name value)

let pjrt_device_id_of_env ?(var = "RUNE_PJRT_DEVICE_ID") ?(default = 0) () =
  int_of_env var ~default

let pjrt_backend = function
  | Pjrt_cpu -> Some `Cpu
  | Pjrt_cuda -> Some `Cuda
  | Tolk_cpu -> None

let available = function
  | Tolk_cpu -> true
  | (Pjrt_cpu | Pjrt_cuda) as backend ->
      Rune_pjrt.backend_available (Option.get (pjrt_backend backend))

let require backend =
  if not (available backend) then
    match pjrt_backend backend with
    | None -> invalid_arg "Tolk CPU backend is unavailable"
    | Some pjrt ->
        failwith
          (Printf.sprintf "PJRT backend %s is unavailable: %s"
             (Rune_pjrt.Backend.to_string pjrt)
             (Rune_pjrt.status ()))

let device ?(tolk_name = "CPU") ?pjrt_device_id backend =
  match backend with
  | Tolk_cpu -> Jit.Device.tolk (Tolk_cpu.create tolk_name)
  | Pjrt_cpu ->
      let device_id =
        Option.value pjrt_device_id ~default:(pjrt_device_id_of_env ())
      in
      let device = Rune_pjrt.Device.cpu ~device_id () in
      require backend;
      Jit.Device.pjrt device
  | Pjrt_cuda ->
      let device_id =
        Option.value pjrt_device_id ~default:(pjrt_device_id_of_env ())
      in
      let device = Rune_pjrt.Device.cuda ~device_id () in
      require backend;
      Jit.Device.pjrt device
