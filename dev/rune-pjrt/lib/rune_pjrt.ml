(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Backend = Backend
module Causal_lm = Causal_lm
module Device = Device
module Error = Error
module Ffi = Ffi
module Grouped_gemm = Grouped_gemm
module Ir = Ir
module Runtime = Runtime
module Signature = Signature
module Stablehlo = Stablehlo
module Trace = Trace
module Triton = Triton

type packed = Trace.packed = Tensor : ('a, 'b) Nx.t -> packed

let backend_available = Runtime.backend_available
let status = Runtime.status
let pack_tensor (type a b) (t : (a, b) Nx.t) = Trace.Tensor t
let unpack_tensor (Trace.Tensor t) = Obj.magic t

module Device_buffer = struct
  type ('a, 'b) t = ('a, 'b) Runtime.device_buffer
  type packed = Pack : ('a, 'b) t -> packed

  let of_host ?(backend = `Cuda) ?(device_id = 0) tensor =
    Runtime.device_buffer_of_host ~backend ~device_id tensor

  let to_host = Runtime.device_buffer_to_host
  let await = Runtime.device_buffer_await
  let shape = Runtime.device_buffer_shape
  let dtype = Runtime.device_buffer_dtype
end

let pack_device_buffer (type a b) (buffer : (a, b) Device_buffer.t) =
  Device_buffer.Pack buffer

let unpack_device_buffer (Device_buffer.Pack buffer) = Obj.magic buffer

let jits_packed ?(backend = `Cuda) ?(device_id = 0) f =
  let cache = Hashtbl.create 8 in
  fun inputs ->
    let signature = Signature.of_packed ~backend ~device_id inputs in
    let key = Signature.key signature in
    let compiled =
      match Hashtbl.find_opt cache key with
      | Some compiled -> compiled
      | None ->
          let typed_inputs = List.map unpack_tensor inputs in
          let capture =
            Trace.capture_many ~name:"jit" ~enable_ffi:(backend = `Cuda)
              (fun xs -> f (List.map pack_tensor xs) |> List.map unpack_tensor)
              typed_inputs
          in
          let compiled =
            Runtime.compile ~backend ~device_id ~signature capture.program
              capture.outputs
          in
          Hashtbl.replace cache key compiled;
          compiled
    in
    Runtime.execute compiled (List.map unpack_tensor inputs)

let jits ?(backend = `Cuda) ?(device_id = 0) f =
  let packed =
    jits_packed ~backend ~device_id (fun inputs ->
        f (List.map unpack_tensor inputs) |> List.map pack_tensor)
  in
  fun inputs ->
    packed (List.map pack_tensor inputs)
    |> List.map (fun (Trace.Tensor t) -> Obj.magic t)

let jit ?backend ?device_id f =
  let many =
    jits ?backend ?device_id (fun inputs ->
        match inputs with [ x ] -> [ f x ] | _ -> assert false)
  in
  fun input ->
    match many [ input ] with [ output ] -> output | _ -> assert false

let jits_device_packed ?(backend = `Cuda) ?(device_id = 0) f =
  let cache = Hashtbl.create 8 in
  let last_compiled : Runtime.compiled option ref = ref None in
  fun inputs ->
    let runtime_inputs =
      List.map
        (fun (Device_buffer.Pack buffer) -> Runtime.Device_buffer buffer)
        inputs
    in
    let compiled =
      match !last_compiled with
      | Some compiled
        when Runtime.device_buffers_match_signature compiled.signature
               runtime_inputs ->
          compiled
      | _ ->
          let signature_inputs =
            List.map
              (fun (Runtime.Device_buffer buffer) ->
                {
                  Signature.shape = Runtime.device_buffer_shape buffer;
                  dtype =
                    Nx_core.Dtype.to_string (Runtime.device_buffer_dtype buffer);
                })
              runtime_inputs
          in
          let signature : Signature.t =
            { backend; device_id; inputs = signature_inputs }
          in
          let key = Signature.key signature in
          let compiled =
            match Hashtbl.find_opt cache key with
            | Some compiled -> compiled
            | None ->
                let placeholders =
                  List.map
                    (fun (Runtime.Device_buffer buffer) ->
                      pack_tensor (Runtime.device_buffer_placeholder buffer))
                    runtime_inputs
                in
                let typed_inputs = List.map unpack_tensor placeholders in
                let capture =
                  Trace.capture_many ~name:"jit" ~enable_ffi:(backend = `Cuda)
                    (fun xs ->
                      f (List.map pack_tensor xs) |> List.map unpack_tensor)
                    typed_inputs
                in
                let compiled =
                  Runtime.compile ~backend ~device_id ~signature capture.program
                    capture.outputs
                in
                Hashtbl.replace cache key compiled;
                compiled
          in
          last_compiled := Some compiled;
          compiled
    in
    Runtime.execute_device compiled runtime_inputs
    |> List.map (fun (Runtime.Device_buffer buffer) ->
        Device_buffer.Pack buffer)

let jits_device ?(backend = `Cuda) ?(device_id = 0) f =
  let packed =
    jits_device_packed ~backend ~device_id (fun inputs ->
        f (List.map unpack_tensor inputs) |> List.map pack_tensor)
  in
  fun inputs ->
    packed (List.map pack_device_buffer inputs) |> List.map unpack_device_buffer

let jit_device ?backend ?device_id f =
  let many =
    jits_device ?backend ?device_id (fun inputs ->
        match inputs with [ input ] -> [ f input ] | _ -> assert false)
  in
  fun input ->
    match many [ input ] with [ output ] -> output | _ -> assert false
