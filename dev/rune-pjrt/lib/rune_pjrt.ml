(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Backend = Backend
module Causal_lm = Causal_lm
module Error = Error
module Ir = Ir
module Runtime = Runtime
module Signature = Signature
module Stablehlo = Stablehlo
module Trace = Trace

type packed = Trace.packed = Tensor : ('a, 'b) Nx.t -> packed

let backend_available = Runtime.backend_available
let status = Runtime.status

let pack_tensor (type a b) (t : (a, b) Nx.t) = Trace.Tensor t
let unpack_tensor (Trace.Tensor t) = Obj.magic t

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
            Trace.capture_many ~name:"jit"
              (fun xs ->
                f (List.map pack_tensor xs)
                |> List.map unpack_tensor)
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
        f (List.map unpack_tensor inputs)
        |> List.map pack_tensor)
  in
  fun inputs ->
    packed (List.map pack_tensor inputs)
    |> List.map (fun (Trace.Tensor t) -> Obj.magic t)

let jit ?backend ?device_id f =
  let many =
    jits ?backend ?device_id (fun inputs ->
        match inputs with
        | [ x ] -> [ f x ]
        | _ -> assert false)
  in
  fun input ->
    match many [ input ] with
    | [ output ] -> output
    | _ -> assert false
