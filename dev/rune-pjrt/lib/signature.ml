(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type tensor = {
  shape : int array;
  dtype : string;
}

type t = {
  backend : Backend.t;
  device_id : int;
  inputs : tensor list;
}

let tensor_of_t (type a b) (t : (a, b) Nx.t) =
  { shape = Array.copy (Nx.shape t); dtype = Nx_core.Dtype.to_string (Nx.dtype t) }

let tensor_of_packed (Trace.Tensor t) = tensor_of_t t

let of_packed ~backend ~device_id inputs =
  let inputs = List.map tensor_of_packed inputs in
  { backend; device_id; inputs }

let of_tensors ~backend ~device_id inputs =
  let inputs = List.map tensor_of_t inputs in
  { backend; device_id; inputs }

let key signature =
  let inputs =
    signature.inputs
    |> List.map (fun input ->
           Printf.sprintf "%s%s" input.dtype (Nx_core.Shape.to_string input.shape))
    |> String.concat ";"
  in
  Printf.sprintf "%s:%d:%s" (Backend.to_string signature.backend)
    signature.device_id inputs
