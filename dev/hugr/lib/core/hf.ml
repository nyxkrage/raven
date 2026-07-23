(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let member name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, value) -> Some value
      | None -> None)
  | _ -> None

let int_exn ~ctx name json =
  match member name json with
  | Some (Jsont.Number (value, _)) -> int_of_float value
  | _ -> invalid_argf "%s: expected integer field %S" ctx name

let int_opt name json =
  match member name json with
  | Some (Jsont.Number (value, _)) -> Some (int_of_float value)
  | _ -> None

let float_exn ~ctx name json =
  match member name json with
  | Some (Jsont.Number (value, _)) -> value
  | _ -> invalid_argf "%s: expected number field %S" ctx name

let float_opt name json =
  match member name json with
  | Some (Jsont.Number (value, _)) -> Some value
  | _ -> None

let bool_opt name json =
  match member name json with
  | Some (Jsont.Bool (value, _)) -> Some value
  | _ -> None

let string_opt name json =
  match member name json with
  | Some (Jsont.String (value, _)) -> Some value
  | _ -> None

type weights = {
  tensors : (string, Kaun.Ptree.tensor) Hashtbl.t;
  used : (string, unit) Hashtbl.t;
}

let weights tensors =
  let table = Hashtbl.create (List.length tensors) in
  List.iter
    (fun (name, tensor) ->
      if Hashtbl.mem table name then
        invalid_argf "Hf.weights: duplicate tensor %S" name;
      Hashtbl.add table name tensor)
    tensors;
  { tensors = table; used = Hashtbl.create (List.length tensors) }

let tensor t ~name ~shape =
  let tensor =
    match Hashtbl.find_opt t.tensors name with
    | Some tensor -> tensor
    | None -> invalid_argf "Hf.weights: missing tensor %S" name
  in
  let actual = Kaun.Ptree.Tensor.shape tensor in
  if actual <> shape then
    invalid_argf "Hf.weights: tensor %S has shape [%s], expected [%s]" name
      (String.concat "; " (List.map string_of_int (Array.to_list actual)))
      (String.concat "; " (List.map string_of_int (Array.to_list shape)));
  Hashtbl.replace t.used name ();
  tensor

let cast dtype (Kaun.Ptree.P tensor) = Kaun.Ptree.tensor (Nx.cast dtype tensor)

let matrix t dtype ~name ~rows ~cols =
  let (Kaun.Ptree.P tensor) = tensor t ~name ~shape:[| rows; cols |] in
  Nx.cast dtype tensor |> Nx.transpose ~axes:[ 1; 0 ] |> Kaun.Ptree.tensor

let ensure_consumed ?(allow = fun _ -> false) t =
  Hashtbl.iter
    (fun name _ ->
      if (not (Hashtbl.mem t.used name)) && not (allow name) then
        invalid_argf "Hf.weights: unexpected tensor %S" name)
    t.tensors
