(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let fields ~ctx t = Kaun.Ptree.Dict.fields_exn ~ctx t
let find ~ctx key fs = Kaun.Ptree.Dict.find_exn ~ctx key fs
let get fs ~name dtype = Kaun.Ptree.Dict.get_tensor_exn fs ~name dtype

let require_same_float_dtype (type p in_elt) ~ctx
    (expected : (float, p) Nx.dtype) (x : (float, in_elt) Nx.t) :
    (float, p) Nx.t =
  match Nx_core.Dtype.equal_witness expected (Nx.dtype x) with
  | Some Type.Equal -> (x : (float, p) Nx.t)
  | None ->
      invalid_arg
        (Printf.sprintf "%s: input dtype %s does not match model dtype %s" ctx
           (Nx_core.Dtype.to_string (Nx.dtype x))
           (Nx_core.Dtype.to_string expected))

let child_vars ~ctx ~params ~state name =
  let param_fields = fields ~ctx:(ctx ^ ".params") params in
  let state_fields = fields ~ctx:(ctx ^ ".state") state in
  ( find ~ctx:(ctx ^ ".params") name param_fields,
    find ~ctx:(ctx ^ ".state") name state_fields )

let init_children dtype children =
  let init_child (_, init) =
    Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () -> init ~dtype)
  in
  let vars = List.map init_child children in
  let names = List.map fst children in
  let params =
    List.map2 (fun name vars -> (name, Kaun.Layer.params vars)) names vars
  in
  let state =
    List.map2 (fun name vars -> (name, Kaun.Layer.state vars)) names vars
  in
  Kaun.Layer.make_vars ~params:(Kaun.Ptree.dict params)
    ~state:(Kaun.Ptree.dict state) ~dtype

let apply_child ~ctx layer ~name ~params ~state ~dtype ~training ?call_ctx x =
  let child_params, child_state = child_vars ~ctx ~params ~state name in
  layer.Kaun.Layer.apply ~params:child_params ~state:child_state ~dtype
    ~training ?ctx:call_ctx x

let merge_state ~names states =
  Kaun.Ptree.dict (List.map2 (fun name state -> (name, state)) names states)
