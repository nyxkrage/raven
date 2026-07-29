(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type 'layout t = {
  layer_count : int;
  dtype : (float, 'layout) Nx.dtype;
  device_id : int;
  execute : Rune_pjrt.packed list -> Rune_pjrt.packed list;
  execute_device :
    Rune_pjrt.Device_buffer.packed list -> Rune_pjrt.Device_buffer.packed list;
  true_masks :
    (int * int, (bool, Nx.bool_elt) Rune_pjrt.Device_buffer.t) Hashtbl.t;
}

let pack tensor = Rune_pjrt.Tensor tensor
let pack_device buffer = Rune_pjrt.Device_buffer.Pack buffer

let unpack (type a b) (dtype : (a, b) Nx.dtype) ~name = function
  | Rune_pjrt.Tensor tensor -> (
      match Nx_core.Dtype.equal_witness dtype (Nx.dtype tensor) with
      | Some Type.Equal -> (tensor : (a, b) Nx.t)
      | None ->
          invalid_argf "Dense_pjrt: %s has dtype %s, expected %s" name
            (Nx_core.Dtype.to_string (Nx.dtype tensor))
            (Nx_core.Dtype.to_string dtype))

let unpack_device (type a b) (dtype : (a, b) Nx.dtype) ~name = function
  | Rune_pjrt.Device_buffer.Pack buffer -> (
      match
        Nx_core.Dtype.equal_witness dtype (Rune_pjrt.Device_buffer.dtype buffer)
      with
      | Some Type.Equal -> (buffer : (a, b) Rune_pjrt.Device_buffer.t)
      | None ->
          invalid_argf "Dense_pjrt: %s has dtype %s, expected %s" name
            (Nx_core.Dtype.to_string (Rune_pjrt.Device_buffer.dtype buffer))
            (Nx_core.Dtype.to_string dtype))

let compile ?(device_id = 0) ~layer_count ~dtype forward =
  if layer_count <= 0 then
    invalid_argf "Dense_pjrt.compile: layer_count must be positive, got %d"
      layer_count;
  let traced inputs =
    let inputs = Array.of_list inputs in
    let expected = 4 + (2 * layer_count) in
    if Array.length inputs <> expected then
      invalid_argf "Dense_pjrt: expected %d inputs, got %d" expected
        (Array.length inputs);
    let input_ids = unpack Nx.int32 ~name:"input_ids" inputs.(0) in
    let attention_mask = unpack Nx.bool ~name:"attention_mask" inputs.(1) in
    let position = unpack Nx.int32 ~name:"position" inputs.(2) in
    let valid = unpack Nx.bool ~name:"valid" inputs.(3) in
    let keys =
      Array.init layer_count (fun index ->
          unpack dtype ~name:(Printf.sprintf "keys.%d" index) inputs.(4 + index))
    in
    let values =
      Array.init layer_count (fun index ->
          unpack dtype
            ~name:(Printf.sprintf "values.%d" index)
            inputs.(4 + layer_count + index))
    in
    let input_shape = Nx.shape input_ids in
    let cache =
      {
        Dense_cache.keys;
        values;
        valid;
        position;
        batch_size = input_shape.(0);
        max_length = (Nx.shape valid).(1);
        length = 0;
      }
    in
    let logits, cache = forward ~attention_mask cache input_ids in
    let cache : _ Dense_cache.t = cache in
    pack logits :: pack cache.position :: pack cache.valid
    :: (Array.to_list cache.keys |> List.map pack)
    @ (Array.to_list cache.values |> List.map pack)
  in
  let execute = Rune_pjrt.jits_packed ~backend:`Cuda ~device_id traced in
  let execute_device =
    Rune_pjrt.jits_device_packed ~backend:`Cuda ~device_id traced
  in
  {
    layer_count;
    dtype;
    device_id;
    execute;
    execute_device;
    true_masks = Hashtbl.create 4;
  }

let prefill runner cache ?attention_mask input_ids =
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 then
    invalid_argf
      "Dense_pjrt.prefill: expected input IDs with shape [batch; seq], got [%s]"
      (String.concat "; " (List.map string_of_int (Array.to_list shape)));
  let batch = shape.(0) in
  let seq = shape.(1) in
  if batch <> cache.Dense_cache.batch_size then
    invalid_argf "Dense_pjrt.prefill: expected batch size %d, got %d"
      cache.batch_size batch;
  if seq <= 0 then
    invalid_argf "Dense_pjrt.prefill: sequence length must be positive";
  if cache.length + seq > cache.max_length then
    invalid_argf
      "Dense_pjrt.prefill: cache capacity %d exceeded by position %d + %d"
      cache.max_length cache.length seq;
  let attention_mask =
    match attention_mask with
    | None -> Nx.ones Nx.bool [| batch; seq |]
    | Some mask ->
        if Nx.shape mask <> [| batch; seq |] then
          invalid_argf
            "Dense_pjrt.prefill: attention mask must have shape [%d; %d]" batch
            seq;
        mask
  in
  let inputs =
    pack input_ids :: pack attention_mask :: pack cache.position
    :: pack cache.valid
    :: (Array.to_list cache.keys |> List.map pack)
    @ (Array.to_list cache.values |> List.map pack)
  in
  let outputs = Array.of_list (runner.execute inputs) in
  let expected = 3 + (2 * runner.layer_count) in
  if Array.length outputs <> expected then
    invalid_argf "Dense_pjrt: expected %d outputs, got %d" expected
      (Array.length outputs);
  let logits = unpack runner.dtype ~name:"logits" outputs.(0) in
  let position = unpack Nx.int32 ~name:"position" outputs.(1) in
  let valid = unpack Nx.bool ~name:"valid" outputs.(2) in
  let keys =
    Array.init runner.layer_count (fun index ->
        unpack runner.dtype
          ~name:(Printf.sprintf "keys.%d" index)
          outputs.(3 + index))
  in
  let values =
    Array.init runner.layer_count (fun index ->
        unpack runner.dtype
          ~name:(Printf.sprintf "values.%d" index)
          outputs.(3 + runner.layer_count + index))
  in
  ( logits,
    {
      cache with
      Dense_cache.keys;
      values;
      valid;
      position;
      length = cache.length + seq;
    } )

let decode_step runner cache input_ids =
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 || shape.(1) <> 1 then
    invalid_argf
      "Dense_pjrt.decode_step: expected input IDs with shape [batch; 1], got \
       [%s]"
      (String.concat "; " (List.map string_of_int (Array.to_list shape)));
  prefill runner cache input_ids

type 'layout resident = {
  keys : (float, 'layout) Rune_pjrt.Device_buffer.t array;
  values : (float, 'layout) Rune_pjrt.Device_buffer.t array;
  valid : (bool, Nx.bool_elt) Rune_pjrt.Device_buffer.t;
  position : (int32, Nx.int32_elt) Rune_pjrt.Device_buffer.t;
  batch_size : int;
  max_length : int;
  length : int;
}

let resident_length cache = cache.length

let resident_of_host runner cache =
  if Array.length cache.Dense_cache.keys <> runner.layer_count then
    invalid_argf "Dense_pjrt.Resident.of_host: expected %d key caches, got %d"
      runner.layer_count (Array.length cache.keys);
  if Array.length cache.values <> runner.layer_count then
    invalid_argf "Dense_pjrt.Resident.of_host: expected %d value caches, got %d"
      runner.layer_count
      (Array.length cache.values);
  let upload tensor =
    Rune_pjrt.Device_buffer.of_host ~backend:`Cuda ~device_id:runner.device_id
      tensor
  in
  {
    keys = Array.map upload cache.keys;
    values = Array.map upload cache.values;
    valid = upload cache.valid;
    position = upload cache.position;
    batch_size = cache.batch_size;
    max_length = cache.max_length;
    length = cache.length;
  }

let resident_prefill runner cache ?attention_mask input_ids =
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 then
    invalid_argf
      "Dense_pjrt.Resident.prefill: expected input IDs with shape [batch; \
       seq], got [%s]"
      (String.concat "; " (List.map string_of_int (Array.to_list shape)));
  let batch = shape.(0) in
  let seq = shape.(1) in
  if batch <> cache.batch_size then
    invalid_argf "Dense_pjrt.Resident.prefill: expected batch size %d, got %d"
      cache.batch_size batch;
  if seq <= 0 then
    invalid_argf "Dense_pjrt.Resident.prefill: sequence length must be positive";
  if cache.length + seq > cache.max_length then
    invalid_argf
      "Dense_pjrt.Resident.prefill: cache capacity %d exceeded by position %d \
       + %d"
      cache.max_length cache.length seq;
  let upload tensor =
    Rune_pjrt.Device_buffer.of_host ~backend:`Cuda ~device_id:runner.device_id
      tensor
  in
  let attention_mask =
    match attention_mask with
    | None -> (
        let key = (batch, seq) in
        match Hashtbl.find_opt runner.true_masks key with
        | Some mask -> mask
        | None ->
            let mask = upload (Nx.ones Nx.bool [| batch; seq |]) in
            Hashtbl.add runner.true_masks key mask;
            mask)
    | Some mask ->
        if Nx.shape mask <> [| batch; seq |] then
          invalid_argf
            "Dense_pjrt.Resident.prefill: attention mask must have shape [%d; \
             %d]"
            batch seq;
        upload mask
  in
  let inputs =
    pack_device (upload input_ids)
    :: pack_device attention_mask :: pack_device cache.position
    :: pack_device cache.valid
    :: (Array.to_list cache.keys |> List.map pack_device)
    @ (Array.to_list cache.values |> List.map pack_device)
  in
  let outputs = Array.of_list (runner.execute_device inputs) in
  let expected = 3 + (2 * runner.layer_count) in
  if Array.length outputs <> expected then
    invalid_argf "Dense_pjrt.Resident: expected %d outputs, got %d" expected
      (Array.length outputs);
  let logits =
    unpack_device runner.dtype ~name:"logits" outputs.(0)
    |> Rune_pjrt.Device_buffer.to_host
  in
  let position = unpack_device Nx.int32 ~name:"position" outputs.(1) in
  let valid = unpack_device Nx.bool ~name:"valid" outputs.(2) in
  let keys =
    Array.init runner.layer_count (fun index ->
        unpack_device runner.dtype
          ~name:(Printf.sprintf "keys.%d" index)
          outputs.(3 + index))
  in
  let values =
    Array.init runner.layer_count (fun index ->
        unpack_device runner.dtype
          ~name:(Printf.sprintf "values.%d" index)
          outputs.(3 + runner.layer_count + index))
  in
  ( logits,
    { cache with keys; values; valid; position; length = cache.length + seq } )

let resident_decode_step runner cache input_ids =
  let shape = Nx.shape input_ids in
  if Array.length shape <> 2 || shape.(1) <> 1 then
    invalid_argf
      "Dense_pjrt.Resident.decode_step: expected input IDs with shape [batch; \
       1], got [%s]"
      (String.concat "; " (List.map string_of_int (Array.to_list shape)));
  resident_prefill runner cache input_ids
