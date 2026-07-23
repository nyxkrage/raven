(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type 'layout t = {
  layer_count : int;
  dtype : (float, 'layout) Nx.dtype;
  execute : Rune_pjrt.packed list -> Rune_pjrt.packed list;
}

let pack tensor = Rune_pjrt.Tensor tensor

let unpack (type a b) (dtype : (a, b) Nx.dtype) ~name = function
  | Rune_pjrt.Tensor tensor -> (
      match Nx_core.Dtype.equal_witness dtype (Nx.dtype tensor) with
      | Some Type.Equal -> (tensor : (a, b) Nx.t)
      | None ->
          invalid_argf "Dense_pjrt: %s has dtype %s, expected %s" name
            (Nx_core.Dtype.to_string (Nx.dtype tensor))
            (Nx_core.Dtype.to_string dtype))

let compile ?(device_id = 0) ~layer_count ~dtype forward =
  if layer_count <= 0 then
    invalid_argf "Dense_pjrt.compile: layer_count must be positive, got %d"
      layer_count;
  let execute =
    Rune_pjrt.jits_packed ~backend:`Cuda ~device_id (fun inputs ->
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
              unpack dtype
                ~name:(Printf.sprintf "keys.%d" index)
                inputs.(4 + index))
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
        @ (Array.to_list cache.values |> List.map pack))
  in
  { layer_count; dtype; execute }

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
