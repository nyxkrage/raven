(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type 'layout t = {
  keys : (float, 'layout) Nx.t array;
  values : (float, 'layout) Nx.t array;
  valid : Nx.bool_t;
  position : Nx.int32_t;
  batch_size : int;
  max_length : int;
  length : int;
}

let create ~num_layers ~num_kv_heads ~head_dim ~max_position_embeddings
    ~batch_size ~max_length ~dtype =
  if num_layers <= 0 then
    invalid_argf "Dense_cache.create: num_layers must be positive, got %d"
      num_layers;
  if num_kv_heads <= 0 then
    invalid_argf "Dense_cache.create: num_kv_heads must be positive, got %d"
      num_kv_heads;
  if head_dim <= 0 then
    invalid_argf "Dense_cache.create: head_dim must be positive, got %d"
      head_dim;
  if batch_size <= 0 then
    invalid_argf "Dense_cache.create: batch_size must be positive, got %d"
      batch_size;
  if max_length <= 0 || max_length > max_position_embeddings then
    invalid_argf "Dense_cache.create: max_length must be in [1, %d], got %d"
      max_position_embeddings max_length;
  let shape = [| batch_size; num_kv_heads; max_length; head_dim |] in
  {
    keys = Array.init num_layers (fun _ -> Nx.zeros dtype shape);
    values = Array.init num_layers (fun _ -> Nx.zeros dtype shape);
    valid = Nx.zeros Nx.bool [| batch_size; max_length |];
    position = Nx.scalar Nx.int32 0l;
    batch_size;
    max_length;
    length = 0;
  }

let batch_size t = t.batch_size
let max_length t = t.max_length
let length t = t.length

let append_valid t token_valid seq =
  let writes =
    Nx.equal
      (Nx.reshape [| 1; seq; 1 |]
         (Nx.add t.position (Nx.arange Nx.int32 0 seq 1)))
      (Nx.reshape [| 1; 1; t.max_length |]
         (Nx.arange Nx.int32 0 t.max_length 1))
  in
  let valid_writes =
    Nx.where (Nx.expand_dims [ 2 ] token_valid) writes (Nx.scalar Nx.bool false)
    |> Nx.cast Nx.int32 |> Nx.sum ~axes:[ 1 ]
    |> fun count -> Nx.greater count (Nx.scalar Nx.int32 0l)
  in
  Nx.where t.valid (Nx.scalar Nx.bool true) valid_writes
