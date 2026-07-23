(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let attention_mask_key = "attention_mask"
let boolean_and left right = Nx.where left right (Nx.scalar Nx.bool false)

let causal ?window ~batch ~query_len ~key_len () =
  if query_len <= 0 || key_len <= 0 || query_len > key_len then
    invalid_argf
      "Mask.causal: expected 0 < query_len <= key_len, got query_len=%d and \
       key_len=%d"
      query_len key_len;
  Option.iter
    (fun size ->
      if size <= 0 then
        invalid_argf "Mask.causal: window must be positive, got %d" size)
    window;
  let query_start = key_len - query_len in
  let query_positions =
    Nx.arange Nx.int32 query_start key_len 1
    |> Nx.reshape [| 1; 1; query_len; 1 |]
  in
  let key_positions =
    Nx.arange Nx.int32 0 key_len 1 |> Nx.reshape [| 1; 1; 1; key_len |]
  in
  let visible = Nx.less_equal key_positions query_positions in
  let visible =
    match window with
    | None -> visible
    | Some size ->
        let first_visible =
          Nx.sub query_positions (Nx.scalar Nx.int32 (Int32.of_int (size - 1)))
        in
        boolean_and visible (Nx.greater_equal key_positions first_visible)
  in
  Nx.broadcast_to [| batch; 1; query_len; key_len |] visible

let padding_from_context ~batch ~key_len = function
  | None -> None
  | Some ctx -> (
      match Kaun.Context.find ctx ~name:attention_mask_key with
      | None -> None
      | Some tensor ->
          let mask =
            match Kaun.Ptree.Tensor.to_typed Nx.bool tensor with
            | Some mask -> mask
            | None ->
                let (Kaun.Ptree.P mask) = tensor in
                let mask = Nx.cast Nx.int32 mask in
                Nx.not_equal mask (Nx.zeros Nx.int32 (Nx.shape mask))
          in
          let shape = Nx.shape mask in
          if shape <> [| batch; key_len |] then
            invalid_argf
              "Mask.attention_mask: expected shape [%d; %d], got [%s]" batch
              key_len
              (String.concat "; "
                 (List.map string_of_int (Array.to_list shape)));
          Some (Nx.reshape [| batch; 1; 1; key_len |] mask))

let combined ?window ~batch ~query_len ~key_len ctx =
  let causal = causal ?window ~batch ~query_len ~key_len () in
  match padding_from_context ~batch ~key_len ctx with
  | None -> causal
  | Some padding -> boolean_and causal padding
