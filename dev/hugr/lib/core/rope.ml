(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let position_ids_key = "position_ids"

let default_positions ~batch ~seq =
  Nx.arange Nx.int32 0 seq 1
  |> Nx.reshape [| 1; seq |]
  |> Nx.broadcast_to [| batch; seq |]
  |> Nx.contiguous

let positions_from_context ~batch ~seq = function
  | None -> default_positions ~batch ~seq
  | Some ctx -> (
      match Kaun.Context.find ctx ~name:position_ids_key with
      | None -> default_positions ~batch ~seq
      | Some tensor -> (
          let positions = Kaun.Ptree.Tensor.to_typed_exn Nx.int32 tensor in
          let shape = Nx.shape positions in
          match Array.to_list shape with
          | [ positions_seq ] when positions_seq = seq ->
              Nx.reshape [| 1; seq |] positions
              |> Nx.broadcast_to [| batch; seq |]
              |> Nx.contiguous
          | [ positions_batch; positions_seq ]
            when positions_batch = batch && positions_seq = seq ->
              positions
          | _ ->
              invalid_argf
                "Rope.position_ids: expected shape [%d] or [%d; %d], got [%s]"
                seq batch seq
                (String.concat "; "
                   (List.map string_of_int (Array.to_list shape)))))

let apply (type layout) ~theta ~rotary_dim ~positions (x : (float, layout) Nx.t)
    : (float, layout) Nx.t =
  if theta <= 0.0 then invalid_argf "Rope.apply: theta must be positive";
  let shape = Nx.shape x in
  if Array.length shape <> 4 then
    invalid_argf "Rope.apply: expected rank 4 [batch; heads; seq; head_dim]";
  let batch = shape.(0) in
  let seq = shape.(2) in
  let head_dim = shape.(3) in
  if rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim mod 2 <> 0 then
    invalid_argf
      "Rope.apply: rotary_dim must be positive, even, and <= head_dim; got %d \
       and head_dim=%d"
      rotary_dim head_dim;
  let positions_shape = Nx.shape positions in
  if positions_shape <> [| batch; seq |] then
    invalid_argf "Rope.apply: positions must have shape [%d; %d]" batch seq;
  let dtype = Nx.dtype x in
  let half = rotary_dim / 2 in
  let trig (type trig_layout) (trig_dtype : (float, trig_layout) Nx.dtype) =
    let exponents =
      Nx.arange_f trig_dtype 0.0 (float_of_int rotary_dim) 2.0 |> fun values ->
      Nx.div values (Nx.scalar trig_dtype (float_of_int rotary_dim))
    in
    let inv_freq =
      Nx.exp (Nx.mul (Nx.neg exponents) (Nx.log (Nx.scalar trig_dtype theta)))
    in
    let positions = Nx.cast trig_dtype positions in
    let angles =
      Nx.mul
        (Nx.reshape [| batch; seq; 1 |] positions)
        (Nx.reshape [| 1; 1; half |] inv_freq)
      |> Nx.expand_dims [ 1 ]
    in
    (Nx.cast dtype (Nx.cos angles), Nx.cast dtype (Nx.sin angles))
  in
  let cos, sin =
    match dtype with Nx.Float64 -> trig Nx.float64 | _ -> trig Nx.float32
  in
  let x1 = Nx.slice [ Nx.A; Nx.A; Nx.A; Nx.R (0, half) ] x in
  let x2 = Nx.slice [ Nx.A; Nx.A; Nx.A; Nx.R (half, rotary_dim) ] x in
  let rotated =
    Nx.concatenate ~axis:(-1)
      [
        Nx.sub (Nx.mul x1 cos) (Nx.mul x2 sin);
        Nx.add (Nx.mul x1 sin) (Nx.mul x2 cos);
      ]
  in
  if rotary_dim = head_dim then rotated
  else
    let rest = Nx.slice [ Nx.A; Nx.A; Nx.A; Nx.R (rotary_dim, head_dim) ] x in
    Nx.concatenate ~axis:(-1) [ rotated; rest ]
