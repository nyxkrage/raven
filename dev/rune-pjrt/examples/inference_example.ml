type result = {
  prompt : int32 array;
  generated : int32 array;
  expected : int32 array;
}

let vocab_size = 6

let prompt_ids = [| 1l; 2l; 3l |]

let toy_embedding = Nx.eye Nx.float32 vocab_size

let shift_projection =
  Nx.create Nx.float32 [| vocab_size; vocab_size |]
    [|
      0.; 1.; 0.; 0.; 0.; 0.;
      0.; 0.; 1.; 0.; 0.; 0.;
      0.; 0.; 0.; 1.; 0.; 0.;
      0.; 0.; 0.; 0.; 1.; 0.;
      0.; 0.; 0.; 0.; 0.; 1.;
      1.; 0.; 0.; 0.; 0.; 0.;
    |]

let toy_lm input_ids =
  let shape = Nx.shape input_ids in
  let batch = shape.(0) in
  let seq = shape.(1) in
  let flat_ids = Nx.reshape [| batch * seq |] input_ids in
  let hidden =
    Nx.take ~axis:0 flat_ids toy_embedding
    |> Nx.reshape [| batch; seq; vocab_size |]
  in
  Nx.matmul hidden shift_projection

let expected_tokens ~max_tokens =
  let total = Array.length prompt_ids + max_tokens in
  let full = Array.make total 0l in
  Array.blit prompt_ids 0 full 0 (Array.length prompt_ids);
  for i = Array.length prompt_ids to total - 1 do
    let prev = Int32.to_int full.(i - 1) in
    full.(i) <- Int32.of_int ((prev + 1) mod vocab_size)
  done;
  full

let run ?(backend = `Cpu) ?(device_id = 0) ?(max_tokens = 4) () =
  Support.require_backend backend;
  let input_ids = Nx.create Nx.int32 [| 1; Array.length prompt_ids |] prompt_ids in
  let generated =
    Rune_pjrt.Causal_lm.greedy_decode ~backend ~device_id ~max_tokens toy_lm
      input_ids
    |> Support.int32_array
  in
  let expected = expected_tokens ~max_tokens in
  { prompt = Array.copy prompt_ids; generated; expected }

let validate result =
  if result.generated <> result.expected then
    failwith
      (Printf.sprintf
         "inference example mismatch\nexpected: [%s]\nactual:   [%s]"
         (Support.pp_i32_array result.expected)
         (Support.pp_i32_array result.generated))
