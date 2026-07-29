(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let layer_norm x =
  let gamma = Nx.ones Nx.float32 [| 8 |] in
  let beta = Nx.zeros Nx.float32 [| 8 |] in
  let mean = Nx.mean x ~axes:[ -1 ] ~keepdims:true in
  let centered = Nx.sub x mean in
  let variance =
    Nx.mean (Nx.mul centered centered) ~axes:[ -1 ] ~keepdims:true
  in
  let normalized =
    Nx.mul centered (Nx.rsqrt (Nx.add variance (Nx.scalar_like variance 1e-5)))
  in
  Nx.add (Nx.mul normalized gamma) beta

let embedding weights indices =
  let indices_shape = Nx.shape indices in
  let flattened = Nx.reshape [| Nx.numel indices |] indices in
  let gathered = Nx.take ~axis:0 flattened weights in
  Nx.reshape (Array.append indices_shape [| (Nx.shape weights).(1) |]) gathered

let transpose_last_two tensor =
  let rank = Nx.ndim tensor in
  let axes = Array.init rank Fun.id in
  let previous = axes.(rank - 2) in
  axes.(rank - 2) <- axes.(rank - 1);
  axes.(rank - 1) <- previous;
  Nx.transpose tensor ~axes:(Array.to_list axes)

let causal_attention q k v =
  let depth = (Nx.shape q).(Nx.ndim q - 1) in
  let scores =
    Nx.matmul q (transpose_last_two k)
    |> Nx.mul (Nx.scalar Nx.float32 (1.0 /. Stdlib.sqrt (float_of_int depth)))
  in
  let shape = Nx.shape scores in
  let sequence_length = shape.(Array.length shape - 1) in
  let mask =
    Nx.ones Nx.float32 [| sequence_length; sequence_length |]
    |> Nx.tril |> Nx.cast Nx.bool |> Nx.broadcast_to shape
  in
  let scores = Nx.where mask scores (Nx.scalar_like scores (-1e9)) in
  Nx.matmul (Nx.softmax ~axes:[ -1 ] scores) v

let gelu_approx x =
  let one = Nx.scalar_like x 1.0 in
  let half = Nx.scalar_like x 0.5 in
  let sqrt_two_over_pi = Nx.scalar_like x 0.7978845608 in
  let coefficient = Nx.scalar_like x 0.044715 in
  let argument =
    Nx.mul
      (Nx.mul x sqrt_two_over_pi)
      (Nx.add one (Nx.mul coefficient (Nx.mul x x)))
  in
  Nx.mul half (Nx.mul x (Nx.add one (Nx.tanh argument)))

let skip_cuda () = Sys.getenv_opt "RUNE_PJRT_TEST_SKIP_CUDA" <> None

let backend_available backend =
  match backend with
  | `Cuda when skip_cuda () -> false
  | (`Cpu | `Cuda) as backend -> Rune_pjrt.Runtime.backend_available backend

let small_tensor rows cols scale =
  Nx.create Nx.float32 [| rows; cols |]
    (Array.init (rows * cols) (fun i ->
         scale *. float_of_int ((i mod cols) + 1)))

let small_vector n scale =
  Nx.create Nx.float32 [| n |]
    (Array.init n (fun i -> scale *. float_of_int (i + 1)))

let tiny_gpt2_like input_ids =
  let shape = Nx.shape input_ids in
  let batch = shape.(0) in
  let seq = shape.(1) in
  let vocab = 32 in
  let embed_dim = 8 in
  let heads = 2 in
  let head_dim = embed_dim / heads in
  let inner_dim = 16 in
  let wte = small_tensor vocab embed_dim 0.01 in
  let wpe = small_tensor 16 embed_dim 0.02 in
  let position_ids =
    Nx.arange_f Nx.float32 0.0 (float_of_int seq) 1.0
    |> Nx.cast Nx.int32
    |> Nx.reshape [| 1; seq |]
    |> Nx.broadcast_to [| batch; seq |]
    |> Nx.contiguous
  in
  let tok = embedding wte input_ids in
  let pos = embedding wpe position_ids in
  let x = Nx.add tok pos in
  let qkv_w = small_tensor embed_dim (3 * embed_dim) 0.005 in
  let qkv_b = small_vector (3 * embed_dim) 0.001 in
  let qkv = Nx.add (Nx.matmul x qkv_w) qkv_b in
  let qkv_parts = Nx.split ~axis:(-1) 3 qkv in
  let split_heads t =
    Nx.reshape [| batch; seq; heads; head_dim |] t
    |> Nx.transpose ~axes:[ 0; 2; 1; 3 ]
  in
  let q = split_heads (layer_norm (List.nth qkv_parts 0)) in
  let k = split_heads (layer_norm (List.nth qkv_parts 1)) in
  let v = split_heads (layer_norm (List.nth qkv_parts 2)) in
  let attn = causal_attention q k v in
  let merged =
    Nx.transpose attn ~axes:[ 0; 2; 1; 3 ]
    |> Nx.contiguous
    |> Nx.reshape [| batch; seq; embed_dim |]
  in
  let o_w = small_tensor embed_dim embed_dim 0.004 in
  let o_b = small_vector embed_dim 0.001 in
  let x = Nx.add x (Nx.add (Nx.matmul merged o_w) o_b) in
  let x' = layer_norm x in
  let ffn_up_w = small_tensor embed_dim inner_dim 0.003 in
  let ffn_up_b = small_vector inner_dim 0.001 in
  let ffn_down_w = small_tensor inner_dim embed_dim 0.002 in
  let ffn_down_b = small_vector embed_dim 0.001 in
  let y = Nx.add (Nx.matmul x' ffn_up_w) ffn_up_b |> gelu_approx in
  let hidden =
    Nx.add x (Nx.add (Nx.matmul y ffn_down_w) ffn_down_b) |> layer_norm
  in
  Nx.matmul hidden (Nx.transpose wte ~axes:[ 1; 0 ])

let contains_substring haystack needle =
  let hay_len = String.length haystack in
  let needle_len = String.length needle in
  let rec loop i =
    if i + needle_len > hay_len then false
    else if String.sub haystack i needle_len = needle then true
    else loop (i + 1)
  in
  loop 0

let check_close name expected actual =
  let exp = Nx.to_array expected in
  let got = Nx.to_array actual in
  if Array.length exp <> Array.length got then
    failwith (Printf.sprintf "%s: length mismatch" name);
  Array.iteri
    (fun i x ->
      let y = got.(i) in
      if Float.abs (x -. y) > 1e-4 then
        failwith
          (Printf.sprintf "%s: mismatch at %d expected=%g got=%g" name i x y))
    exp

let check_equal_i32 name expected actual =
  let exp = Nx.to_array expected in
  let got = Nx.to_array actual in
  if exp <> got then failwith (Printf.sprintf "%s: token mismatch" name)

let test_large_closed_over_constants_are_parameterized () =
  let rows = 1024 in
  let cols = 256 in
  let weights =
    Nx.create Nx.float32 [| rows; cols |]
      (Array.init (rows * cols) (fun i -> float_of_int (i mod 97) *. 0.001))
  in
  let input =
    Nx.create Nx.float32 [| 1; rows |]
      (Array.init rows (fun i -> float_of_int ((i mod 31) + 1) *. 0.01))
  in
  let capture =
    Rune_pjrt.Trace.capture_one (fun x -> Nx.matmul x weights) input
  in
  let program, lifted = Rune_pjrt.Ir.parameterize_constants capture.program in
  if lifted = [] then
    failwith "test_gpt2_shape: expected large closure constant to be lifted";
  let module_text = Rune_pjrt.Stablehlo.of_program program in
  if String.length module_text > 1_000_000 then
    failwith
      (Printf.sprintf
         "test_gpt2_shape: parameterized module still too large (%d bytes)"
         (String.length module_text))

let greedy_decode_eager ~max_tokens forward input_ids =
  let tokens = ref (Array.to_list (Nx.to_array input_ids)) in
  for _ = 1 to max_tokens do
    let ids = Array.of_list !tokens in
    let input = Nx.create Nx.int32 [| 1; Array.length ids |] ids in
    let logits = forward input in
    let last = Nx.slice [ I 0; I (Array.length ids - 1) ] logits in
    let next : int32 = Nx.item [] (Nx.argmax ~axis:0 last) in
    tokens := !tokens @ [ next ]
  done;
  let ids = Array.of_list !tokens in
  Nx.create Nx.int32 [| 1; Array.length ids |] ids

let test_gpt2_like_trace () =
  test_large_closed_over_constants_are_parameterized ();
  let input = Nx.create Nx.int32 [| 1; 4 |] [| 1l; 2l; 3l; 4l |] in
  let capture = Rune_pjrt.Trace.capture_one tiny_gpt2_like input in
  let unsupported = Rune_pjrt.Ir.unsupported_ops capture.program in
  if unsupported <> [] then
    failwith
      (Printf.sprintf "test_gpt2_shape: unsupported ops: %s"
         (String.concat ", " unsupported));
  if
    not
      (List.exists
         (fun node -> Rune_pjrt.Ir.op_name node.Rune_pjrt.Ir.op = "gather")
         capture.program.nodes)
  then failwith "test_gpt2_shape: missing gather op";
  if
    not
      (List.exists
         (fun node ->
           let name = Rune_pjrt.Ir.op_name node.Rune_pjrt.Ir.op in
           name = "reduce_sum" || name = "matmul")
         capture.program.nodes)
  then failwith "test_gpt2_shape: missing reduce_sum or matmul op";
  let module_text =
    let signature =
      Rune_pjrt.Signature.of_tensors ~backend:`Cpu ~device_id:0 [ input ]
    in
    try
      let compiled =
        Rune_pjrt.Runtime.compile ~backend:`Cpu ~device_id:0 ~signature
          capture.program capture.outputs
      in
      compiled.module_text
    with Rune_pjrt.Error.Error (Runtime_unavailable _) ->
      Rune_pjrt.Stablehlo.of_program capture.program
  in
  if not (contains_substring module_text "stablehlo.gather") then
    failwith "test_gpt2_shape: gather did not lower to StableHLO";
  if not (contains_substring module_text "stablehlo.dot_general") then
    failwith "test_gpt2_shape: matmul did not lower to StableHLO";
  if not (contains_substring module_text "stablehlo.reduce") then
    failwith "test_gpt2_shape: reductions did not lower to StableHLO";
  if backend_available `Cpu then (
    let expected = tiny_gpt2_like input in
    let actual = Rune_pjrt.jit ~backend:`Cpu tiny_gpt2_like input in
    check_close "gpt2_like_cpu_executes" expected actual;
    let expected_tokens =
      greedy_decode_eager ~max_tokens:3 tiny_gpt2_like input
    in
    let actual_tokens =
      Rune_pjrt.Causal_lm.greedy_decode ~backend:`Cpu ~max_tokens:3
        tiny_gpt2_like input
    in
    check_equal_i32 "gpt2_like_cpu_greedy_decode" expected_tokens actual_tokens);
  if backend_available `Cuda then (
    let expected = tiny_gpt2_like input in
    let actual = Rune_pjrt.jit ~backend:`Cuda tiny_gpt2_like input in
    check_close "gpt2_like_cuda_executes" expected actual;
    let expected_tokens =
      greedy_decode_eager ~max_tokens:3 tiny_gpt2_like input
    in
    let actual_tokens =
      Rune_pjrt.Causal_lm.greedy_decode ~backend:`Cuda ~max_tokens:3
        tiny_gpt2_like input
    in
    check_equal_i32 "gpt2_like_cuda_greedy_decode" expected_tokens actual_tokens)

let () =
  try test_gpt2_like_trace ()
  with Rune_pjrt.Error.Error err ->
    failwith ("test_gpt2_shape: " ^ Rune_pjrt.Error.to_string err)
