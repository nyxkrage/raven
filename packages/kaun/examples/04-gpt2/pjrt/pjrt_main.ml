(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let load_tokenizer model_id =
  let vocab = Kaun_hf.download_file ~model_id ~filename:"vocab.json" () in
  let merges = Kaun_hf.download_file ~model_id ~filename:"merges.txt" () in
  Brot.from_model_file ~vocab ~merges
    ~pre:
      (Brot.Pre_tokenizer.byte_level ~add_prefix_space:false ~use_regex:true ())
    ~decoder:(Brot.Decoder.byte_level ())
    ()

let encode tokenizer text =
  Array.map Int32.of_int (Brot.encode_ids tokenizer text)

let decode tokenizer ids = Brot.decode tokenizer (Array.map Int32.to_int ids)

let backend_of_env () =
  match Sys.getenv_opt "RUNE_PJRT_BACKEND" with
  | Some "cpu" -> `Cpu
  | Some "cuda" | Some "gpu" -> `Cuda
  | Some backend ->
      invalid_argf
        "RUNE_PJRT_BACKEND=%S is invalid, expected one of: cpu, cuda" backend
  | None ->
      if Rune_pjrt.backend_available `Cuda then `Cuda else `Cpu

let int_of_env name ~default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value -> int_of_string value

let print_top_k ~k forward input_ids ~pos =
  let logits = forward input_ids in
  let row = Nx.slice [ I 0; I pos ] logits in
  let sorted = Nx.argsort ~descending:true ~axis:0 row in
  let probs = Nx.softmax ~axes:[ 0 ] row in
  for i = 0 to k - 1 do
    let idx = Int32.to_int (Nx.item [ i ] sorted) in
    let prob : float = Nx.item [ idx ] probs in
    Printf.printf "    #%d  token %-6d  p=%.4f\n" (i + 1) idx prob
  done

let () =
  let model_id = "gpt2" in
  let dtype = Nx.float32 in
  let backend = backend_of_env () in
  let max_tokens = int_of_env "RUNE_PJRT_MAX_TOKENS" ~default:30 in
  if not (Rune_pjrt.backend_available backend) then
    invalid_argf "PJRT %s backend is not available (%s)"
      (Rune_pjrt.Backend.to_string backend)
      (Rune_pjrt.status ());

  Printf.printf "Loading %s for PJRT backend %s...\n%!" model_id
    (Rune_pjrt.Backend.to_string backend);
  let tokenizer = load_tokenizer model_id in
  let cfg, params = Gpt2.from_pretrained ~model_id () in
  Printf.printf "  vocab=%d  n_embd=%d  layers=%d  heads=%d\n\n" cfg.vocab_size
    cfg.n_embd cfg.n_layer cfg.n_head;

  let model = Gpt2.for_causal_lm cfg () in
  let vars = Layer.make_vars ~params ~state:Ptree.empty ~dtype in
  let forward_fn input_ids =
    let logits, _ = Layer.apply model vars ~training:false input_ids in
    logits
  in
  let forward = Rune_pjrt.jit ~backend forward_fn in
  let generate =
    Rune_pjrt.Causal_lm.greedy_decode ~backend ~max_tokens forward_fn
  in

  Printf.printf "=== Next-token predictions ===\n";
  Printf.printf "  Prompt: \"Hello world\"\n";
  Printf.printf "  Top 5 continuations:\n";
  let hello_ids = encode tokenizer "Hello world" in
  let hello = Nx.create Nx.int32 [| 1; Array.length hello_ids |] hello_ids in
  print_top_k ~k:5 forward hello ~pos:(Array.length hello_ids - 1);

  Printf.printf "\n=== Greedy generation (%d tokens each) ===\n\n" max_tokens;
  let prompts =
    [ "The meaning of life is"; "Once upon a time"; "The quick brown fox" ]
  in
  List.iter
    (fun text ->
      let prompt = encode tokenizer text in
      let prompt_ids = Nx.create Nx.int32 [| 1; Array.length prompt |] prompt in
      let generated = Nx.to_array (generate prompt_ids) in
      let continuation =
        Array.sub generated (Array.length prompt)
          (Array.length generated - Array.length prompt)
      in
      Printf.printf "  \"%s\" ->\n" text;
      Printf.printf "    %s\n\n" (decode tokenizer continuation))
    prompts
