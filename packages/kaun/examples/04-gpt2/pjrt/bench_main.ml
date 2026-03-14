open Kaun

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type stats = {
  first_ms : float;
  mean_ms : float;
  median_ms : float;
  min_ms : float;
  max_ms : float;
}

let backend_of_env () =
  match Sys.getenv_opt "RUNE_PJRT_BACKEND" with
  | Some "cpu" -> `Cpu
  | Some "cuda" | Some "gpu" -> `Cuda
  | Some backend ->
      invalid_argf
        "RUNE_PJRT_BACKEND=%S is invalid, expected cpu or cuda" backend
  | None ->
      if Rune_pjrt.backend_available `Cuda then `Cuda else `Cpu

let int_of_env name ~default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value -> (
      match int_of_string_opt value with
      | Some value -> value
      | None -> invalid_argf "%s=%S is not an integer" name value)

let json_escape s =
  let buf = Buffer.create (String.length s + 16) in
  String.iter
    (function
      | '"' -> Buffer.add_string buf "\\\""
      | '\\' -> Buffer.add_string buf "\\\\"
      | '\n' -> Buffer.add_string buf "\\n"
      | '\r' -> Buffer.add_string buf "\\r"
      | '\t' -> Buffer.add_string buf "\\t"
      | c -> Buffer.add_char buf c)
    s;
  Buffer.contents buf

let time_call f x =
  let start = Unix.gettimeofday () in
  let y = f x in
  let stop = Unix.gettimeofday () in
  ((stop -. start) *. 1000.0, y)

let summarize_samples ~first_ms samples =
  let sorted = Array.copy samples in
  Array.sort Float.compare sorted;
  let n = Array.length sorted in
  let median_ms =
    if n = 0 then first_ms
    else if n mod 2 = 1 then sorted.(n / 2)
    else (sorted.((n / 2) - 1) +. sorted.(n / 2)) /. 2.0
  in
  let total = Array.fold_left ( +. ) 0.0 sorted in
  let mean_ms = if n = 0 then first_ms else total /. float_of_int n in
  let min_ms = if n = 0 then first_ms else sorted.(0) in
  let max_ms = if n = 0 then first_ms else sorted.(n - 1) in
  { first_ms; mean_ms; median_ms; min_ms; max_ms }

let bench ~warmup ~iterations f x =
  if warmup < 0 then invalid_arg "warmup must be >= 0";
  if iterations <= 0 then invalid_arg "iterations must be > 0";
  let first_ms, first_result = time_call f x in
  for _ = 1 to warmup do
    ignore (f x)
  done;
  let results = Array.init iterations (fun _ -> time_call f x) in
  let samples = Array.map fst results in
  let last_result = snd results.(iterations - 1) in
  (summarize_samples ~first_ms samples, first_result, last_result)

let stats_to_json name stats =
  Printf.sprintf
    "\"%s\":{\"first_ms\":%.6f,\"mean_ms\":%.6f,\"median_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f}"
    name stats.first_ms stats.mean_ms stats.median_ms stats.min_ms stats.max_ms

let () =
  let model_id = "gpt2" in
  let dtype = Nx.float32 in
  let backend = backend_of_env () in
  let warmup = int_of_env "RUNE_PJRT_BENCH_WARMUP" ~default:1 in
  let iterations = int_of_env "RUNE_PJRT_BENCH_ITERS" ~default:1 in
  let prompt_tokens = int_of_env "RUNE_PJRT_BENCH_PROMPT_TOKENS" ~default:2048 in
  let max_tokens = int_of_env "RUNE_PJRT_BENCH_MAX_TOKENS" ~default:512 in
  if not (Rune_pjrt.backend_available backend) then
    invalid_argf "PJRT %s backend is unavailable (%s)"
      (Rune_pjrt.Backend.to_string backend)
      (Rune_pjrt.status ());

  let cfg, params = Gpt2.from_pretrained ~model_id () in
  let max_seq = prompt_tokens + max_tokens in
  let prompt_position_ids =
    Gpt2_bench_support.wrapped_position_ids ~n_positions:cfg.n_positions ~batch:1
      ~seq:prompt_tokens
  in
  let generate_position_ids =
    Gpt2_bench_support.wrapped_position_ids ~n_positions:cfg.n_positions ~batch:1
      ~seq:max_seq
  in
  let forward_prompt input_ids =
    Gpt2_bench_support.forward_with_position_ids ~cfg ~params ~dtype
      ~training:false ~position_ids:prompt_position_ids input_ids
  in
  let forward_generate input_ids =
    Gpt2_bench_support.forward_with_position_ids ~cfg ~params ~dtype
      ~training:false ~position_ids:generate_position_ids input_ids
  in
  let prefill_next_token input_ids =
    let logits = forward_prompt input_ids in
    let row = Nx.slice [ I 0; I (prompt_tokens - 1) ] logits in
    Nx.argmax ~axis:0 row
  in
  let prefill = Rune_pjrt.jit ~backend prefill_next_token in
  let generate =
    Rune_pjrt.Causal_lm.greedy_decode ~backend ~max_tokens forward_generate
  in
  let prompt_ids =
    Gpt2_bench_support.make_prompt_ids ~vocab_size:cfg.vocab_size ~prompt_tokens
  in
  let input_ids = Nx.create Nx.int32 [| 1; prompt_tokens |] prompt_ids in
  let prefill_stats, _, prefill_token = bench ~warmup ~iterations prefill input_ids in
  let decode_stats, _, generated = bench ~warmup ~iterations generate input_ids in
  let generated_ids = Nx.to_array generated in
  let prefill_token : int = Nx.item [] prefill_token |> Int32.to_int in
  let tail_sum =
    Array.sub generated_ids prompt_tokens (Array.length generated_ids - prompt_tokens)
    |> Array.fold_left (fun acc id -> acc + Int32.to_int id) 0
  in
  let prefill_tokens_per_s =
    float_of_int prompt_tokens /. (prefill_stats.mean_ms /. 1000.0)
  in
  let decode_tokens_per_s =
    float_of_int max_tokens /. (decode_stats.mean_ms /. 1000.0)
  in
  Printf.printf
    "{\
     \"backend\":\"%s\",\
     \"model_id\":\"%s\",\
     \"prompt_tokens\":%d,\
     \"max_tokens\":%d,\
     \"max_seq\":%d,\
     \"position_mode\":\"wrapped_mod_n_positions\",\
     \"warmup\":%d,\
     \"iterations\":%d,\
     %s,\
     %s,\
     \"prefill_token\":%d,\
     \"prefill_tokens_per_s\":%.6f,\
     \"decode_tokens_per_s\":%.6f,\
     \"generated_tail_sum\":%d\
     }\n"
    (Rune_pjrt.Backend.to_string backend)
    model_id prompt_tokens max_tokens max_seq warmup iterations
    (stats_to_json "prefill" prefill_stats)
    (stats_to_json "decode" decode_stats)
    prefill_token prefill_tokens_per_s decode_tokens_per_s tail_sum
