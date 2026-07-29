(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Layer = Kaun.Layer
module Llama = Hugr.Llama
module Ptree = Kaun.Ptree
module Resident = Llama.Pjrt.Resident

type benchmark = Prefill | Decode

type dimensions = {
  vocab_size : int;
  hidden_size : int;
  intermediate_size : int;
  num_hidden_layers : int;
  num_attention_heads : int;
  num_key_value_heads : int;
  max_position_embeddings : int;
  rope_theta : float;
}

let llama3_1b =
  {
    vocab_size = 128_256;
    hidden_size = 2_048;
    intermediate_size = 8_192;
    num_hidden_layers = 16;
    num_attention_heads = 32;
    num_key_value_heads = 8;
    max_position_embeddings = 131_072;
    rope_theta = 500_000.0;
  }

let smoke =
  {
    vocab_size = 4_096;
    hidden_size = 256;
    intermediate_size = 768;
    num_hidden_layers = 2;
    num_attention_heads = 8;
    num_key_value_heads = 2;
    max_position_embeddings = 2_048;
    rope_theta = 10_000.0;
  }

let benchmark = ref Decode
let preset = ref "llama3-1b"
let prompt_length = ref 128
let cache_length = ref 2_048
let warmups = ref 5
let iterations = ref 20
let device_id = ref 0
let vocab_size = ref None
let hidden_size = ref None
let intermediate_size = ref None
let num_hidden_layers = ref None
let num_attention_heads = ref None
let num_key_value_heads = ref None
let max_position_embeddings = ref None
let rope_theta = ref None
let set_optional target value = target := Some value

let options =
  [
    ( "--case",
      Arg.Symbol
        ( [ "prefill"; "decode" ],
          function
          | "prefill" -> benchmark := Prefill
          | "decode" -> benchmark := Decode
          | value -> invalid_arg ("unknown benchmark case " ^ value) ),
      " Benchmark prefill or one-token cached decode (default: decode)" );
    ( "--preset",
      Arg.Symbol ([ "llama3-1b"; "smoke" ], fun value -> preset := value),
      " Model shape preset (default: llama3-1b)" );
    ( "--prompt-length",
      Arg.Set_int prompt_length,
      " Prompt tokens for the prefill case (default: 128)" );
    ( "--cache-length",
      Arg.Set_int cache_length,
      " Fixed KV-cache capacity (default: 2048)" );
    ("--warmups", Arg.Set_int warmups, " Warmup executions (default: 5)");
    ("--iterations", Arg.Set_int iterations, " Timed executions (default: 20)");
    ("--device", Arg.Set_int device_id, " CUDA device index (default: 0)");
    ( "--vocab-size",
      Arg.Int (set_optional vocab_size),
      " Override preset vocabulary size" );
    ( "--hidden-size",
      Arg.Int (set_optional hidden_size),
      " Override preset hidden size" );
    ( "--intermediate-size",
      Arg.Int (set_optional intermediate_size),
      " Override preset SwiGLU intermediate size" );
    ( "--layers",
      Arg.Int (set_optional num_hidden_layers),
      " Override preset decoder layer count" );
    ( "--q-heads",
      Arg.Int (set_optional num_attention_heads),
      " Override preset query head count" );
    ( "--kv-heads",
      Arg.Int (set_optional num_key_value_heads),
      " Override preset key/value head count" );
    ( "--max-position-embeddings",
      Arg.Int (set_optional max_position_embeddings),
      " Override preset maximum position count" );
    ( "--rope-theta",
      Arg.Float (set_optional rope_theta),
      " Override preset standard-RoPE theta" );
  ]

let usage =
  "llama_profile.exe [OPTIONS]\n\
   Synthetic Llama-3.2-1B-shaped cached-inference benchmark on PJRT CUDA."

let parse_arguments () =
  Arg.parse options
    (fun argument -> raise (Arg.Bad ("unexpected argument " ^ argument)))
    usage;
  let base =
    match !preset with
    | "llama3-1b" -> llama3_1b
    | "smoke" -> smoke
    | value -> invalid_arg ("unknown preset " ^ value)
  in
  let get override fallback = Option.value !override ~default:fallback in
  let dimensions =
    {
      vocab_size = get vocab_size base.vocab_size;
      hidden_size = get hidden_size base.hidden_size;
      intermediate_size = get intermediate_size base.intermediate_size;
      num_hidden_layers = get num_hidden_layers base.num_hidden_layers;
      num_attention_heads = get num_attention_heads base.num_attention_heads;
      num_key_value_heads = get num_key_value_heads base.num_key_value_heads;
      max_position_embeddings =
        get max_position_embeddings base.max_position_embeddings;
      rope_theta = get rope_theta base.rope_theta;
    }
  in
  if !prompt_length <= 0 then invalid_arg "--prompt-length must be positive";
  if !cache_length <= 0 then invalid_arg "--cache-length must be positive";
  if !cache_length > dimensions.max_position_embeddings then
    invalid_arg "--cache-length exceeds --max-position-embeddings";
  if !benchmark = Prefill && !prompt_length > !cache_length then
    invalid_arg "--prompt-length exceeds --cache-length";
  if !warmups < 0 then invalid_arg "--warmups may not be negative";
  if !iterations <= 0 then invalid_arg "--iterations must be positive";
  if !device_id < 0 then invalid_arg "--device may not be negative";
  dimensions

let tensor tensor = Ptree.tensor tensor

let empty_children names =
  Ptree.dict (List.map (fun name -> (name, Ptree.empty)) names)

let make_vars dimensions =
  let dtype = Nx.float16 in
  let vector size = tensor (Nx.ones dtype [| size |]) in
  let matrix rows columns = tensor (Nx.zeros dtype [| rows; columns |]) in
  let projection rows columns =
    Ptree.dict [ ("weight", matrix rows columns) ]
  in
  let head_dim = dimensions.hidden_size / dimensions.num_attention_heads in
  let kv_size = dimensions.num_key_value_heads * head_dim in
  let layer_params () =
    Ptree.dict
      [
        ( "input_layernorm",
          Ptree.dict [ ("scale", vector dimensions.hidden_size) ] );
        ( "self_attn",
          Ptree.dict
            [
              ( "q_proj",
                projection dimensions.hidden_size dimensions.hidden_size );
              ("k_proj", projection dimensions.hidden_size kv_size);
              ("v_proj", projection dimensions.hidden_size kv_size);
              ( "o_proj",
                projection dimensions.hidden_size dimensions.hidden_size );
            ] );
        ( "post_attention_layernorm",
          Ptree.dict [ ("scale", vector dimensions.hidden_size) ] );
        ( "mlp",
          Ptree.dict
            [
              ( "gate_proj",
                projection dimensions.hidden_size dimensions.intermediate_size
              );
              ( "up_proj",
                projection dimensions.hidden_size dimensions.intermediate_size
              );
              ( "down_proj",
                projection dimensions.intermediate_size dimensions.hidden_size
              );
            ] );
      ]
  in
  let layer_state () =
    Ptree.dict
      [
        ("input_layernorm", Ptree.empty);
        ("self_attn", empty_children [ "q_proj"; "k_proj"; "v_proj"; "o_proj" ]);
        ("post_attention_layernorm", Ptree.empty);
        ("mlp", empty_children [ "gate_proj"; "up_proj"; "down_proj" ]);
      ]
  in
  let params =
    Ptree.dict
      [
        ( "embed_tokens",
          Ptree.dict
            [ ("weight", matrix dimensions.vocab_size dimensions.hidden_size) ]
        );
        ( "layers",
          Ptree.list
            (List.init dimensions.num_hidden_layers (fun _ -> layer_params ()))
        );
        ("norm", Ptree.dict [ ("scale", vector dimensions.hidden_size) ]);
      ]
  in
  let state =
    Ptree.dict
      [
        ("embed_tokens", Ptree.empty);
        ( "layers",
          Ptree.list
            (List.init dimensions.num_hidden_layers (fun _ -> layer_state ()))
        );
        ("norm", Ptree.empty);
      ]
  in
  Layer.make_vars ~params ~state ~dtype

let input_ids ~vocab_size sequence =
  Array.init sequence (fun index -> Int32.of_int (index mod vocab_size))
  |> Nx.create Nx.int32 [| 1; sequence |]

let now () = Unix.gettimeofday ()

let percentile sorted fraction =
  let last = Array.length sorted - 1 in
  sorted.(min last (int_of_float (fraction *. float_of_int last)))

let elapsed_ms started = (now () -. started) *. 1_000.0
let benchmark_name = function Prefill -> "prefill" | Decode -> "decode"

let () =
  let dimensions = parse_arguments () in
  if not (Rune_pjrt.backend_available `Cuda) then
    failwith "the PJRT CUDA backend is unavailable";
  let sequence =
    match !benchmark with Prefill -> !prompt_length | Decode -> 1
  in
  let cfg =
    Llama.config ~vocab_size:dimensions.vocab_size
      ~hidden_size:dimensions.hidden_size
      ~intermediate_size:dimensions.intermediate_size
      ~num_hidden_layers:dimensions.num_hidden_layers
      ~num_attention_heads:dimensions.num_attention_heads
      ~num_key_value_heads:dimensions.num_key_value_heads
      ~max_position_embeddings:dimensions.max_position_embeddings
      ~rms_norm_eps:1e-5 ~rope_theta:dimensions.rope_theta
      ~tie_word_embeddings:true ()
  in
  Printf.printf
    "workload=synthetic-llama-3.2-1b-shaped implementation=hugr-pjrt case=%s \
     dtype=float16 batch=1 sequence=%d cache=%d warmups=%d iterations=%d\n\
     %!"
    (benchmark_name !benchmark)
    sequence !cache_length !warmups !iterations;
  Printf.printf
    "vocab=%d hidden=%d intermediate=%d layers=%d q_heads=%d kv_heads=%d \
     head_dim=%d max_positions=%d rope_theta=%.0f tied_embeddings=true\n\
     %!"
    dimensions.vocab_size dimensions.hidden_size dimensions.intermediate_size
    dimensions.num_hidden_layers dimensions.num_attention_heads
    dimensions.num_key_value_heads
    (dimensions.hidden_size / dimensions.num_attention_heads)
    dimensions.max_position_embeddings dimensions.rope_theta;
  Printf.printf
    "note=synthetic_zero_weights; rope=standard_not_llama3_scaled; \
     initial_cache_position=0_with_fixed_capacity_cache\n\
     %!";
  let started = now () in
  let vars = make_vars dimensions in
  let parameter_count = Ptree.count_parameters (Layer.params vars) in
  Printf.printf "parameter_initialization_ms=%.6f\n%!" (elapsed_ms started);
  Printf.printf "parameters=%d parameter_bytes=%d\n%!" parameter_count
    (2 * parameter_count);
  let runner = Llama.Pjrt.compile ~device_id:!device_id cfg vars in
  let host_cache =
    Llama.Cache.create cfg ~batch_size:1 ~max_length:!cache_length
      ~dtype:Nx.float16
  in
  let started = now () in
  let resident_cache = Resident.of_host runner host_cache in
  Printf.printf "initial_cache_upload_ms=%.6f\n%!" (elapsed_ms started);
  let ids = input_ids ~vocab_size:dimensions.vocab_size sequence in
  let execute () =
    match !benchmark with
    | Prefill -> Resident.prefill runner resident_cache ids
    | Decode -> Resident.decode_step runner resident_cache ids
  in
  let started = now () in
  let first_logits, first_cache = execute () in
  Printf.printf "first_compile_and_execute_ms=%.6f\n%!" (elapsed_ms started);
  Printf.printf "output_cache_length=%d\n%!" (Resident.length first_cache);
  let last_logits = ref first_logits in
  for _ = 1 to !warmups do
    let logits, _ = execute () in
    last_logits := logits
  done;
  let samples = Array.make !iterations 0.0 in
  for index = 0 to !iterations - 1 do
    let started = now () in
    let logits, _ = execute () in
    samples.(index) <- elapsed_ms started;
    last_logits := logits
  done;
  let sorted = Array.copy samples in
  Array.sort Float.compare sorted;
  let mean = Array.fold_left ( +. ) 0.0 samples /. float_of_int !iterations in
  Printf.printf
    "steady_e2e_ms mean=%.6f p10=%.6f median=%.6f p90=%.6f min=%.6f max=%.6f\n\
     %!"
    mean (percentile sorted 0.10) (percentile sorted 0.50)
    (percentile sorted 0.90) sorted.(0)
    sorted.(!iterations - 1);
  Printf.printf "tokens_per_second=%.6f\n%!"
    (float_of_int sequence *. 1_000.0 /. mean);
  let shape = Nx.shape !last_logits |> Array.to_list in
  Printf.printf "logits_shape=[%s] first_logit=%.9g\n%!"
    (String.concat "," (List.map string_of_int shape))
    (Nx.item [ 0; 0; 0 ] !last_logits)
