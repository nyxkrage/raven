(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type result = {
  backend : Rune.Backend.t;
  prompt : int array;
  generated : int array;
  expected : int array;
}

let vocab_size = 6
let prompt_ids = [| 1; 2; 3 |]
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let int_of_env name ~default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value -> (
      match int_of_string_opt value with
      | Some value -> value
      | None -> invalid_argf "%s must be an integer, got %S" name value)

let usage () =
  Printf.eprintf
    "usage: dune exec packages/rune/examples/02-jit-inference/main.exe -- \
     [tolk-cpu|pjrt-cpu|pjrt-cuda] [max_tokens]\n";
  Printf.eprintf "       or set RUNE_JIT_BACKEND and RUNE_JIT_MAX_TOKENS\n"

let config () =
  match Array.to_list Sys.argv with
  | [ _ ] ->
      (Rune.Backend.of_env (), int_of_env "RUNE_JIT_MAX_TOKENS" ~default:4)
  | [ _; ("--help" | "-h") ] ->
      usage ();
      exit 0
  | [ _; backend ] ->
      ( Rune.Backend.of_string backend,
        int_of_env "RUNE_JIT_MAX_TOKENS" ~default:4 )
  | [ _; backend; max_tokens ] -> (
      match int_of_string_opt max_tokens with
      | Some max_tokens -> (Rune.Backend.of_string backend, max_tokens)
      | None -> invalid_argf "max_tokens must be an integer, got %S" max_tokens)
  | _ ->
      usage ();
      invalid_arg "too many arguments"

let next_token prev =
  let one = Nx.scalar Nx.float32 1.0 in
  Nx.add prev one

let token_of_tensor t =
  (Nx.item [] t |> Float.round |> int_of_float) mod vocab_size

let expected_tokens ~max_tokens =
  let total = Array.length prompt_ids + max_tokens in
  let full = Array.make total 0 in
  Array.blit prompt_ids 0 full 0 (Array.length prompt_ids);
  for i = Array.length prompt_ids to total - 1 do
    full.(i) <- (full.(i - 1) + 1) mod vocab_size
  done;
  full

let decode ~device ~max_tokens =
  if max_tokens < 0 then invalid_arg "max_tokens must be non-negative";
  let step = Rune.jit ~device next_token in
  let total = Array.length prompt_ids + max_tokens in
  let generated = Array.make total 0 in
  Array.blit prompt_ids 0 generated 0 (Array.length prompt_ids);
  let last_prompt = prompt_ids.(Array.length prompt_ids - 1) in
  let current = ref (Nx.scalar Nx.float32 (float_of_int last_prompt)) in
  for i = Array.length prompt_ids to total - 1 do
    let next = step !current in
    let token = token_of_tensor next in
    generated.(i) <- token;
    current := Nx.scalar Nx.float32 (float_of_int token)
  done;
  generated

let run ~backend ~max_tokens =
  if backend = Rune.Backend.Tolk_cpu && max_tokens > 2 then
    failwith
      "tolk-cpu currently reaches the non-functional Rune/Tolk replay path; \
       use pjrt-cpu/pjrt-cuda to run the full decode";
  let device = Rune.Backend.device backend in
  let generated = decode ~device ~max_tokens in
  {
    backend;
    prompt = Array.copy prompt_ids;
    generated;
    expected = expected_tokens ~max_tokens;
  }

let pp_int_array arr =
  arr |> Array.to_list |> List.map string_of_int |> String.concat ", "

let validate result =
  if result.generated <> result.expected then
    failwith
      (Printf.sprintf
         "jit inference mismatch for %s\nexpected: [%s]\nactual:   [%s]"
         (Rune.Backend.to_string result.backend)
         (pp_int_array result.expected)
         (pp_int_array result.generated))

let () =
  let backend, max_tokens = config () in
  let result = run ~backend ~max_tokens in
  validate result;
  Printf.printf "backend:   %s\n" (Rune.Backend.to_string backend);
  Printf.printf "prompt:    [%s]\n" (pp_int_array result.prompt);
  Printf.printf "generated: [%s]\n" (pp_int_array result.generated)
