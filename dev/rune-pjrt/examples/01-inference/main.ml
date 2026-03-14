let () =
  let backend = Rune_pjrt_examples.Support.backend_of_env () in
  let device_id = Rune_pjrt_examples.Support.device_id_of_env () in
  let max_tokens =
    Rune_pjrt_examples.Support.int_of_env "RUNE_PJRT_MAX_TOKENS" ~default:4
  in
  let result =
    Rune_pjrt_examples.Inference_example.run ~backend ~device_id ~max_tokens ()
  in
  Rune_pjrt_examples.Inference_example.validate result;
  Printf.printf "backend: %s\n"
    (Rune_pjrt.Backend.to_string backend);
  Printf.printf "prompt:    [%s]\n"
    (Rune_pjrt_examples.Support.pp_i32_array result.prompt);
  Printf.printf "generated: [%s]\n"
    (Rune_pjrt_examples.Support.pp_i32_array result.generated)
