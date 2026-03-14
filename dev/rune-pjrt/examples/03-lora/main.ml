let () =
  let backend = Rune_pjrt_examples.Support.backend_of_env () in
  let device_id = Rune_pjrt_examples.Support.device_id_of_env () in
  let steps =
    Rune_pjrt_examples.Support.int_of_env "RUNE_PJRT_LORA_STEPS" ~default:180
  in
  let learning_rate =
    Rune_pjrt_examples.Support.float_of_env "RUNE_PJRT_LORA_LR" ~default:0.5
  in
  let result =
    Rune_pjrt_examples.Lora_example.run ~backend ~device_id ~steps
      ~learning_rate ()
  in
  Rune_pjrt_examples.Lora_example.validate result;
  Printf.printf "backend: %s\n"
    (Rune_pjrt.Backend.to_string backend);
  Printf.printf "base loss:  %.6f\nfinal loss: %.6f\n" result.base_loss
    result.final_loss
