let () =
  let backend = Rune_pjrt_examples.Support.backend_of_env () in
  let device_id = Rune_pjrt_examples.Support.device_id_of_env () in
  let steps =
    Rune_pjrt_examples.Support.int_of_env "RUNE_PJRT_TRAIN_STEPS" ~default:120
  in
  let learning_rate =
    Rune_pjrt_examples.Support.float_of_env "RUNE_PJRT_LEARNING_RATE"
      ~default:0.1
  in
  let result =
    Rune_pjrt_examples.Training_example.run ~backend ~device_id ~steps
      ~learning_rate ()
  in
  Rune_pjrt_examples.Training_example.validate result;
  Printf.printf "backend: %s\n"
    (Rune_pjrt.Backend.to_string backend);
  Printf.printf "initial loss: %.6f\nfinal loss:   %.6f\n" result.initial_loss
    result.final_loss
