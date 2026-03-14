let skip_cuda () = Sys.getenv_opt "RUNE_PJRT_TEST_SKIP_CUDA" <> None

let backend_available backend =
  match backend with
  | `Cuda when skip_cuda () -> false
  | (`Cpu | `Cuda) as backend -> Rune_pjrt.backend_available backend

let run_backend backend =
  Rune_pjrt_examples.Inference_example.run ~backend ()
  |> Rune_pjrt_examples.Inference_example.validate;
  Rune_pjrt_examples.Training_example.run ~backend ()
  |> Rune_pjrt_examples.Training_example.validate;
  Rune_pjrt_examples.Lora_example.run ~backend ()
  |> Rune_pjrt_examples.Lora_example.validate

let () =
  if backend_available `Cpu then run_backend `Cpu;
  if backend_available `Cuda then run_backend `Cuda
