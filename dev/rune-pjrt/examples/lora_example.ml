type result = {
  base_loss : float;
  final_loss : float;
  adapted_predictions : float array;
  target_predictions : float array;
}

let x_train =
  Nx.create Nx.float32 [| 6; 4 |]
    [|
      1.0; 0.0; 0.5; -1.0;
      0.0; 1.0; -0.5; 0.5;
      1.0; 1.0; 0.0; 0.0;
      -1.0; 0.5; 1.0; 0.5;
      0.5; -0.5; 1.0; 1.0;
      1.5; 0.5; -1.0; 0.0;
    |]

let base_w =
  Nx.create Nx.float32 [| 4; 3 |]
    [|
      0.40; -0.10; 0.20;
      -0.20; 0.30; 0.10;
      0.05; 0.25; -0.15;
      0.30; -0.05; 0.35;
    |]

let delta_a_true =
  Nx.create Nx.float32 [| 4; 2 |]
    [|
      0.30; -0.10;
      -0.20; 0.25;
      0.10; 0.15;
      0.05; -0.30;
    |]

let delta_b_true =
  Nx.create Nx.float32 [| 2; 3 |]
    [|
      0.40; -0.20; 0.10;
      -0.10; 0.35; 0.25;
    |]

let target_w = Nx.add base_w (Nx.matmul delta_a_true delta_b_true)

let target_predictions = Nx.matmul x_train target_w

let lora_forward ~base_w ~a ~b x =
  let delta = Nx.matmul a b in
  Nx.matmul x (Nx.add base_w delta)

let mse x y =
  let diff = Nx.sub x y in
  Nx.mean (Nx.mul diff diff)

let train_step ?(backend = `Cpu) ?(device_id = 0) ~learning_rate () =
  Rune_pjrt.jits ~backend ~device_id (fun tensors ->
      match tensors with
      | [ a; b; x; y ] ->
          let objective params =
            match params with
            | [ a; b ] -> mse (lora_forward ~base_w ~a ~b x) y
            | _ -> invalid_arg "lora example expected [a; b] parameters"
          in
          let loss, grads = Rune.value_and_grads objective [ a; b ] in
          let grad_a, grad_b =
            match grads with
            | [ grad_a; grad_b ] -> (grad_a, grad_b)
            | _ -> invalid_arg "lora example expected two gradients"
          in
          let a' = Nx.sub a (Nx.mul (Nx.scalar_like a learning_rate) grad_a) in
          let b' = Nx.sub b (Nx.mul (Nx.scalar_like b learning_rate) grad_b) in
          [ loss; a'; b' ]
      | _ ->
          invalid_arg "lora example expected [a; b; inputs; targets]")

let run ?(backend = `Cpu) ?(device_id = 0) ?(steps = 180)
    ?(learning_rate = 0.5) () =
  Support.require_backend backend;
  let step = train_step ~backend ~device_id ~learning_rate () in
  let a0 =
    Nx.create Nx.float32 [| 4; 2 |]
      [|
        0.05; -0.02;
        -0.03; 0.04;
        0.02; 0.01;
        -0.01; 0.03;
      |]
  in
  let b0 = Nx.zeros Nx.float32 [| 2; 3 |] in
  let base_loss =
    Support.float_scalar (mse (lora_forward ~base_w ~a:a0 ~b:b0 x_train) target_predictions)
  in
  let rec loop n a b last_loss =
    if n = 0 then (last_loss, a, b)
    else
      match step [ a; b; x_train; target_predictions ] with
      | [ loss; a'; b' ] ->
          loop (n - 1) a' b' (Support.float_scalar loss)
      | _ -> failwith "lora example returned unexpected output arity"
  in
  let final_loss, a_final, b_final = loop steps a0 b0 base_loss in
  let predict =
    Rune_pjrt.jit ~backend ~device_id (fun x ->
        lora_forward ~base_w ~a:a_final ~b:b_final x)
  in
  let adapted_predictions = predict x_train |> Support.float_array in
  let target_predictions = Support.float_array target_predictions in
  { base_loss; final_loss; adapted_predictions; target_predictions }

let validate result =
  if result.final_loss >= result.base_loss *. 0.2 then
    failwith
      (Printf.sprintf
         "lora example did not improve enough (base=%g final=%g)"
         result.base_loss result.final_loss);
  let max_diff =
    Support.max_abs_diff result.target_predictions result.adapted_predictions
  in
  if max_diff > 1.2e-1 then
    failwith
      (Printf.sprintf "lora example predictions too far from target (max diff=%g)"
         max_diff)
