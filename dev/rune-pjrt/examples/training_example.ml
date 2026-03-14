type result = {
  initial_loss : float;
  final_loss : float;
  predictions : float array;
  targets : float array;
}

let x_train =
  Nx.create Nx.float32 [| 6; 2 |]
    [|
      0.; 0.;
      0.; 1.;
      1.; 0.;
      1.; 1.;
      2.; 1.;
      -1.; 2.;
    |]

let true_w = Nx.create Nx.float32 [| 2; 1 |] [| 2.0; -1.5 |]
let true_b = Nx.create Nx.float32 [| 1 |] [| 0.25 |]
let y_train = Nx.add (Nx.matmul x_train true_w) true_b

let linear w b x = Nx.add (Nx.matmul x w) b

let mse_loss w b x y =
  let diff = Nx.sub (linear w b x) y in
  Nx.mean (Nx.mul diff diff)

let train_step ?(backend = `Cpu) ?(device_id = 0) ~learning_rate () =
  Rune_pjrt.jits ~backend ~device_id (fun tensors ->
      match tensors with
      | [ w; b; x; y ] ->
          let objective params =
            match params with
            | [ w; b ] -> mse_loss w b x y
            | _ -> invalid_arg "training example expected [w; b] parameters"
          in
          let loss, grads = Rune.value_and_grads objective [ w; b ] in
          let grad_w, grad_b =
            match grads with
            | [ grad_w; grad_b ] -> (grad_w, grad_b)
            | _ -> invalid_arg "training example expected two gradients"
          in
          let w' = Nx.sub w (Nx.mul (Nx.scalar_like w learning_rate) grad_w) in
          let b' = Nx.sub b (Nx.mul (Nx.scalar_like b learning_rate) grad_b) in
          [ loss; w'; b' ]
      | _ ->
          invalid_arg
            "training example expected [weights; bias; inputs; targets]")

let run ?(backend = `Cpu) ?(device_id = 0) ?(steps = 120)
    ?(learning_rate = 0.1) () =
  Support.require_backend backend;
  let step = train_step ~backend ~device_id ~learning_rate () in
  let rec loop n w b last_loss =
    if n = 0 then (last_loss, w, b)
    else
      match step [ w; b; x_train; y_train ] with
      | [ loss; w'; b' ] ->
          loop (n - 1) w' b' (Support.float_scalar loss)
      | _ -> failwith "training example returned unexpected output arity"
  in
  let w0 = Nx.zeros Nx.float32 [| 2; 1 |] in
  let b0 = Nx.zeros Nx.float32 [| 1 |] in
  let initial_loss = Support.float_scalar (mse_loss w0 b0 x_train y_train) in
  let final_loss, w_final, b_final =
    loop steps w0 b0 initial_loss
  in
  let predict =
    Rune_pjrt.jit ~backend ~device_id (fun x -> linear w_final b_final x)
  in
  let predictions = predict x_train |> Support.float_array in
  let targets = Support.float_array y_train in
  { initial_loss; final_loss; predictions; targets }

let validate result =
  if result.final_loss >= result.initial_loss *. 0.02 then
    failwith
      (Printf.sprintf
         "training example did not reduce loss enough (initial=%g final=%g)"
         result.initial_loss result.final_loss);
  let max_diff = Support.max_abs_diff result.targets result.predictions in
  if max_diff > 5e-2 then
    failwith
      (Printf.sprintf
         "training example predictions too far from targets (max diff=%g)"
         max_diff)
