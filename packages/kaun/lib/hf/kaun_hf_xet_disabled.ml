let available () = false

let download_hf_file ?token:_ ~model_id:_ ~filename:_ ~revision:_ ~destination:_
    () =
  invalid_arg "Kaun_hf: Xet support is unavailable"
