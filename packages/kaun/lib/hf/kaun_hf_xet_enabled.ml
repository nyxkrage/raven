let available () =
  try
    ignore (Xet.version ());
    true
  with Failure _ | Invalid_argument _ -> false

let download_hf_file ?token ~model_id ~filename ~revision ~destination () =
  Xet.download_hf_file ?token ~repo_id:model_id ~filename ~revision ~destination
    ~repo_type:Model ()
