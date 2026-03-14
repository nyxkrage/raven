type backend = [ `Cpu | `Cuda ]

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let backend_of_string = function
  | "cpu" -> `Cpu
  | "cuda" -> `Cuda
  | value ->
      invalid_argf
        "RUNE_PJRT_BACKEND must be \"cpu\" or \"cuda\", got %S" value

let backend_of_env () =
  match Sys.getenv_opt "RUNE_PJRT_BACKEND" with
  | None -> `Cpu
  | Some value -> backend_of_string (String.lowercase_ascii value)

let int_of_env name ~default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value -> (
      match int_of_string_opt value with
      | Some value -> value
      | None ->
          invalid_argf "%s must be an integer, got %S" name value)

let float_of_env name ~default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value -> (
      match float_of_string_opt value with
      | Some value -> value
      | None ->
          invalid_argf "%s must be a float, got %S" name value)

let device_id_of_env () = int_of_env "RUNE_PJRT_DEVICE_ID" ~default:0

let require_backend backend =
  if not (Rune_pjrt.backend_available backend) then
    failwith
      (Printf.sprintf "PJRT backend %s is unavailable: %s"
         (Rune_pjrt.Backend.to_string backend)
         (Rune_pjrt.status ()))

let float_scalar (t : (float, 'a) Nx.t) : float = Nx.item [] t

let float_array t = Nx.to_array t

let int32_array t = Nx.to_array t

let pp_i32_array arr =
  arr
  |> Array.to_list
  |> List.map Int32.to_string
  |> String.concat ", "

let max_abs_diff expected actual =
  let max_diff = ref 0.0 in
  Array.iter2
    (fun x y -> max_diff := max !max_diff (Float.abs (x -. y)))
    expected actual;
  !max_diff
