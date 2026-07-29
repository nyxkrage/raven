(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type compiled = {
  backend : Backend.t;
  device_id : int;
  plugin_path : string;
  cache_key : string;
  signature : Signature.t;
  program : Ir.program;
  artifact_dir : string;
  spec_path : string;
  module_path : string;
  module_text : string;
  output_descs : Ir.desc list;
  extra_inputs : (Ir.desc * string) list;
  mutable ffi_registered : bool;
}

type native_buffer

type ('a, 'b) device_buffer = {
  native : native_buffer;
  backend : Backend.t;
  device_id : int;
  dtype : ('a, 'b) Nx_core.Dtype.t;
  shape : int array;
}

type packed_device_buffer =
  | Device_buffer : ('a, 'b) device_buffer -> packed_device_buffer

external native_execute :
  plugin_path:string ->
  cache_key:string ->
  device_id:int ->
  stablehlo:string ->
  dynamic_input_dtypes:string array ->
  dynamic_input_shapes:int array array ->
  dynamic_input_data:Obj.t array ->
  constant_input_dtypes:string array ->
  constant_input_shapes:int array array ->
  constant_input_data:string array ->
  output_dtypes:string array ->
  output_shapes:int array array ->
  output_data:Obj.t array ->
  unit = "caml_rune_pjrt_execute_bc" "caml_rune_pjrt_execute"

external native_buffer_of_host :
  plugin_path:string ->
  device_id:int ->
  dtype:string ->
  shape:int array ->
  data:Obj.t ->
  native_buffer = "caml_rune_pjrt_buffer_of_host"

external native_buffer_to_host :
  native_buffer -> dtype:string -> shape:int array -> data:Obj.t -> unit
  = "caml_rune_pjrt_buffer_to_host"

external native_buffer_await : native_buffer -> unit
  = "caml_rune_pjrt_buffer_await"

external native_execute_device :
  plugin_path:string ->
  cache_key:string ->
  device_id:int ->
  stablehlo:string ->
  dynamic_input_buffers:native_buffer array ->
  constant_input_dtypes:string array ->
  constant_input_shapes:int array array ->
  constant_input_data:string array ->
  output_dtypes:string array ->
  output_shapes:int array array ->
  native_buffer array
  = "caml_rune_pjrt_execute_device_bc" "caml_rune_pjrt_execute_device"

external native_register_ffi_handler :
  plugin_path:string ->
  library_path:string ->
  library_digest:string ->
  symbol:string ->
  target:string ->
  unit = "caml_rune_pjrt_register_ffi_handler"

let ensure_dir path =
  let rec loop path =
    if path = "." || path = "/" || Sys.file_exists path then ()
    else (
      loop (Filename.dirname path);
      Unix.mkdir path 0o755)
  in
  loop path

let artifact_dir key =
  let dir = Filename.concat "_build/default/dev/rune-pjrt" "artifacts" in
  ensure_dir dir;
  let digest = Digest.to_hex (Digest.string key) in
  let dir = Filename.concat dir digest in
  ensure_dir dir;
  dir

let plugin_dir = Filename.concat "_build/default/dev/rune-pjrt" "plugins"
let bazel_root = Filename.concat "_build/default/dev/rune-pjrt" "bazel"
let plugin_path_env = "RUNE_PJRT_PLUGIN_PATH"
let plugin_search_max_depth = 6

let plugin_name : Backend.t -> string = function
  | `Cpu -> "pjrt_c_api_cpu_plugin.so"
  | `Cuda -> "pjrt_c_api_gpu_plugin.so"

let plugin_record_path backend =
  Filename.concat plugin_dir (plugin_name backend ^ ".path")

let bazel_plugin_path backend =
  Filename.concat "vendor/xla/bazel-bin/xla/pjrt/c" (plugin_name backend)

let expand_home path =
  let len = String.length path in
  if len = 0 || path.[0] <> '~' then path
  else
    match Sys.getenv_opt "HOME" with
    | None | Some "" -> path
    | Some home ->
        if len = 1 then home
        else if path.[1] = '/' then home ^ String.sub path 1 (len - 1)
        else path

let plugin_search_entries () =
  match Sys.getenv_opt plugin_path_env with
  | None -> []
  | Some entries ->
      entries |> String.split_on_char ':' |> List.map String.trim
      |> List.filter (fun entry -> entry <> "")
      |> List.map expand_home

let is_directory path = try Sys.is_directory path with Sys_error _ -> false

let realpath_opt path =
  try Some (Unix.realpath path) with Unix.Unix_error _ -> None

let existing_file path =
  if Sys.file_exists path && not (is_directory path) then
    match realpath_opt path with
    | Some resolved -> Some resolved
    | None -> Some path
  else None

let read_path path =
  if not (Sys.file_exists path) then None
  else
    let ic = open_in_bin path in
    Fun.protect
      (fun () ->
        let line = input_line ic |> String.trim in
        if line = "" then None else Some (expand_home line))
      ~finally:(fun () -> close_in ic)

let resolve_record_path path =
  match read_path path with Some path -> existing_file path | None -> None

let resolve_plugin_entry backend entry =
  let name = plugin_name backend in
  if is_directory entry then
    let record = Filename.concat entry (name ^ ".path") in
    match resolve_record_path record with
    | Some path -> Some path
    | None -> existing_file (Filename.concat entry name)
  else
    match Filename.basename entry with
    | basename when basename = name ^ ".path" -> resolve_record_path entry
    | basename when basename = name -> existing_file entry
    | _ -> None

let sorted_readdir path =
  try Sys.readdir path |> Array.to_list |> List.sort String.compare
  with Sys_error _ -> []

let rec find_plugin_under backend ~depth dir =
  match resolve_plugin_entry backend dir with
  | Some path -> Some path
  | None ->
      if depth = 0 then None
      else
        sorted_readdir dir
        |> List.find_map (fun entry ->
            let path = Filename.concat dir entry in
            if is_directory path then
              find_plugin_under backend ~depth:(depth - 1) path
            else resolve_plugin_entry backend path)

let env_plugin_path backend =
  plugin_search_entries ()
  |> List.find_map (fun entry ->
      match resolve_plugin_entry backend entry with
      | Some path -> Some path
      | None ->
          if is_directory entry then
            find_plugin_under backend ~depth:plugin_search_max_depth entry
          else None)

let resolved_plugin_path backend =
  match env_plugin_path backend with
  | Some path -> Some path
  | None -> (
      let recorded =
        match read_path (plugin_record_path backend) with
        | Some path -> existing_file path
        | None -> None
      in
      match recorded with
      | Some path -> Some path
      | None -> (
          match existing_file (bazel_plugin_path backend) with
          | Some path -> Some path
          | None ->
              let copied = Filename.concat plugin_dir (plugin_name backend) in
              existing_file copied))

let plugin_path backend =
  match resolved_plugin_path backend with
  | Some path -> path
  | None -> (
      match plugin_search_entries () with
      | entry :: _ -> Filename.concat entry (plugin_name backend)
      | [] -> Filename.concat plugin_dir (plugin_name backend))

let has_prefix ~prefix s =
  let prefix_len = String.length prefix in
  String.length s >= prefix_len && String.sub s 0 prefix_len = prefix

let split_on_space s =
  s |> String.split_on_char ' ' |> List.filter (fun part -> part <> "")

let cuda_root_from_plugin_path path =
  let marker = "/execroot/xla/bazel-out/" in
  let path_len = String.length path in
  let marker_len = String.length marker in
  let rec find_from i =
    if i + marker_len > path_len then None
    else if String.sub path i marker_len = marker then Some i
    else find_from (i + 1)
  in
  match find_from 0 with
  | None -> None
  | Some idx ->
      let base = String.sub path 0 idx in
      let cuda_root = Filename.concat base "external/cuda_nvcc" in
      let libdevice =
        Filename.concat cuda_root "nvvm/libdevice/libdevice.10.bc"
      in
      if Sys.file_exists libdevice then Some cuda_root else None

let find_cuda_data_dir () =
  let cuda_root_suffix = [ "external"; "cuda_nvcc" ] in
  let candidate_root root =
    let cuda_root = List.fold_left Filename.concat root cuda_root_suffix in
    let libdevice =
      Filename.concat cuda_root "nvvm/libdevice/libdevice.10.bc"
    in
    if Sys.file_exists libdevice then Some cuda_root else None
  in
  let candidate_base root =
    match candidate_root root with
    | Some _ as hit -> hit
    | None ->
        if not (Sys.file_exists root) then None
        else
          Sys.readdir root
          |> Array.to_list
          |> List.map (Filename.concat root)
          |> List.find_map candidate_root
  in
  match Option.bind (resolved_plugin_path `Cuda) cuda_root_from_plugin_path with
  | Some _ as hit -> hit
  | None ->
      if not (Sys.file_exists bazel_root) then None
      else
        Sys.readdir bazel_root
        |> Array.to_list
        |> List.map (Filename.concat bazel_root)
        |> List.find_map candidate_base

let configure_xla_flags () =
  match find_cuda_data_dir () with
  | None -> ()
  | Some cuda_root ->
      let flag = "--xla_gpu_cuda_data_dir=" ^ cuda_root in
      let current = Sys.getenv_opt "XLA_FLAGS" |> Option.value ~default:"" in
      let already_set =
        split_on_space current
        |> List.exists (has_prefix ~prefix:"--xla_gpu_cuda_data_dir=")
      in
      if not already_set then
        let updated =
          if current = "" then flag else current ^ " " ^ flag
        in
        Unix.putenv "XLA_FLAGS" updated

let () = configure_xla_flags ()

let backend_available backend = Option.is_some (resolved_plugin_path backend)

let is_available () = backend_available `Cpu || backend_available `Cuda

let status () =
  if backend_available `Cpu && backend_available `Cuda then
    "PJRT CPU and CUDA plugins available"
  else if backend_available `Cpu then
    "PJRT CPU plugin available; build CUDA plugin with `bash dev/rune-pjrt/scripts/build_plugin.sh cuda`"
  else if backend_available `Cuda then
    "PJRT CUDA plugin available"
  else if Sys.file_exists "vendor/xla" then
    "PJRT plugins missing; build one with `bash dev/rune-pjrt/scripts/build_plugin.sh cpu`"
  else
    "vendor/xla not detected; clone XLA under vendor/ and build a PJRT plugin"

let json_escape s =
  let b = Buffer.create (String.length s + 8) in
  String.iter
    (function
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | '\n' -> Buffer.add_string b "\\n"
      | '\r' -> Buffer.add_string b "\\r"
      | '\t' -> Buffer.add_string b "\\t"
      | c -> Buffer.add_char b c)
    s;
  Buffer.contents b

let write_file path f =
  let oc = open_out_bin path in
  Fun.protect
    (fun () -> f oc)
    ~finally:(fun () -> close_out oc)

let supported_op = function
  | Ir.Parameter _
  | Constant _
  | Unary { op = Contiguous | Copy; _ }
  | Unary
      {
        op =
          ( Neg
          | Sin
          | Sqrt
          | Recip
          | Log
          | Exp
          | Cos
          | Abs
          | Sign
          | Tan
          | Asin
          | Acos
          | Atan
          | Sinh
          | Cosh
          | Tanh
          | Trunc
          | Ceil
          | Floor
          | Round
          | Erf );
        _;
      }
  | Binary
      {
        op =
          ( Add
          | Sub
          | Mul
          | Idiv
          | Fdiv
          | Max
          | Min
          | Mod
          | Pow
          | Xor
          | Or
          | And
          | Atan2
          | CmpEq
          | CmpNe
          | CmpLt
          | CmpLe );
        _;
      }
  | Where _
  | Reduce { op = (Reduce_sum | Reduce_max); _ }
  | Arg_reduce { op = Argmax; _ }
  | Reshape _
  | Expand _
  | Permute _
  | Shrink _
  | Flip _
  | Cat _
  | Pad _
  | Cast _
  | Gather _
  | Matmul _
  | Custom_call _
  | Triton_call _ ->
      true
  | Reduce _ | Arg_reduce _ | Assign _ | Buffer _ | Unsupported _ ->
      false

let validate_program program =
  match Ir.unsupported_ops program with
  | name :: _ ->
      Error.raise (Error.Unsupported_op name)
  | [] ->
      (match List.find_opt (fun node -> not (supported_op node.Ir.op)) program.Ir.nodes with
      | Some node ->
          Error.raise
            (Error.Unsupported_program
               (Printf.sprintf "node %d uses unsupported op %s" node.id
                  (Ir.op_name node.op)))
      | None -> ())

let write_buffer path (type a b) (dtype : (a, b) Nx_core.Dtype.t)
    (buffer : (a, b) Nx_buffer.t) =
  let bytes =
    Bytes.create (Nx_buffer.length buffer * Nx_core.Dtype.itemsize dtype)
  in
  Nx_buffer.blit_to_bytes buffer bytes;
  write_file path (fun oc -> output_bytes oc bytes)

let write_literal artifact_dir node_id (Ir.Literal { dtype; buffer; _ } as literal)
    =
  ignore literal;
  let file = Printf.sprintf "const_%d.bin" node_id in
  let path = Filename.concat artifact_dir file in
  write_buffer path dtype buffer;
  file

let data_string_of_literal (Ir.Literal { dtype; buffer; _ }) =
  let bytes =
    Bytes.create (Nx_buffer.length buffer * Nx_core.Dtype.itemsize dtype)
  in
  Nx_buffer.blit_to_bytes buffer bytes;
  Bytes.unsafe_to_string bytes

let packed_output_descs outputs =
  List.map
    (fun (Trace.Tensor t) -> Ir.desc_of_tensor t)
    outputs

let desc_key (desc : Ir.desc) =
  Printf.sprintf "%s:%s" desc.dtype (Nx_core.Shape.to_string desc.shape)

let compiled_cache_key signature module_text extra_inputs =
  let b = Buffer.create 256 in
  Buffer.add_string b (Signature.key signature);
  Buffer.add_char b '\n';
  Buffer.add_string b module_text;
  List.iter
    (fun ((desc : Ir.desc), data) ->
      Buffer.add_char b '\n';
      Buffer.add_string b (desc_key desc);
      Buffer.add_char b '\n';
      Buffer.add_string b (Digest.to_hex (Digest.string data)))
    extra_inputs;
  Digest.to_hex (Digest.string (Buffer.contents b))

let write_json_list oc pp_item items =
  output_char oc '[';
  let rec loop = function
    | [] -> ()
    | [ x ] -> pp_item oc x
    | x :: xs ->
        pp_item oc x;
        output_char oc ',';
        loop xs
  in
  loop items;
  output_char oc ']'

let write_json_int_array oc xs =
  write_json_list oc (fun oc x -> output_string oc (string_of_int x))
    (Array.to_list xs)

let write_json_bool_array oc xs =
  write_json_list oc
    (fun oc x -> output_string oc (if x then "true" else "false"))
    (Array.to_list xs)

let write_json_limits oc limits =
  write_json_list oc
    (fun oc (lo, hi) ->
      output_char oc '[';
      output_string oc (string_of_int lo);
      output_char oc ',';
      output_string oc (string_of_int hi);
      output_char oc ']')
    (Array.to_list limits)

let write_op_json artifact_dir oc node =
  let open Ir in
  match node.op with
  | Parameter index ->
      Printf.fprintf oc "{\"tag\":\"parameter\",\"index\":%d}" index
  | Constant literal ->
      let file = write_literal artifact_dir node.id literal in
      Printf.fprintf oc "{\"tag\":\"constant\",\"file\":\"%s\"}" file
  | Buffer { size_in_elements } ->
      Printf.fprintf oc "{\"tag\":\"buffer\",\"size\":%d}" size_in_elements
  | Unary { op; input } ->
      Printf.fprintf oc "{\"tag\":\"unary\",\"op\":\"%s\",\"input\":%d}"
        (json_escape (op_name (Unary { op; input })))
        input
  | Binary { op; lhs; rhs } ->
      Printf.fprintf oc
        "{\"tag\":\"binary\",\"op\":\"%s\",\"lhs\":%d,\"rhs\":%d}"
        (json_escape (op_name (Binary { op; lhs; rhs })))
        lhs rhs
  | Where { condition; if_true; if_false } ->
      Printf.fprintf oc
        "{\"tag\":\"where\",\"condition\":%d,\"if_true\":%d,\"if_false\":%d}"
        condition if_true if_false
  | Reduce { op; input; axes; keepdims } ->
      Printf.fprintf oc
        "{\"tag\":\"reduce\",\"op\":\"%s\",\"input\":%d,\"axes\":"
        (json_escape (op_name (Reduce { op; input; axes; keepdims })))
        input;
      write_json_int_array oc axes;
      Printf.fprintf oc ",\"keepdims\":%s}" (if keepdims then "true" else "false")
  | Arg_reduce { op; input; axis; keepdims } ->
      Printf.fprintf oc
        "{\"tag\":\"arg_reduce\",\"op\":\"%s\",\"input\":%d,\"axis\":%d,\
         \"keepdims\":%s}"
        (json_escape (op_name (Arg_reduce { op; input; axis; keepdims })))
        input axis
        (if keepdims then "true" else "false")
  | Reshape { input; shape } ->
      Printf.fprintf oc "{\"tag\":\"reshape\",\"input\":%d,\"shape\":" input;
      write_json_int_array oc shape;
      output_char oc '}'
  | Expand { input; shape } ->
      Printf.fprintf oc "{\"tag\":\"expand\",\"input\":%d,\"shape\":" input;
      write_json_int_array oc shape;
      output_char oc '}'
  | Permute { input; axes } ->
      Printf.fprintf oc "{\"tag\":\"permute\",\"input\":%d,\"axes\":" input;
      write_json_int_array oc axes;
      output_char oc '}'
  | Shrink { input; limits } ->
      Printf.fprintf oc "{\"tag\":\"shrink\",\"input\":%d,\"limits\":" input;
      write_json_limits oc limits;
      output_char oc '}'
  | Flip { input; dims } ->
      Printf.fprintf oc "{\"tag\":\"flip\",\"input\":%d,\"dims\":" input;
      write_json_bool_array oc dims;
      output_char oc '}'
  | Pad { input; padding; fill_value } ->
      Printf.fprintf oc
        "{\"tag\":\"pad\",\"input\":%d,\"padding\":"
        input;
      write_json_limits oc padding;
      Printf.fprintf oc ",\"fill_value\":\"%s\"}" (json_escape fill_value)
  | Cat { inputs; axis } ->
      Printf.fprintf oc "{\"tag\":\"cat\",\"axis\":%d,\"inputs\":" axis;
      write_json_list oc (fun oc x -> output_string oc (string_of_int x)) inputs;
      output_char oc '}'
  | Cast { input; dtype } ->
      Printf.fprintf oc "{\"tag\":\"cast\",\"input\":%d,\"dtype\":\"%s\"}" input
        (json_escape dtype)
  | Gather { data; indices; axis } ->
      Printf.fprintf oc
        "{\"tag\":\"gather\",\"data\":%d,\"indices\":%d,\"axis\":%d}" data
        indices axis
  | Matmul { lhs; rhs } ->
      Printf.fprintf oc "{\"tag\":\"matmul\",\"lhs\":%d,\"rhs\":%d}" lhs rhs
  | Custom_call { handler; inputs } ->
      Printf.fprintf oc
        "{\"tag\":\"custom_call\",\"library\":\"%s\",\"symbol\":\"%s\",\
         \"target\":\"%s\",\"inputs\":"
        (json_escape handler.library) (json_escape handler.symbol)
        (json_escape handler.target);
      write_json_list oc (fun oc x -> output_string oc (string_of_int x)) inputs;
      output_char oc '}'
  | Triton_call { kernel; inputs } ->
      Printf.fprintf oc
        "{\"tag\":\"triton_call\",\"name\":\"%s\",\"ir\":\"%s\",\
         \"num_warps\":%d,\"num_stages\":%d,\"grid\":[%d,%d,%d],\"inputs\":"
        (json_escape kernel.name) (json_escape kernel.ir) kernel.num_warps
        kernel.num_stages kernel.grid_x kernel.grid_y kernel.grid_z;
      write_json_list oc (fun oc x -> output_string oc (string_of_int x)) inputs;
      output_char oc '}'
  | Assign { dst; src } ->
      Printf.fprintf oc "{\"tag\":\"assign\",\"dst\":%d,\"src\":%d}" dst src
  | Unsupported name ->
      Printf.fprintf oc "{\"tag\":\"unsupported\",\"name\":\"%s\"}"
        (json_escape name)

let write_spec artifact_dir spec_path ~backend ~device_id signature program
    output_descs =
  write_file spec_path (fun oc ->
      output_string oc "{";
      Printf.fprintf oc "\"backend\":\"%s\",\"device_id\":%d,"
        (Backend.to_string backend) device_id;
      output_string oc "\"signature\":{\"inputs\":";
      write_json_list oc
        (fun oc (input : Signature.tensor) ->
          Printf.fprintf oc "{\"dtype\":\"%s\",\"shape\":"
            (json_escape input.dtype);
          write_json_int_array oc input.shape;
          output_char oc '}')
        signature.Signature.inputs;
      output_string oc "},";
      output_string oc "\"program\":{\"inputs\":";
      write_json_list oc (fun oc id -> output_string oc (string_of_int id))
        program.Ir.inputs;
      output_string oc ",\"outputs\":";
      write_json_list oc (fun oc id -> output_string oc (string_of_int id))
        program.Ir.outputs;
      output_string oc ",\"nodes\":";
      write_json_list oc
        (fun oc (node : Ir.node) ->
          Printf.fprintf oc "{\"id\":%d,\"dtype\":\"%s\",\"shape\":"
            node.Ir.id (json_escape node.Ir.desc.dtype);
          write_json_int_array oc node.Ir.desc.shape;
          output_string oc ",\"op\":";
          write_op_json artifact_dir oc node;
          output_char oc '}')
        program.Ir.nodes;
      output_string oc "},";
      output_string oc "\"outputs\":";
      write_json_list oc
        (fun oc (desc : Ir.desc) ->
          Printf.fprintf oc "{\"dtype\":\"%s\",\"shape\":"
            (json_escape desc.dtype);
          write_json_int_array oc desc.shape;
          output_char oc '}')
        output_descs;
      output_char oc '}')

let compile ~backend ~device_id ~signature program output_examples =
  let program = Ir.prune program in
  let program, lifted_constants = Ir.parameterize_constants program in
  validate_program program;
  let module_text = Stablehlo.of_program program in
  let output_descs = packed_output_descs output_examples in
  let extra_inputs =
    List.map
      (fun (lifted : Ir.lifted_constant) ->
        (lifted.desc, data_string_of_literal lifted.literal))
      lifted_constants
  in
  let cache_key = compiled_cache_key signature module_text extra_inputs in
  let artifact_dir = artifact_dir cache_key in
  let spec_path = Filename.concat artifact_dir "program.json" in
  let module_path = Filename.concat artifact_dir "program.mlir" in
  write_spec artifact_dir spec_path ~backend ~device_id signature program
    output_descs;
  write_file module_path (fun oc -> output_string oc module_text);
  if not (is_available ()) then
    Error.raise
      (Error.Runtime_unavailable
         (Printf.sprintf "%s (spec written to %s)" (status ()) spec_path));
  if not (backend_available backend) then
    Error.raise
      (Error.Runtime_unavailable
         (Printf.sprintf "PJRT %s plugin missing at %s"
            (Backend.to_string backend)
            (plugin_path backend)));
  let plugin_path = plugin_path backend in
  {
    backend;
    device_id;
    plugin_path;
    cache_key;
    signature;
    program;
    artifact_dir;
    spec_path;
    module_path;
    module_text;
    output_descs;
    extra_inputs;
    ffi_registered = false;
  }

let compile_stablehlo ~backend ~device_id ~signature ~module_text ~output_descs
    ~extra_inputs =
  let cache_key = compiled_cache_key signature module_text extra_inputs in
  let artifact_dir = artifact_dir cache_key in
  let spec_path = Filename.concat artifact_dir "program.json" in
  let module_path = Filename.concat artifact_dir "program.mlir" in
  let program = { Ir.name = Some "custom"; inputs = []; outputs = []; nodes = [] } in
  write_spec artifact_dir spec_path ~backend ~device_id signature program
    output_descs;
  write_file module_path (fun oc -> output_string oc module_text);
  if not (is_available ()) then
    Error.raise
      (Error.Runtime_unavailable
         (Printf.sprintf "%s (spec written to %s)" (status ()) spec_path));
  if not (backend_available backend) then
    Error.raise
      (Error.Runtime_unavailable
         (Printf.sprintf "PJRT %s plugin missing at %s"
            (Backend.to_string backend)
            (plugin_path backend)));
  let plugin_path = plugin_path backend in
  {
    backend;
    device_id;
    plugin_path;
    cache_key;
    signature;
    program;
    artifact_dir;
    spec_path;
    module_path;
    module_text;
    output_descs;
    extra_inputs;
    ffi_registered = false;
  }

let tensor_of_output desc =
  let shape = desc.Ir.shape in
  let numel = Array.fold_left ( * ) 1 shape in
  match Nx_core.Dtype.Packed.of_string desc.dtype with
  | None ->
      Error.raise
        (Error.Runtime_unavailable
           (Printf.sprintf "unknown dtype in output descriptor: %s" desc.dtype))
  | Some (Nx_core.Dtype.Pack dtype) ->
      let kind = Nx_core.Dtype.to_buffer_kind dtype in
      let buffer = Nx_buffer.create kind numel in
      Trace.Tensor (Nx.of_buffer buffer ~shape)

let ensure_ffi_registered compiled =
  if not compiled.ffi_registered then (
    let handlers = Ir.ffi_handlers compiled.program in
    if handlers <> [] && compiled.backend <> `Cuda then
      Error.raise
        (Error.Unsupported_program
           "typed PJRT FFI custom calls currently require the CUDA backend");
    List.iter
      (fun (handler : Ir.ffi_handler) ->
        let identity =
          Ffi.Internal.identity
            { library = handler.library; symbol = handler.symbol }
        in
        if
          identity.library_digest <> handler.library_digest
          || identity.target <> handler.target
        then
          Error.raise
            (Error.Runtime_unavailable
               (Printf.sprintf
                  "FFI library %S changed after tracing; retrace with an \
                   immutable or content-addressed artifact"
                  handler.library));
        native_register_ffi_handler
          ~plugin_path:compiled.plugin_path
          ~library_path:identity.library ~library_digest:identity.library_digest
          ~symbol:handler.symbol ~target:handler.target)
      handlers;
    compiled.ffi_registered <- true)

let constant_input_metadata compiled =
  let constant_input_dtypes =
    List.map (fun ((desc : Ir.desc), _) -> desc.dtype) compiled.extra_inputs
  in
  let constant_input_shapes =
    List.map
      (fun ((desc : Ir.desc), _) -> Array.copy desc.shape)
      compiled.extra_inputs
  in
  let constant_input_data = List.map snd compiled.extra_inputs in
  ( Array.of_list constant_input_dtypes,
    Array.of_list constant_input_shapes,
    Array.of_list constant_input_data )

let output_metadata compiled =
  let output_dtypes =
    Array.of_list
      (List.map (fun (desc : Ir.desc) -> desc.dtype) compiled.output_descs)
  in
  let output_shapes =
    Array.of_list
      (List.map
         (fun (desc : Ir.desc) -> Array.copy desc.shape)
         compiled.output_descs)
  in
  (output_dtypes, output_shapes)

let execute compiled inputs =
  ensure_ffi_registered compiled;
  let constant_input_dtypes, constant_input_shapes, constant_input_data =
    constant_input_metadata compiled
  in
  let dynamic_input_dtypes =
    Array.of_list
      (List.map
         (fun tensor -> Nx_core.Dtype.to_string (Nx.dtype tensor))
         inputs)
  in
  let dynamic_input_shapes =
    Array.of_list (List.map (fun tensor -> Nx.shape tensor) inputs)
  in
  let inputs = List.map Nx.contiguous inputs in
  let dynamic_input_data =
    Array.of_list (List.map (fun tensor -> Obj.repr (Nx.data tensor)) inputs)
  in
  let output_dtypes, output_shapes = output_metadata compiled in
  let outputs = List.map tensor_of_output compiled.output_descs in
  let output_data =
    Array.of_list
      (List.map
         (fun (Trace.Tensor tensor) -> Obj.repr (Nx.data tensor))
         outputs)
  in
  native_execute
    ~plugin_path:compiled.plugin_path
    ~cache_key:compiled.cache_key ~device_id:compiled.device_id
    ~stablehlo:compiled.module_text ~dynamic_input_dtypes ~dynamic_input_shapes
    ~dynamic_input_data ~constant_input_dtypes ~constant_input_shapes
    ~constant_input_data ~output_dtypes ~output_shapes ~output_data;
  outputs

let device_buffer_of_host ~backend ~device_id tensor =
  let tensor = Nx.contiguous tensor in
  let dtype = Nx.dtype tensor in
  let shape = Nx.shape tensor in
  let native =
    native_buffer_of_host ~plugin_path:(plugin_path backend) ~device_id
      ~dtype:(Nx_core.Dtype.to_string dtype)
      ~shape
      ~data:(Obj.repr (Nx.data tensor))
  in
  { native; backend; device_id; dtype; shape = Array.copy shape }

let device_buffer_to_host buffer =
  let shape = Array.copy buffer.shape in
  let numel = Array.fold_left ( * ) 1 shape in
  let kind = Nx_core.Dtype.to_buffer_kind buffer.dtype in
  let data = Nx_buffer.create kind numel in
  native_buffer_to_host buffer.native
    ~dtype:(Nx_core.Dtype.to_string buffer.dtype)
    ~shape ~data:(Obj.repr data);
  Nx.of_buffer data ~shape

let device_buffer_await buffer = native_buffer_await buffer.native
let device_buffer_shape buffer = Array.copy buffer.shape
let device_buffer_dtype buffer = buffer.dtype

let device_buffer_placeholder buffer =
  let scalar = Nx.scalar buffer.dtype (Nx_core.Dtype.zero buffer.dtype) in
  Nx.broadcast_to buffer.shape scalar

let device_buffers_match_signature signature inputs =
  let rec loop expected actual =
    match (expected, actual) with
    | [], [] -> true
    | [], _ :: _ | _ :: _, [] -> false
    | (spec : Signature.tensor) :: expected, Device_buffer buffer :: actual ->
        buffer.backend = signature.Signature.backend
        && buffer.device_id = signature.device_id
        && Nx_core.Dtype.to_string buffer.dtype = spec.dtype
        && Array.equal Int.equal buffer.shape spec.shape
        && loop expected actual
  in
  loop signature.inputs inputs

let validate_device_inputs compiled inputs =
  let rec loop index expected actual =
    match (expected, actual) with
    | [], [] -> ()
    | [], _ :: _ | _ :: _, [] ->
        invalid_arg
          (Printf.sprintf
             "Rune_pjrt: compiled function expects %d device inputs, got %d"
             (List.length compiled.signature.inputs)
             (List.length inputs))
    | (spec : Signature.tensor) :: expected, Device_buffer buffer :: actual ->
        if
          buffer.backend <> compiled.backend
          || buffer.device_id <> compiled.device_id
        then
          invalid_arg
            (Printf.sprintf
               "Rune_pjrt: device input %d belongs to %s:%d, expected %s:%d"
               index
               (Backend.to_string buffer.backend)
               buffer.device_id
               (Backend.to_string compiled.backend)
               compiled.device_id);
        let dtype = Nx_core.Dtype.to_string buffer.dtype in
        if dtype <> spec.dtype then
          invalid_arg
            (Printf.sprintf
               "Rune_pjrt: device input %d has dtype %s, expected %s" index
               dtype spec.dtype);
        if not (Array.equal Int.equal buffer.shape spec.shape) then
          invalid_arg
            (Printf.sprintf
               "Rune_pjrt: device input %d has shape %s, expected %s" index
               (Nx_core.Shape.to_string buffer.shape)
               (Nx_core.Shape.to_string spec.shape));
        loop (index + 1) expected actual
  in
  loop 0 compiled.signature.inputs inputs

let packed_device_buffer_of_native (compiled : compiled) (desc : Ir.desc) native
    =
  match Nx_core.Dtype.Packed.of_string desc.dtype with
  | None ->
      Error.raise
        (Error.Runtime_unavailable
           (Printf.sprintf "unknown dtype in output descriptor: %s" desc.dtype))
  | Some (Nx_core.Dtype.Pack dtype) ->
      Device_buffer
        {
          native;
          backend = compiled.backend;
          device_id = compiled.device_id;
          dtype;
          shape = Array.copy desc.shape;
        }

let execute_device compiled inputs =
  ensure_ffi_registered compiled;
  validate_device_inputs compiled inputs;
  let dynamic_input_buffers =
    Array.of_list
      (List.map (fun (Device_buffer buffer) -> buffer.native) inputs)
  in
  let constant_input_dtypes, constant_input_shapes, constant_input_data =
    constant_input_metadata compiled
  in
  let output_dtypes, output_shapes = output_metadata compiled in
  let native_outputs =
    native_execute_device
      ~plugin_path:compiled.plugin_path
      ~cache_key:compiled.cache_key ~device_id:compiled.device_id
      ~stablehlo:compiled.module_text ~dynamic_input_buffers
      ~constant_input_dtypes ~constant_input_shapes ~constant_input_data
      ~output_dtypes ~output_shapes
  in
  let output_descs = Array.of_list compiled.output_descs in
  if Array.length native_outputs <> Array.length output_descs then
    failwith "Rune_pjrt: PJRT returned the wrong number of device buffers";
  Array.mapi
    (fun index native ->
      packed_device_buffer_of_native compiled output_descs.(index) native)
    native_outputs
  |> Array.to_list
