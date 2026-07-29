(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Rune_pjrt.Triton.Dsl

type implementation = Baseline | Dsl
type execution_mode = Host | Resident
type action = Profile of implementation * execution_mode | Validate

type workload = {
  name : string;
  shape : string;
  schedule : string;
  inputs : Nx.float32_t list;
  body : Nx.float32_t list -> Nx.float32_t list;
}

let failf fmt = Printf.ksprintf failwith fmt

let require_power_of_two name value =
  if value <= 0 || value land (value - 1) <> 0 then
    invalid_arg (Printf.sprintf "%s must be a positive power of two" name)

let next_power_of_two value =
  if value <= 0 then invalid_arg "next_power_of_two expects a positive value";
  let rec loop result = if result >= value then result else loop (result * 2) in
  loop 1

let int_of_env name ~default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value -> int_of_string value

let profile_num_warps () =
  let num_warps = int_of_env "RUNE_TRITON_PROFILE_WARPS" ~default:4 in
  require_power_of_two "RUNE_TRITON_PROFILE_WARPS" num_warps;
  num_warps

let one_input name function_ = function
  | [ input ] -> [ function_ input ]
  | _ -> failf "%s expects one input" name

let two_inputs name function_ = function
  | [ lhs; rhs ] -> [ function_ lhs rhs ]
  | _ -> failf "%s expects two inputs" name

let three_inputs name function_ = function
  | [ input; scale; bias ] -> [ function_ input scale bias ]
  | _ -> failf "%s expects three inputs" name

let pointwise input =
  Nx.sigmoid
    (Nx.add (Nx.mul input input) (Nx.mul input (Nx.scalar_like input 0.5)))

let pointwise_scalar input =
  let value =
    Nx.add (Nx.mul input input) (Nx.mul input (Nx.scalar_like input 0.5))
  in
  let one = Nx.scalar_like value 1.0 in
  Nx.div one (Nx.add one (Nx.exp (Nx.neg value)))

let make_pointwise_kernel ~block_size ~num_warps =
  require_power_of_two "pointwise block size" block_size;
  let config = D.Config.make ~block_size ~num_warps () in
  let definition =
    D.Kernel.define ~name:"profile_pointwise"
      ~signature:D.Signature.(f32 @-> returning f32)
      ~config
      ~guard:(fun spec -> D.Spec.input_numel spec 0 = D.Spec.output_numel spec)
      ~grid:(fun spec ->
        let numel = D.Spec.output_numel spec in
        ((numel + block_size - 1) / block_size, 1, 1))
      (fun%rune.kernel spec input output ->
        let block = D.Value.int D.Dtype.i32 block_size in
        let offsets =
          (D.Value.program_id D.X * block)
          + D.Value.arange ~start:0 ~stop:block_size
        in
        let bound = D.Value.int D.Dtype.i32 (D.Spec.output_numel spec) in
        let mask = offsets < bound in
        let values =
          D.Pointer.load ~mask
            ~other:(D.Value.zeros D.Dtype.f32 ~shape:[| block_size |])
            (D.Pointer.offset input offsets)
        in
        let result = D.Value.sigmoid ((values * values) + (values * 0.5)) in
        [ D.Statement.store ~mask (D.Pointer.offset output offsets) result ])
  in
  D.Kernel.bind definition ~fallback:pointwise

let make_pointwise_workload ?(name = "pointwise") ?(baseline = pointwise)
    implementation ~size ~configuration =
  let num_warps = profile_num_warps () in
  let input =
    Nx.init Nx.float32 [| size |] (fun indices ->
        (float_of_int (indices.(0) mod 257) -. 128.) /. 128.)
  in
  let function_ =
    match implementation with
    | Baseline -> baseline
    | Dsl -> make_pointwise_kernel ~block_size:configuration ~num_warps
  in
  {
    name;
    shape = Printf.sprintf "[%d]" size;
    schedule =
      Printf.sprintf "block_size=%d,num_warps=%d" configuration num_warps;
    inputs = [ input ];
    body = one_input "pointwise" function_;
  }

let softmax input = Nx.softmax ~axes:[ 1 ] input

let make_softmax_kernel ~width ~rows_per_program ~num_warps =
  require_power_of_two "softmax rows per program" rows_per_program;
  let block_width = next_power_of_two width in
  let config =
    D.Config.make ~block_size:(block_width * rows_per_program) ~num_warps ()
  in
  let definition =
    D.Kernel.define ~name:"profile_softmax"
      ~signature:D.Signature.(f32 @-> returning f32)
      ~config
      ~guard:(fun spec ->
        let input = D.Spec.input_shape spec 0 in
        let output = D.Spec.output_shape spec in
        input = output
        && Array.length input = 2
        && input.(0) mod rows_per_program = 0
        && input.(1) = width)
      ~grid:(fun spec ->
        let shape = D.Spec.output_shape spec in
        (shape.(0) / rows_per_program, 1, 1))
      (fun%rune.kernel spec input output ->
        let row_lanes = D.Value.arange ~start:0 ~stop:rows_per_program in
        let row_tile = D.Value.int D.Dtype.i32 rows_per_program in
        let row =
          (D.Value.program_id D.X * row_tile) + row_lanes
          |> D.Value.expand_dims ~axis:1
        in
        let columns =
          D.Value.arange ~start:0 ~stop:block_width
          |> D.Value.expand_dims ~axis:0
        in
        let row_width = D.Value.int D.Dtype.i32 width in
        let offsets = (row * row_width) + columns in
        let column_mask = columns < row_width in
        let values =
          D.Pointer.load ~mask:column_mask
            ~other:(D.Value.float D.Dtype.f32 (-1e30))
            (D.Pointer.offset input offsets)
        in
        [
          D.Statement.static_assert
            [%rune.host D.Spec.input_count spec = 1]
            "expected one input";
          D.Statement.store ~mask:column_mask
            (D.Pointer.offset output offsets)
            (D.Value.softmax ~axis:1 values);
        ])
  in
  D.Kernel.bind definition ~fallback:softmax

let rows_for_width width = Int.max 1 (4 * 1024 * 1024 / width)

let make_softmax_workload implementation ~width ~configuration =
  let rows = rows_for_width width in
  let num_warps = profile_num_warps () in
  let input =
    Nx.init Nx.float32 [| rows; width |] (fun indices ->
        float_of_int ((indices.(0) * 17) - (indices.(1) * 13)) /. 256.)
  in
  let function_ =
    match implementation with
    | Baseline -> softmax
    | Dsl ->
        make_softmax_kernel ~width ~rows_per_program:configuration ~num_warps
  in
  {
    name = "softmax";
    shape = Printf.sprintf "[%d,%d]" rows width;
    schedule =
      Printf.sprintf "rows_per_program=%d,num_warps=%d" configuration num_warps;
    inputs = [ input ];
    body = one_input "softmax" function_;
  }

let layer_norm input scale bias =
  let mean = Nx.mean ~axes:[ 1 ] ~keepdims:true input in
  let centered = Nx.sub input mean in
  let variance =
    Nx.mean ~axes:[ 1 ] ~keepdims:true (Nx.mul centered centered)
  in
  let normalized =
    Nx.mul centered (Nx.rsqrt (Nx.add variance (Nx.scalar_like variance 1e-5)))
  in
  Nx.add (Nx.mul normalized scale) bias

let make_layer_norm_kernel ~width ~rows_per_program ~num_warps =
  require_power_of_two "layer norm rows per program" rows_per_program;
  let block_width = next_power_of_two width in
  let config =
    D.Config.make ~block_size:(block_width * rows_per_program) ~num_warps ()
  in
  let definition =
    D.Kernel.define ~name:"profile_layer_norm"
      ~signature:D.Signature.(f32 @-> f32 @-> f32 @-> returning f32)
      ~config
      ~guard:(fun spec ->
        let input = D.Spec.input_shape spec 0 in
        let scale = D.Spec.input_shape spec 1 in
        let bias = D.Spec.input_shape spec 2 in
        input = D.Spec.output_shape spec
        && Array.length input = 2
        && input.(0) mod rows_per_program = 0
        && input.(1) = width
        && scale = [| width |] && bias = [| width |])
      ~grid:(fun spec ->
        let shape = D.Spec.output_shape spec in
        (shape.(0) / rows_per_program, 1, 1))
      (fun%rune.kernel spec input scale bias output ->
        let columns =
          D.Value.arange ~start:0 ~stop:block_width
          |> D.Value.expand_dims ~axis:0
        in
        let row_lanes = D.Value.arange ~start:0 ~stop:rows_per_program in
        let row_tile = D.Value.int D.Dtype.i32 rows_per_program in
        let row =
          (D.Value.program_id D.X * row_tile) + row_lanes
          |> D.Value.expand_dims ~axis:1
        in
        let row_width = D.Value.int D.Dtype.i32 width in
        let offsets = (row * row_width) + columns in
        let column_mask = columns < row_width in
        let zero = D.Value.float D.Dtype.f32 0.0 in
        let values =
          D.Pointer.load ~mask:column_mask ~other:zero
            (D.Pointer.offset input offsets)
        in
        let divisor = D.Value.float D.Dtype.f32 (float_of_int width) in
        let mean = D.Value.sum ~keep_dims:true ~axis:1 values / divisor in
        let centered = D.Value.where column_mask (values - mean) zero in
        let variance =
          D.Value.sum ~keep_dims:true ~axis:1 (centered * centered) / divisor
        in
        let normalized = centered * D.Value.rsqrt (variance + 1e-5) in
        let scales =
          D.Pointer.load ~mask:column_mask ~other:zero
            (D.Pointer.offset scale columns)
        in
        let biases =
          D.Pointer.load ~mask:column_mask ~other:zero
            (D.Pointer.offset bias columns)
        in
        [
          D.Statement.static_assert
            [%rune.host D.Spec.input_count spec = 3]
            "expected three inputs";
          D.Statement.store ~mask:column_mask
            (D.Pointer.offset output offsets)
            ((normalized * scales) + biases);
        ])
  in
  D.Kernel.bind definition ~fallback:layer_norm

let make_layer_norm_workload implementation ~width ~configuration =
  let rows = rows_for_width width in
  let num_warps = profile_num_warps () in
  let input =
    Nx.init Nx.float32 [| rows; width |] (fun indices ->
        float_of_int ((indices.(0) * 7) - (indices.(1) * 11)) /. 512.)
  in
  let scale =
    Nx.init Nx.float32 [| width |] (fun indices ->
        0.75 +. (float_of_int (indices.(0) mod 31) /. 64.))
  in
  let bias =
    Nx.init Nx.float32 [| width |] (fun indices ->
        float_of_int ((indices.(0) mod 17) - 8) /. 32.)
  in
  let function_ =
    match implementation with
    | Baseline -> layer_norm
    | Dsl ->
        make_layer_norm_kernel ~width ~rows_per_program:configuration ~num_warps
  in
  {
    name = "layer_norm";
    shape = Printf.sprintf "[%d,%d]" rows width;
    schedule =
      Printf.sprintf "rows_per_program=%d,num_warps=%d" configuration num_warps;
    inputs = [ input; scale; bias ];
    body = three_inputs "layer_norm" function_;
  }

let matmul lhs rhs = Nx.matmul lhs rhs

let make_gemm_kernel ~size ~tile ~num_warps =
  if tile < 16 || tile mod 16 <> 0 || size <= 0 || size mod tile <> 0 then
    invalid_arg
      "GEMM tile must be a multiple of 16 that evenly divides the positive \
       matrix size";
  require_power_of_two "GEMM tile" tile;
  let config =
    D.Config.make ~block_size:(tile * tile) ~num_warps ~num_stages:2 ()
  in
  let definition =
    D.Kernel.define ~name:"profile_gemm"
      ~signature:D.Signature.(f32 @-> f32 @-> returning f32)
      ~config
      ~guard:(fun spec ->
        D.Spec.input_shape spec 0 = [| size; size |]
        && D.Spec.input_shape spec 1 = [| size; size |]
        && D.Spec.output_shape spec = [| size; size |])
      ~grid:(fun _ -> (size / tile, size / tile, 1))
      (fun%rune.kernel spec lhs rhs output ->
        let lanes = D.Value.arange ~start:0 ~stop:tile in
        let tile_value = D.Value.int D.Dtype.i32 tile in
        let size_value = D.Value.int D.Dtype.i32 size in
        let rows =
          (D.Value.program_id D.X * tile_value) + lanes
          |> D.Value.expand_dims ~axis:1
        in
        let columns =
          (D.Value.program_id D.Y * tile_value) + lanes
          |> D.Value.expand_dims ~axis:0
        in
        let accumulator =
          D.Value.range
            ~start:(D.Value.int D.Dtype.i32 0)
            ~stop:size_value ~step:tile_value
            ~init:(D.Value.zeros D.Dtype.f32 ~shape:[| tile; tile |])
            (fun k_start accumulator ->
              let inner = k_start + lanes in
              let lhs_offsets =
                (rows * size_value) + D.Value.expand_dims ~axis:0 inner
              in
              let rhs_offsets =
                (D.Value.expand_dims ~axis:1 inner * size_value) + columns
              in
              let lhs_tile =
                D.Pointer.load (D.Pointer.offset lhs lhs_offsets)
              in
              let rhs_tile =
                D.Pointer.load (D.Pointer.offset rhs rhs_offsets)
              in
              D.Value.dot lhs_tile rhs_tile accumulator)
        in
        let output_offsets = (rows * size_value) + columns in
        [
          D.Statement.static_assert
            [%rune.host D.Spec.input_count spec = 2]
            "expected two inputs";
          D.Statement.store (D.Pointer.offset output output_offsets) accumulator;
        ])
  in
  D.Kernel.bind definition ~fallback:matmul

let make_gemm_workload implementation ~size ~configuration =
  let tile = int_of_env "RUNE_TRITON_PROFILE_GEMM_TILE" ~default:16 in
  let lhs =
    Nx.init Nx.float32 [| size; size |] (fun indices ->
        float_of_int ((indices.(0) * 3) - indices.(1)) /. float_of_int size)
  in
  let rhs =
    Nx.init Nx.float32 [| size; size |] (fun indices ->
        float_of_int (indices.(0) + (indices.(1) * 2)) /. float_of_int size)
  in
  let function_ =
    match implementation with
    | Baseline -> matmul
    | Dsl -> make_gemm_kernel ~size ~tile ~num_warps:configuration
  in
  {
    name = "gemm";
    shape = Printf.sprintf "[%d,%d]x[%d,%d]" size size size size;
    schedule =
      Printf.sprintf "tile=%dx%dx%d,num_warps=%d,num_stages=2" tile tile tile
        configuration;
    inputs = [ lhs; rhs ];
    body = two_inputs "gemm" function_;
  }

let parse_action = function
  | "baseline" -> Profile (Baseline, Host)
  | "dsl" -> Profile (Dsl, Host)
  | "baseline_resident" -> Profile (Baseline, Resident)
  | "dsl_resident" -> Profile (Dsl, Resident)
  | "validate" -> Validate
  | value -> invalid_arg ("unknown action " ^ value)

let parse_arguments () =
  if Array.length Sys.argv <> 7 then
    failwith
      "usage: triton_profile.exe \
       (pointwise|pointwise_scalar|softmax|layer_norm|gemm|suite) \
       (baseline|dsl|baseline_resident|dsl_resident|validate) SIZE \
       CONFIGURATION WARMUPS ITERATIONS";
  let case = Sys.argv.(1) in
  let action = parse_action Sys.argv.(2) in
  let size = int_of_string Sys.argv.(3) in
  let configuration = int_of_string Sys.argv.(4) in
  let warmups = int_of_string Sys.argv.(5) in
  let iterations = int_of_string Sys.argv.(6) in
  if size <= 0 || configuration <= 0 || warmups < 0 || iterations <= 0 then
    invalid_arg
      "size, configuration, and iterations must be positive; warmups may be \
       zero";
  (case, action, size, configuration, warmups, iterations)

let make_workload case implementation ~size ~configuration =
  match case with
  | "pointwise" -> make_pointwise_workload implementation ~size ~configuration
  | "pointwise_scalar" ->
      make_pointwise_workload ~name:"pointwise_scalar"
        ~baseline:pointwise_scalar implementation ~size ~configuration
  | "softmax" -> make_softmax_workload implementation ~width:size ~configuration
  | "layer_norm" ->
      make_layer_norm_workload implementation ~width:size ~configuration
  | "gemm" -> make_gemm_workload implementation ~size ~configuration
  | value -> invalid_arg ("unknown benchmark " ^ value)

let percentile sorted fraction =
  let last = Array.length sorted - 1 in
  sorted.(Int.min last (int_of_float (fraction *. float_of_int last)))

let elapsed_ms started = (Unix.gettimeofday () -. started) *. 1000.

let sole_output workload = function
  | [ output ] -> output
  | _ -> failf "%s returned the wrong number of outputs" workload.name

let implementation_name = function Baseline -> "baseline" | Dsl -> "dsl"

let report_samples samples =
  let iterations = Array.length samples in
  let sorted = Array.copy samples in
  Array.sort Float.compare sorted;
  let mean = Array.fold_left ( +. ) 0. samples /. float_of_int iterations in
  Printf.printf
    "steady_e2e_ms mean=%.6f p10=%.6f median=%.6f p90=%.6f min=%.6f max=%.6f\n\
     %!"
    mean (percentile sorted 0.10) (percentile sorted 0.50)
    (percentile sorted 0.90) sorted.(0)
    sorted.(iterations - 1)

let report_checksum output =
  let values = Nx.to_array output in
  let checksum = ref 0.0 in
  for index = 0 to Array.length values - 1 do
    checksum := !checksum +. values.(index)
  done;
  Printf.printf "checksum=%.17g\n%!" !checksum

let run_host workload implementation warmups iterations =
  let compiled = Rune_pjrt.jits ~backend:`Cuda workload.body in
  Printf.printf
    "case=%s implementation=%s mode=host shape=%s schedule=%s warmups=%d \
     iterations=%d\n\
     %!"
    workload.name
    (implementation_name implementation)
    workload.shape workload.schedule warmups iterations;
  let started = Unix.gettimeofday () in
  let first = compiled workload.inputs in
  Printf.printf "first_compile_and_execute_ms=%.6f\n%!" (elapsed_ms started);
  let last = ref first in
  for _ = 1 to warmups do
    last := compiled workload.inputs
  done;
  let samples = Array.make iterations 0. in
  for index = 0 to iterations - 1 do
    let started = Unix.gettimeofday () in
    last := compiled workload.inputs;
    samples.(index) <- elapsed_ms started
  done;
  report_samples samples;
  report_checksum (sole_output workload !last)

let await_device_outputs outputs =
  List.iter Rune_pjrt.Device_buffer.await outputs

let run_resident workload implementation warmups iterations =
  let compiled = Rune_pjrt.jits_device ~backend:`Cuda workload.body in
  let started = Unix.gettimeofday () in
  let inputs =
    List.map (Rune_pjrt.Device_buffer.of_host ~backend:`Cuda) workload.inputs
  in
  await_device_outputs inputs;
  Printf.printf "initial_device_put_ms=%.6f\n%!" (elapsed_ms started);
  Printf.printf
    "case=%s implementation=%s mode=resident shape=%s schedule=%s warmups=%d \
     iterations=%d\n\
     %!"
    workload.name
    (implementation_name implementation)
    workload.shape workload.schedule warmups iterations;
  let started = Unix.gettimeofday () in
  let first = compiled inputs in
  await_device_outputs first;
  Printf.printf "first_compile_and_execute_ms=%.6f\n%!" (elapsed_ms started);
  let last = ref first in
  for _ = 1 to warmups do
    last := compiled inputs;
    await_device_outputs !last
  done;
  let samples = Array.make iterations 0. in
  let dispatch_samples = Array.make iterations 0. in
  let await_samples = Array.make iterations 0. in
  for index = 0 to iterations - 1 do
    let started = Unix.gettimeofday () in
    last := compiled inputs;
    let dispatched = Unix.gettimeofday () in
    await_device_outputs !last;
    let finished = Unix.gettimeofday () in
    dispatch_samples.(index) <- (dispatched -. started) *. 1000.;
    await_samples.(index) <- (finished -. dispatched) *. 1000.;
    samples.(index) <- (finished -. started) *. 1000.
  done;
  report_samples samples;
  let mean values =
    Array.fold_left ( +. ) 0. values /. float_of_int iterations
  in
  Printf.printf "resident_breakdown_ms dispatch=%.6f await=%.6f\n%!"
    (mean dispatch_samples) (mean await_samples);
  let output = sole_output workload !last |> Rune_pjrt.Device_buffer.to_host in
  report_checksum output

let run workload implementation mode warmups iterations =
  match mode with
  | Host -> run_host workload implementation warmups iterations
  | Resident -> run_resident workload implementation warmups iterations

let validate case ~size ~configuration =
  let baseline = make_workload case Baseline ~size ~configuration in
  let dsl = make_workload case Dsl ~size ~configuration in
  let compile_started = Unix.gettimeofday () in
  let baseline_compiled = Rune_pjrt.jits ~backend:`Cuda baseline.body in
  let dsl_compiled = Rune_pjrt.jits ~backend:`Cuda dsl.body in
  let expected =
    baseline_compiled baseline.inputs |> sole_output baseline |> Nx.to_array
  in
  let actual = dsl_compiled baseline.inputs |> sole_output dsl |> Nx.to_array in
  if Array.length expected <> Array.length actual then
    failf "validation output lengths differ: %d versus %d"
      (Array.length expected) (Array.length actual);
  let max_abs_error = ref 0.0 in
  let max_rel_error = ref 0.0 in
  let squared_error = ref 0.0 in
  for index = 0 to Array.length expected - 1 do
    let error = Float.abs (actual.(index) -. expected.(index)) in
    let scale = Float.max 1e-6 (Float.abs expected.(index)) in
    max_abs_error := Float.max !max_abs_error error;
    max_rel_error := Float.max !max_rel_error (error /. scale);
    squared_error := !squared_error +. (error *. error)
  done;
  let rmse =
    Float.sqrt (!squared_error /. float_of_int (Array.length expected))
  in
  Printf.printf
    "case=%s action=validate shape=%s schedule=%s\n\
     compile_and_execute_both_ms=%.6f\n\
     elements=%d max_abs_error=%.9g max_rel_error=%.9g rmse=%.9g\n\
     %!"
    baseline.name baseline.shape dsl.schedule
    (elapsed_ms compile_started)
    (Array.length expected) !max_abs_error !max_rel_error rmse

let () =
  let case, action, size, configuration, warmups, iterations =
    parse_arguments ()
  in
  match action with
  | Profile (implementation, mode) when case = "suite" ->
      [
        make_pointwise_workload implementation ~size:1_048_576
          ~configuration:256;
        make_softmax_workload implementation ~width:768 ~configuration:1;
        make_layer_norm_workload implementation ~width:768 ~configuration:1;
        make_gemm_workload implementation ~size:1024 ~configuration:4;
      ]
      |> List.iter (fun workload ->
          run workload implementation mode warmups iterations)
  | Profile (implementation, mode) ->
      let workload = make_workload case implementation ~size ~configuration in
      run workload implementation mode warmups iterations
  | Validate when case = "suite" ->
      invalid_arg "suite does not support validation"
  | Validate -> validate case ~size ~configuration
