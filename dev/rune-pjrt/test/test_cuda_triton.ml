(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let ttir =
  {|
module {
  tt.func public @raven_add_one(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>) {
    %value = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    %one = arith.constant 1.000000e+00 : f32
    %result = arith.addf %value, %one : f32
    tt.store %arg1, %result {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
    tt.return
  }
}
|}

let kernel =
  Rune_pjrt.Triton.Kernel.create ~name:"raven_add_one" ~ir:ttir ~num_warps:1
    ~num_stages:1 ()

let add_one x =
  Rune_pjrt.Triton.call kernel ~inputs:[ Rune_pjrt.Triton.Tensor x ]
    ~fallback:(fun () -> Nx.add x (Nx.scalar_like x 1.0))

module D = Rune_pjrt.Triton.Dsl

let define_unary_kernel (type a) ~name (dtype : a D.Dtype.t)
    ?(config = D.Config.make ()) build =
  let block_size = D.Config.block_size config in
  D.Kernel.define ~name
    ~signature:D.Signature.(dtype @-> returning dtype)
    ~config
    ~guard:(fun spec -> D.Spec.input_numel spec 0 = D.Spec.output_numel spec)
    ~grid:(fun spec ->
      let numel = D.Spec.output_numel spec in
      ((numel + block_size - 1) / block_size, 1, 1))
    (fun spec input output ->
      let offsets =
        D.Value.add
          (D.Value.mul (D.Value.program_id D.X)
             (D.Value.int D.Dtype.i32 block_size))
          (D.Value.arange ~start:0 ~stop:block_size)
      in
      let mask =
        D.Value.less offsets
          (D.Value.int D.Dtype.i32 (D.Spec.output_numel spec))
      in
      let values =
        D.Pointer.load ~mask
          ~other:(D.Value.zeros dtype ~shape:[| block_size |])
          (D.Pointer.offset input offsets)
      in
      [
        D.Statement.store ~mask (D.Pointer.offset output offsets) (build values);
      ])

let dsl_kernel =
  let config = D.Config.make ~block_size:128 ~num_warps:4 () in
  define_unary_kernel ~name:"raven_square_plus_one" D.Dtype.f32 ~config
    (fun%rune.kernel input -> (input * input) + 1.0)

let dsl_f16_kernel =
  let config = D.Config.make ~block_size:128 ~num_warps:4 () in
  define_unary_kernel ~name:"raven_f16_square_plus_one" D.Dtype.f16 ~config
    (fun%rune.kernel input -> (input * input) + 1.0)

let dsl_math_kernel =
  define_unary_kernel ~name:"raven_math" D.Dtype.f32 (fun input ->
      D.Value.add (D.Value.erf input) (D.Value.log2 (D.Value.exp2 input)))

let dsl_extended_math_kernel =
  define_unary_kernel ~name:"raven_extended_math" D.Dtype.f32 (fun x ->
      let one = D.Value.float D.Dtype.f32 1. in
      let trigonometry = D.Value.add (D.Value.sin x) (D.Value.cos x) in
      let rounding = D.Value.add (D.Value.floor x) (D.Value.ceil x) in
      let inverse_root = D.Value.rsqrt (D.Value.add x one) in
      let logarithm = D.Value.log (D.Value.exp x) in
      let precise =
        D.Value.div_rn (D.Value.fma x x one)
          (D.Value.add (D.Value.sqrt_rn x) one)
      in
      D.Value.add trigonometry
        (D.Value.add rounding
           (D.Value.add inverse_root (D.Value.add logarithm precise))))

let general_abs_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  let config = D.Config.make ~block_size:128 ~num_warps:4 () in
  D.Kernel.define ~name:"raven_general_abs"
    ~signature:D.Signature.(f32 @-> returning f32)
    ~config
    ~guard:(fun spec -> D.Spec.input_numel spec 0 = D.Spec.output_numel spec)
    ~grid:(fun spec ->
      let numel = D.Spec.output_numel spec in
      ((numel + 127) / 128, 1, 1))
    (fun spec input output ->
      let program = D.Value.program_id D.X in
      let offsets =
        D.Value.add
          (D.Value.mul program (D.Value.int D.Dtype.i32 128))
          (D.Value.arange ~start:0 ~stop:128)
      in
      let mask =
        D.Value.less offsets
          (D.Value.int D.Dtype.i32 (D.Spec.output_numel spec))
      in
      let zero = D.Value.zeros D.Dtype.f32 ~shape:[| 128 |] in
      let values =
        D.Pointer.load ~mask ~other:zero (D.Pointer.offset input offsets)
      in
      let output_values =
        D.Value.where
          (D.Value.less values (D.Value.float D.Dtype.f32 0.))
          (D.Value.neg values) values
      in
      [
        D.Statement.store ~mask (D.Pointer.offset output offsets) output_values;
      ])

let reduce_sum_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  let config = D.Config.make ~block_size:128 ~num_warps:4 () in
  D.Kernel.define ~name:"raven_reduce_sum"
    ~signature:D.Signature.(f32 @-> returning f32)
    ~config
    ~guard:(fun spec ->
      let input = D.Spec.input_shape spec 0 in
      Array.length input = 2
      && input.(1) = 128
      && input.(0) = D.Spec.output_numel spec)
    ~grid:(fun spec -> (D.Spec.output_numel spec, 1, 1))
    (fun spec input output ->
      let row = D.Value.program_id D.X in
      let offsets =
        D.Value.add
          (D.Value.mul row (D.Value.int D.Dtype.i32 128))
          (D.Value.arange ~start:0 ~stop:128)
      in
      let values = D.Pointer.load (D.Pointer.offset input offsets) in
      [
        D.Statement.static_assert
          (D.Spec.input_count spec = 1)
          "expected one input";
        D.Statement.store
          (D.Pointer.offset output row)
          (D.Value.sum ~axis:0 values);
      ])

let dot_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  let config = D.Config.make ~block_size:256 ~num_warps:4 () in
  D.Kernel.define ~name:"raven_dot_16"
    ~signature:D.Signature.(f16 @-> f16 @-> returning f32)
    ~config
    ~guard:(fun spec ->
      D.Spec.input_shape spec 0 = [| 16; 16 |]
      && D.Spec.input_shape spec 1 = [| 16; 16 |]
      && D.Spec.output_shape spec = [| 16; 16 |])
    ~grid:(fun _ -> (1, 1, 1))
    (fun spec lhs rhs output ->
      let offsets = D.Value.arange ~start:0 ~stop:256 in
      let lhs =
        D.Pointer.load (D.Pointer.offset lhs offsets)
        |> D.Value.reshape ~shape:[| 16; 16 |]
      in
      let rhs =
        D.Pointer.load (D.Pointer.offset rhs offsets)
        |> D.Value.reshape ~shape:[| 16; 16 |]
      in
      let result =
        D.Value.dot lhs rhs (D.Value.zeros D.Dtype.f32 ~shape:[| 16; 16 |])
        |> D.Value.reshape ~shape:[| 256 |]
      in
      [
        D.Statement.static_assert
          (D.Spec.input_count spec = 2)
          "expected two inputs";
        D.Statement.store (D.Pointer.offset output offsets) result;
      ])

let softmax_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  let config = D.Config.make ~block_size:128 ~num_warps:4 () in
  D.Kernel.define ~name:"raven_softmax_rows"
    ~signature:D.Signature.(f32 @-> returning f32)
    ~config
    ~guard:(fun spec ->
      let input = D.Spec.input_shape spec 0 in
      let output = D.Spec.output_shape spec in
      input = output && Array.length input = 2 && input.(1) = 128)
    ~grid:(fun spec ->
      let shape = D.Spec.output_shape spec in
      (shape.(0), 1, 1))
    (fun%rune.kernel spec input output ->
      let row = D.Value.program_id D.X in
      let offsets = (row * 128) + D.Value.arange ~start:0 ~stop:128 in
      let values = D.Pointer.load (D.Pointer.offset input offsets) in
      [
        D.Statement.static_assert
          [%rune.host D.Spec.input_count spec = 1]
          "expected one input";
        D.Statement.store
          (D.Pointer.offset output offsets)
          (D.Value.softmax ~axis:0 values);
      ])

let transpose_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  D.Kernel.define ~name:"raven_transpose_16"
    ~signature:D.Signature.(f32 @-> returning f32)
    ~guard:(fun spec ->
      D.Spec.input_shape spec 0 = [| 16; 16 |]
      && D.Spec.output_shape spec = [| 16; 16 |])
    ~grid:(fun _ -> (1, 1, 1))
    (fun spec input output ->
      let offsets = D.Value.arange ~start:0 ~stop:256 in
      let values =
        D.Pointer.load (D.Pointer.offset input offsets)
        |> D.Value.reshape ~shape:[| 16; 16 |]
        |> D.Value.permute ~order:[| 1; 0 |]
        |> D.Value.reshape ~shape:[| 256 |]
      in
      [
        D.Statement.static_assert
          (D.Spec.input_count spec = 1)
          "expected one input";
        D.Statement.store (D.Pointer.offset output offsets) values;
      ])

let cast_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  let config = D.Config.make ~block_size:128 ~num_warps:4 () in
  D.Kernel.define ~name:"raven_cast_f16_f32"
    ~signature:D.Signature.(f16 @-> returning f32)
    ~config
    ~guard:(fun spec -> D.Spec.input_numel spec 0 = D.Spec.output_numel spec)
    ~grid:(fun spec ->
      let numel = D.Spec.output_numel spec in
      ((numel + 127) / 128, 1, 1))
    (fun spec input output ->
      let offsets =
        D.Value.add
          (D.Value.mul (D.Value.program_id D.X) (D.Value.int D.Dtype.i32 128))
          (D.Value.arange ~start:0 ~stop:128)
      in
      let mask =
        D.Value.less offsets
          (D.Value.int D.Dtype.i32 (D.Spec.output_numel spec))
      in
      let values =
        D.Pointer.load ~mask
          ~other:(D.Value.zeros D.Dtype.f16 ~shape:[| 128 |])
          (D.Pointer.offset input offsets)
        |> D.Value.cast D.Dtype.f32
      in
      [ D.Statement.store ~mask (D.Pointer.offset output offsets) values ])

let integer_kernel =
  define_unary_kernel ~name:"raven_integer_add" D.Dtype.i32 (fun input ->
      D.Value.add input (D.Value.int D.Dtype.i32 7))

let loop_sum_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  D.Kernel.define ~name:"raven_loop_sum"
    ~signature:D.Signature.(f32 @-> returning f32)
    ~guard:(fun spec ->
      D.Spec.input_numel spec 0 = 128 && D.Spec.output_numel spec = 1)
    ~grid:(fun _ -> (1, 1, 1))
    (fun spec input output ->
      let sum =
        D.Value.range ~start:(D.Value.int D.Dtype.i32 0)
          ~stop:(D.Value.int D.Dtype.i32 128)
          ~init:(D.Value.float D.Dtype.f32 0.) (fun index accumulator ->
            D.Value.add accumulator
              (D.Pointer.load (D.Pointer.offset input index)))
      in
      [
        D.Statement.static_assert
          (D.Spec.input_count spec = 1)
          "expected one input";
        D.Statement.store output sum;
      ])

let tiled_gemm_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  let config = D.Config.make ~block_size:256 ~num_warps:4 ~num_stages:2 () in
  D.Kernel.define ~name:"raven_tiled_gemm_32"
    ~signature:D.Signature.(f16 @-> f16 @-> returning f32)
    ~config
    ~guard:(fun spec ->
      D.Spec.input_shape spec 0 = [| 32; 32 |]
      && D.Spec.input_shape spec 1 = [| 32; 32 |]
      && D.Spec.output_shape spec = [| 32; 32 |])
    ~grid:(fun _ -> (2, 2, 1))
    (fun spec lhs rhs output ->
      let lanes = D.Value.arange ~start:0 ~stop:16 in
      let rows =
        D.Value.add
          (D.Value.mul (D.Value.program_id D.X) (D.Value.int D.Dtype.i32 16))
          lanes
        |> D.Value.expand_dims ~axis:1
      in
      let columns =
        D.Value.add
          (D.Value.mul (D.Value.program_id D.Y) (D.Value.int D.Dtype.i32 16))
          lanes
        |> D.Value.expand_dims ~axis:0
      in
      let accumulator =
        D.Value.range
          ~start:(D.Value.int D.Dtype.i32 0)
          ~stop:(D.Value.int D.Dtype.i32 32)
          ~step:(D.Value.int D.Dtype.i32 16)
          ~init:(D.Value.zeros D.Dtype.f32 ~shape:[| 16; 16 |])
          (fun k_start accumulator ->
            let inner = D.Value.add k_start lanes in
            let lhs_offsets =
              D.Value.add
                (D.Value.mul rows (D.Value.int D.Dtype.i32 32))
                (D.Value.expand_dims ~axis:0 inner)
            in
            let rhs_offsets =
              D.Value.add
                (D.Value.mul
                   (D.Value.expand_dims ~axis:1 inner)
                   (D.Value.int D.Dtype.i32 32))
                columns
            in
            let lhs_tile = D.Pointer.load (D.Pointer.offset lhs lhs_offsets) in
            let rhs_tile = D.Pointer.load (D.Pointer.offset rhs rhs_offsets) in
            D.Value.dot lhs_tile rhs_tile accumulator)
      in
      let output_offsets =
        D.Value.add (D.Value.mul rows (D.Value.int D.Dtype.i32 32)) columns
      in
      [
        D.Statement.static_assert
          (D.Spec.input_count spec = 2)
          "expected two inputs";
        D.Statement.store (D.Pointer.offset output output_offsets) accumulator;
      ])

let dsl_square_plus_one =
  D.Kernel.bind dsl_kernel ~fallback:(fun x ->
      Nx.add (Nx.mul x x) (Nx.scalar_like x 1.0))

let dsl_f16_square_plus_one =
  D.Kernel.bind dsl_f16_kernel ~fallback:(fun x ->
      Nx.add (Nx.mul x x) (Nx.scalar_like x 1.0))

let dsl_math =
  D.Kernel.bind dsl_math_kernel ~fallback:(fun x ->
      Nx.add (Nx.erf x) (Nx.log2 (Nx.exp2 x)))

let dsl_extended_math =
  D.Kernel.bind dsl_extended_math_kernel ~fallback:(fun x ->
      let one = Nx.scalar_like x 1. in
      let trigonometry = Nx.add (Nx.sin x) (Nx.cos x) in
      let rounding = Nx.add (Nx.floor x) (Nx.ceil x) in
      let inverse_root = Nx.rsqrt (Nx.add x one) in
      let logarithm = Nx.log (Nx.exp x) in
      let precise = Nx.div (Nx.add (Nx.mul x x) one) (Nx.add (Nx.sqrt x) one) in
      Nx.add trigonometry
        (Nx.add rounding (Nx.add inverse_root (Nx.add logarithm precise))))

let general_abs =
  D.Kernel.bind general_abs_kernel ~fallback:(fun input -> Nx.abs input)

let reduce_rows =
  D.Kernel.bind reduce_sum_kernel ~fallback:(fun input ->
      Nx.sum ~axes:[ 1 ] input)

let dot_16_kernel =
  D.Kernel.bind dot_kernel ~fallback:(fun lhs rhs ->
      Nx.matmul (Nx.cast Nx.float32 lhs) (Nx.cast Nx.float32 rhs))

let dot_16 = function
  | [ lhs; rhs ] -> [ dot_16_kernel lhs rhs ]
  | _ -> failwith "test_cuda_triton: dot expects two inputs"

let softmax_rows =
  D.Kernel.bind softmax_kernel ~fallback:(fun input ->
      Nx.softmax ~axes:[ 1 ] input)

let transpose_16 =
  D.Kernel.bind transpose_kernel ~fallback:(fun input ->
      Nx.transpose ~axes:[ 1; 0 ] input)

let cast_f16_f32 =
  D.Kernel.bind cast_kernel ~fallback:(fun input -> Nx.cast Nx.float32 input)

let integer_add =
  D.Kernel.bind integer_kernel ~fallback:(fun x ->
      Nx.add x (Nx.scalar_like x 7l))

let loop_sum =
  D.Kernel.bind loop_sum_kernel ~fallback:(fun input -> Nx.sum input)

let tiled_gemm_kernel =
  D.Kernel.bind tiled_gemm_kernel ~fallback:(fun lhs rhs ->
      Nx.matmul (Nx.cast Nx.float32 lhs) (Nx.cast Nx.float32 rhs))

let tiled_gemm = function
  | [ lhs; rhs ] -> [ tiled_gemm_kernel lhs rhs ]
  | _ -> failwith "test_cuda_triton: tiled GEMM expects two inputs"

let max_abs_error expected actual =
  let expected = Nx.to_array expected in
  let actual = Nx.to_array actual in
  if Array.length expected <> Array.length actual then
    failwith "test_cuda_triton: output lengths differ";
  let error = ref 0.0 in
  for index = 0 to Array.length expected - 1 do
    error := Float.max !error (Float.abs (expected.(index) -. actual.(index)))
  done;
  !error

let () =
  if Rune_pjrt.backend_available `Cuda then (
    let x = Nx.scalar Nx.float32 41.0 in
    let actual = Rune_pjrt.jit ~backend:`Cuda add_one x |> Nx.item [] in
    if Float.abs (actual -. 42.0) > 1e-6 then
      failwith (Printf.sprintf "test_cuda_triton: expected 42, got %.9g" actual);
    let vector =
      Nx.init Nx.float32 [| 257 |] (fun indices ->
          (float_of_int indices.(0) -. 128.0) /. 31.0)
    in
    let expected = Nx.add (Nx.mul vector vector) (Nx.scalar_like vector 1.0) in
    let compiled = Rune_pjrt.jit ~backend:`Cuda dsl_square_plus_one in
    let error = max_abs_error expected (compiled vector) in
    if error > 1e-6 then
      failwith (Printf.sprintf "test_cuda_triton: DSL max error %.9g" error);
    let f16_vector =
      Nx.init Nx.float16 [| 257 |] (fun indices ->
          (float_of_int indices.(0) -. 128.0) /. 64.0)
    in
    let f16_expected =
      Nx.add (Nx.mul f16_vector f16_vector) (Nx.scalar_like f16_vector 1.0)
    in
    let f16_compiled = Rune_pjrt.jit ~backend:`Cuda dsl_f16_square_plus_one in
    let f16_error = max_abs_error f16_expected (f16_compiled f16_vector) in
    if f16_error > 5e-3 then
      failwith
        (Printf.sprintf "test_cuda_triton: f16 DSL max error %.9g" f16_error);
    let math_vector =
      Nx.init Nx.float32 [| 257 |] (fun indices ->
          (float_of_int indices.(0) -. 128.) /. 256.)
    in
    let math_expected =
      Nx.add (Nx.erf math_vector) (Nx.log2 (Nx.exp2 math_vector))
    in
    let math_error =
      max_abs_error math_expected
        (Rune_pjrt.jit ~backend:`Cuda dsl_math math_vector)
    in
    if math_error > 2e-5 then
      failwith
        (Printf.sprintf "test_cuda_triton: math DSL max error %.9g" math_error);
    let extended_input =
      Nx.init Nx.float32 [| 257 |] (fun indices ->
          0.25 +. (float_of_int indices.(0) /. 512.))
    in
    let extended_expected =
      let one = Nx.scalar_like extended_input 1. in
      let trigonometry =
        Nx.add (Nx.sin extended_input) (Nx.cos extended_input)
      in
      let rounding =
        Nx.add (Nx.floor extended_input) (Nx.ceil extended_input)
      in
      let inverse_root = Nx.rsqrt (Nx.add extended_input one) in
      let logarithm = Nx.log (Nx.exp extended_input) in
      let precise =
        Nx.div
          (Nx.add (Nx.mul extended_input extended_input) one)
          (Nx.add (Nx.sqrt extended_input) one)
      in
      Nx.add trigonometry
        (Nx.add rounding (Nx.add inverse_root (Nx.add logarithm precise)))
    in
    let extended_error =
      max_abs_error extended_expected
        (Rune_pjrt.jit ~backend:`Cuda dsl_extended_math extended_input)
    in
    if extended_error > 3e-5 then
      failwith
        (Printf.sprintf "test_cuda_triton: extended math max error %.9g"
           extended_error);
    let abs_expected = Nx.abs vector in
    let abs_error =
      max_abs_error abs_expected
        (Rune_pjrt.jit ~backend:`Cuda general_abs vector)
    in
    if abs_error > 1e-6 then
      failwith
        (Printf.sprintf "test_cuda_triton: general DSL max error %.9g" abs_error);
    let rows =
      Nx.init Nx.float32 [| 4; 128 |] (fun indices ->
          float_of_int ((indices.(0) * 17) - indices.(1)) /. 23.)
    in
    let reduce_expected = Nx.sum ~axes:[ 1 ] rows in
    let reduce_error =
      max_abs_error reduce_expected
        (Rune_pjrt.jit ~backend:`Cuda reduce_rows rows)
    in
    if reduce_error > 2e-4 then
      failwith
        (Printf.sprintf "test_cuda_triton: reduction max error %.9g"
           reduce_error);
    let lhs =
      Nx.init Nx.float16 [| 16; 16 |] (fun indices ->
          float_of_int ((indices.(0) * 3) - indices.(1)) /. 32.)
    in
    let rhs =
      Nx.init Nx.float16 [| 16; 16 |] (fun indices ->
          float_of_int (indices.(0) + (indices.(1) * 2)) /. 48.)
    in
    let dot_expected =
      Nx.matmul (Nx.cast Nx.float32 lhs) (Nx.cast Nx.float32 rhs)
    in
    let dot_actual =
      match Rune_pjrt.jits ~backend:`Cuda dot_16 [ lhs; rhs ] with
      | [ output ] -> output
      | _ -> failwith "test_cuda_triton: dot returned the wrong output count"
    in
    let dot_error = max_abs_error dot_expected dot_actual in
    if dot_error > 2e-3 then
      failwith (Printf.sprintf "test_cuda_triton: dot max error %.9g" dot_error);
    let softmax_expected = Nx.softmax ~axes:[ 1 ] rows in
    let softmax_error =
      max_abs_error softmax_expected
        (Rune_pjrt.jit ~backend:`Cuda softmax_rows rows)
    in
    if softmax_error > 2e-5 then
      failwith
        (Printf.sprintf "test_cuda_triton: softmax max error %.9g" softmax_error);
    let square =
      Nx.init Nx.float32 [| 16; 16 |] (fun indices ->
          float_of_int ((indices.(0) * 16) + indices.(1)))
    in
    let transpose_error =
      max_abs_error
        (Nx.transpose ~axes:[ 1; 0 ] square)
        (Rune_pjrt.jit ~backend:`Cuda transpose_16 square)
    in
    if transpose_error > 1e-6 then
      failwith
        (Printf.sprintf "test_cuda_triton: transpose max error %.9g"
           transpose_error);
    let cast_error =
      max_abs_error
        (Nx.cast Nx.float32 f16_vector)
        (Rune_pjrt.jit ~backend:`Cuda cast_f16_f32 f16_vector)
    in
    if cast_error > 1e-6 then
      failwith
        (Printf.sprintf "test_cuda_triton: cast max error %.9g" cast_error);
    let integers =
      Nx.init Nx.int32 [| 257 |] (fun indices ->
          Int32.of_int (indices.(0) - 128))
    in
    let integer_expected = Nx.add integers (Nx.scalar_like integers 7l) in
    let integer_actual = Rune_pjrt.jit ~backend:`Cuda integer_add integers in
    if Nx.to_array integer_expected <> Nx.to_array integer_actual then
      failwith "test_cuda_triton: integer kernel returned incorrect values";
    let loop_input =
      Nx.init Nx.float32 [| 128 |] (fun indices ->
          float_of_int (indices.(0) - 64) /. 17.)
    in
    let loop_error =
      max_abs_error (Nx.sum loop_input)
        (Rune_pjrt.jit ~backend:`Cuda loop_sum loop_input)
    in
    if loop_error > 2e-4 then
      failwith
        (Printf.sprintf "test_cuda_triton: device loop max error %.9g"
           loop_error);
    let tiled_lhs =
      Nx.init Nx.float16 [| 32; 32 |] (fun indices ->
          float_of_int ((indices.(0) * 3) - indices.(1)) /. 64.)
    in
    let tiled_rhs =
      Nx.init Nx.float16 [| 32; 32 |] (fun indices ->
          float_of_int (indices.(0) + (indices.(1) * 2)) /. 96.)
    in
    let tiled_expected =
      Nx.matmul (Nx.cast Nx.float32 tiled_lhs) (Nx.cast Nx.float32 tiled_rhs)
    in
    let tiled_actual =
      match
        Rune_pjrt.jits ~backend:`Cuda tiled_gemm [ tiled_lhs; tiled_rhs ]
      with
      | [ output ] -> output
      | _ ->
          failwith
            "test_cuda_triton: tiled GEMM returned the wrong output count"
    in
    let tiled_error = max_abs_error tiled_expected tiled_actual in
    if tiled_error > 4e-3 then
      failwith
        (Printf.sprintf "test_cuda_triton: tiled GEMM max error %.9g"
           tiled_error);
    Printf.printf
      "test_cuda_triton: raw=%.9g dsl_elements=257 f32_max_abs=%.9g \
       f16_max_abs=%.9g math_max_abs=%.9g extended_math_max_abs=%.9g \
       general_max_abs=%.9g reduce_max_abs=%.9g dot_max_abs=%.9g \
       softmax_max_abs=%.9g transpose_max_abs=%.9g cast_max_abs=%.9g \
       loop_max_abs=%.9g tiled_gemm_max_abs=%.9g integer=ok\n\
       %!"
      actual error f16_error math_error extended_error abs_error reduce_error
      dot_error softmax_error transpose_error cast_error loop_error tiled_error)
  else Printf.printf "test_cuda_triton: CUDA plugin unavailable, skipping\n%!"
