(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let failf fmt = Printf.ksprintf failwith fmt
let require msg condition = if not condition then failf "test_ppx: %s" msg

let require_values msg expected actual =
  if Nx.to_array expected <> Nx.to_array actual then
    failf "test_ppx: %s: values differ" msg

let square_both x = Nx.mul x x
[@@rune.kernel.cuda
  {
    library = "/proc/self/exe";
    fwd = "square_both_fwd";
    bwd = "square_both_bwd";
  }]

let square_fwd x = Nx.mul x x
[@@rune.kernel.cuda { library = "/proc/self/exe"; fwd = "square_fwd" }]

let square_bwd x = Nx.mul x x
[@@rune.kernel.cuda { library = "/proc/self/exe"; bwd = "square_bwd" }]

module D = Rune_pjrt.Triton.Dsl

let syntax_kernel =
  D.Kernel.define ~name:"ppx_normal_syntax"
    ~signature:D.Signature.(f32 @-> returning f32)
    ~grid:(fun _ -> (1, 1, 1))
    (fun%rune.kernel spec input output ->
      let offsets =
        (D.Value.program_id D.X * 4) + D.Value.arange ~start:0 ~stop:4
      in
      let mask = offsets >= 0 && offsets < 4 in
      let values =
        D.Pointer.load ~mask
          ~other:(D.Value.zeros D.Dtype.f32 ~shape:[| 4 |])
          (D.Pointer.offset input offsets)
      in
      let result = ((-values * values) + 1.) / 2. in
      [
        D.Statement.static_assert
          [%rune.host D.Spec.input_count spec = 1]
          "expected one input";
        D.Statement.store ~mask (D.Pointer.offset output offsets) result;
      ])

let symbols program =
  Rune_pjrt.Ir.ffi_handlers program
  |> List.map (fun handler -> handler.Rune_pjrt.Ir.symbol)

let require_symbols msg expected program =
  let actual = symbols program in
  if actual <> expected then
    failf "test_ppx: %s: expected [%s], got [%s]" msg
      (String.concat "; " expected)
      (String.concat "; " actual)

let has_op name program =
  List.exists
    (fun node -> Rune_pjrt.Ir.op_name node.Rune_pjrt.Ir.op = name)
    program.Rune_pjrt.Ir.nodes

let trace_gradient fn x =
  Rune_pjrt.Trace.capture_one (Rune.grad (fun x -> Nx.sum (fn x))) x
  |> fun capture -> capture.Rune_pjrt.Trace.program

let contains text pattern =
  let text_length = String.length text in
  let pattern_length = String.length pattern in
  let rec loop offset =
    if offset + pattern_length > text_length then false
    else if String.sub text offset pattern_length = pattern then true
    else loop (offset + 1)
  in
  loop 0

let test_kernel_syntax () =
  let ir =
    D.Kernel.to_ttir_for syntax_kernel ~input_shapes:[ [| 4 |] ]
      ~output_shape:[| 4 |]
  in
  List.iter
    (fun operation ->
      require ("normal syntax emits " ^ operation) (contains ir operation))
    [ "arith.addf"; "arith.mulf"; "arith.divf"; "arith.cmpi"; "arith.andi" ]

let test_eager_and_transforms () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let expected = Nx.mul x x in
  require_values "eager fallback" expected (square_both x);
  let integers = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |] in
  require_values "polymorphic eager fallback" (Nx.mul integers integers)
    (square_both integers);
  let ones = Nx.ones_like x in
  let y, dy = Rune.jvp square_both x ones in
  require_values "JVP primal" expected y;
  require_values "JVP fallback" (Nx.mul_s x 2.) dy;
  let second_derivative =
    Rune.grad
      (fun x -> Nx.sum (Rune.grad (fun x -> Nx.sum (square_both x)) x))
      x
  in
  require_values "higher-order fallback" (Nx.mul_s ones 2.) second_derivative;
  let batched = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  require_values "vmap fallback" (Nx.mul batched batched)
    (Rune.vmap square_both batched);
  let device = Rune.Device.tolk (Tolk_cpu.create "CPU") in
  let scalar = Nx.scalar Nx.float32 3.0 in
  let traced = Rune.trace_graph ~device square_both scalar in
  require "Rune JIT graph uses fallback" (traced.rendered_source <> [])

let test_cpu_fallback () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let capture =
    try Rune_pjrt.Trace.capture_one ~enable_ffi:false square_both x
    with Rune_pjrt.Error.Error error ->
      failf "test_ppx: CPU trace failed: %s" (Rune_pjrt.Error.to_string error)
  in
  require "CPU trace contains fallback" (has_op "mul" capture.program);
  require "CPU trace contains no handler" (symbols capture.program = [])

let test_pjrt_cpu_fallback () =
  if Rune_pjrt.backend_available `Cpu then
    let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let compiled = Rune_pjrt.jit ~backend:`Cpu square_both in
    require_values "PJRT CPU fallback" (Nx.mul x x) (compiled x)

let test_independent_directions () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let trace_forward fn =
    Rune_pjrt.Trace.capture_one fn x |> fun capture -> capture.program
  in
  require_symbols "both forward handler selected" [ "square_both_fwd" ]
    (trace_forward square_both);
  require_symbols "forward-only primal selected" [ "square_fwd" ]
    (trace_forward square_fwd);
  let bwd_primal = trace_forward square_bwd in
  require_symbols "backward-only primal has no handler" [] bwd_primal;
  require "backward-only primal uses Raven operations" (has_op "mul" bwd_primal);
  let both = trace_gradient square_both x in
  require_symbols "both handlers selected"
    [ "square_both_fwd"; "square_both_bwd" ]
    both;
  let fwd = trace_gradient square_fwd x in
  require_symbols "forward-only gradient uses no backward handler" [] fwd;
  require "forward-only backward uses Raven operations" (has_op "mul" fwd);
  let bwd = trace_gradient square_bwd x in
  require_symbols "backward-only handler selected" [ "square_bwd" ] bwd;
  require "backward-only primal uses Raven operations" (has_op "mul" bwd)

let () =
  test_kernel_syntax ();
  test_eager_and_transforms ();
  test_cpu_fallback ();
  test_pjrt_cpu_fallback ();
  test_independent_directions ()
