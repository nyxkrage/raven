# OCaml CUDA kernel DSL

Status: compiler bridge and typed blocked DSL implemented. The worked grouped
GEMM syntax below predates the concrete API and remains a deliberately
provisional design sketch.

This document explores a blocked OCaml kernel DSL at roughly Triton's level of
abstraction. The DSL should make common fused kernels concise while leaving raw
CUDA as the lower-level path for kernels that need exact instruction, shared
memory, warp, or pipeline control.

The intended split is:

- the DSL describes a program instance operating on blocks of values;
- the compiler chooses thread ownership, memory staging, vectorization, and
  tensor-core instructions;
- a small configuration controls block sizes, warp count, pipeline depth, and
  launch order;
- raw CUDA implements kernels for which those compiler-owned choices are not
  good enough;
- the DSL lowers to TTIR and enters the PJRT executable through XLA's internal
  Triton custom call;
- raw CUDA remains on the existing XLA FFI path;
- the ordinary Raven function remains the semantic reference, eager fallback,
  and AD fallback.

This is intentionally not an OCaml version of every CUDA concept. If the public
DSL grows `cp.async` groups, shared-memory swizzles, warp fragment layouts, and
instruction-specific MMA operands, it has crossed the abstraction boundary.
Those kernels should initially stay in raw CUDA.

Avoiding CUDA details in the source does not promise immediate backend
portability. The first compiler and artifact format are CUDA-specific. A later
PJRT accelerator backend may reuse the blocked language where its semantics
genuinely match.

The authoring model follows Triton's
[blocked-program model](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
and
[matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
The complete public `triton.language` inventory and the exact gaps in the
current Raven DSL are tracked in
[`TRITON_LANGUAGE_COVERAGE.md`](TRITON_LANGUAGE_COVERAGE.md).

## Implemented compiler bridge

The first integration deliberately reuses the Triton and LLVM versions bundled
with XLA:

```text
Rune_pjrt.Triton.call
  -> Rune PJRT trace node
  -> stablehlo.custom_call "__gpu$xla.gpu.triton"
       backend_config = { TTIR, grid, warps, stages }
  -> XLA GPU compiler
  -> XLA's bundled Triton lowering
  -> LLVM/NVIDIA code generation
  -> kernel launch owned by the PJRT executable
```

This is compile-time for the PJRT executable, not compilation on the first
kernel launch. XLA owns the CUDA stream, result allocation, kernel arguments,
shared-memory metadata, executable caching, and launch. Raven does not shell
out to Python Triton and does not generate an FFI shared library for a TTIR
kernel.

The current low-level API is:

```ocaml
let add_one_ttir =
  {|
module {
  tt.func public @add_one(
      %input: !tt.ptr<f32, 1>,
      %output: !tt.ptr<f32, 1>) {
    %value = tt.load %input
      {cache = 1 : i32, evict = 1 : i32, isVolatile = false}
      : !tt.ptr<f32>
    %one = arith.constant 1.000000e+00 : f32
    %result = arith.addf %value, %one : f32
    tt.store %output, %result
      {cache = 1 : i32, evict = 1 : i32}
      : !tt.ptr<f32>
    tt.return
  }
}
|}

let add_one_kernel =
  Rune_pjrt.Triton.Kernel.create
    ~name:"add_one"
    ~ir:add_one_ttir
    ~num_warps:1
    ~num_stages:1
    ~grid:(1, 1, 1)
    ()

let add_one input =
  Rune_pjrt.Triton.call add_one_kernel
    ~inputs:[ Rune_pjrt.Triton.Tensor input ]
    ~fallback:(fun () -> Nx.add input (Nx.scalar_like input 1.))
```

The TTIR function ABI is all input pointers in packed order followed by the
single output pointer. The ordinary body is evaluated outside PJRT CUDA and
while Rune transformations derive fallback behavior. Multiple results,
scratch buffers, TMA, target capability guards, and a statement-level kernel
frontend are not implemented yet.

## Implemented typed DSL

`Rune_pjrt.Triton.Dsl` constructs a typed expression and statement IR, reuses
common subexpressions, renders blocked TTIR, specializes code and launch grids
from traced tensor shapes, and calls the compiler bridge above.

`Kernel.define` is the single kernel-authoring entry point. Even a pointwise
kernel states its launch geometry and memory accesses explicitly:

```ocaml
let square_plus_one_definition =
  let module D = Rune_pjrt.Triton.Dsl in
  let block_size = 128 in
  let config = D.Config.make ~block_size ~num_warps:4 () in
  D.Kernel.define ~name:"square_plus_one"
    ~signature:D.Signature.(f32 @-> returning f32)
    ~config
    ~guard:(fun spec ->
      D.Spec.input_numel spec 0 = D.Spec.output_numel spec)
    ~grid:(fun spec ->
      let numel = D.Spec.output_numel spec in
      ((numel + block_size - 1) / block_size, 1, 1))
    (fun%rune.kernel spec input output ->
      let block = D.Value.int D.Dtype.i32 block_size in
      let offsets =
        (D.Value.program_id D.X * block)
        + D.Value.arange ~start:0 ~stop:block_size
      in
      let mask =
        offsets < D.Value.int D.Dtype.i32 (D.Spec.output_numel spec)
      in
      let input =
        D.Pointer.load ~mask
          ~other:(D.Value.zeros D.Dtype.f32 ~shape:[| block_size |])
          (D.Pointer.offset input offsets)
      in
      let result = (input * input) + 1. in
      [
        D.Statement.store ~mask (D.Pointer.offset output offsets) result;
      ])

let square_plus_one_kernel =
  Rune_pjrt.Triton.Dsl.Kernel.bind square_plus_one_definition
    ~fallback:(fun input ->
      Nx.add (Nx.mul input input) (Nx.scalar_like input 1.))

let result = square_plus_one_kernel input
```

The signature declares heterogeneous input and output dtypes. The builder
receives immutable static shape metadata, typed input pointers, and the output
pointer:

```ocaml
let reduce_rows_kernel =
  let module D = Rune_pjrt.Triton.Dsl in
  D.Kernel.define
    ~name:"reduce_rows"
    ~signature:D.Signature.(f32 @-> returning f32)
    ~guard:(fun spec ->
      let input = D.Spec.input_shape spec 0 in
      Array.length input = 2
      && input.(1) = 128
      && input.(0) = D.Spec.output_numel spec)
    ~grid:(fun spec -> (D.Spec.output_numel spec, 1, 1))
    (fun%rune.kernel spec input output ->
      let row = D.Value.program_id D.X in
      let offsets =
        (row * 128) + D.Value.arange ~start:0 ~stop:128
      in
      let values = D.Pointer.load (D.Pointer.offset input offsets) in
      [
        D.Statement.static_assert
          [%rune.host D.Spec.input_count spec = 1]
          "expected one input";
        D.Statement.store
          (D.Pointer.offset output row)
          (D.Value.sum ~axis:0 values);
      ])
```

The implemented dtypes are f16, bf16, f32, i1, i32, and i64. Values may be
scalars or statically shaped blocks. The public operations cover typed
constants, ranges, program coordinates, broadcasting, reshape, permutation,
casts, comparisons, selection, integer and floating-point arithmetic, math
functions, reductions, softmax, and f16/bf16/f32 block dot with f32
accumulation.

`Signature` maps ordinary tensor arguments to equally structured typed pointer
arguments. Its right-associative `@->` chain makes both builders and bound
kernels curried, so a two-input definition receives `lhs` and `rhs` separately
and is called as `kernel lhs rhs`. `Kernel.bind` attaches the one semantic
fallback at the definition site and returns an ordinary function, packing
`Nx.t` arguments without a public existential wrapper. `Syntax` provides
locally scoped value arithmetic, i32-scalar arithmetic, and pointer offset
operators.

`ppx_rune_kernel` also provides normal syntax throughout an entire kernel
builder:

```lisp
(preprocess
 (pps ppx_rune_kernel))
```

```ocaml
(fun%rune.kernel spec input output ->
  let offsets =
    (D.Value.program_id D.X * 128)
    + D.Value.arange ~start:0 ~stop:128
  in
  let mask = offsets >= 0 && offsets < 128 in
  let values =
    D.Pointer.load ~mask (D.Pointer.offset input offsets)
  in
  let result = (-values * values + 1.) / 2. in
  [
    D.Statement.static_assert
      [%rune.host D.Spec.input_count spec = 1]
      "expected one input";
    D.Statement.store
      ~mask (D.Pointer.offset output offsets) result;
  ])
```

Inside `fun%rune.kernel`, `+`, `-`, `*`, `/`, `mod`, comparisons, Boolean
operators, and their floating-point spellings build DSL values. Integer,
floating-point, and Boolean literals adopt the dtype of a neighboring staged
value, so the same `1.` source works with f16, bf16, or f32 without promoting
the expression. Literal-only operations have no dtype anchor and are rejected.
For a host value such as a shape-derived integer, construct a staged constant
with `Value.int` before using normal syntax.

The function is the syntax boundary, so one annotation covers every staged
expression in the builder. Ordinary OCaml arithmetic in `grid` and `guard`
remains outside it. A rare host-side expression inside the builder uses
`[%rune.host ...]`, as in the specialization assertion above. Pointer
arithmetic stays explicit through `Pointer.offset`, because a PPX runs before
type checking and cannot safely decide whether an arbitrary operand is a value
or a pointer.

A two-input signature therefore has no tuple or trailing unit argument:

```ocaml
let gemm_definition =
  D.Kernel.define
    ~name:"gemm"
    ~signature:D.Signature.(f16 @-> f16 @-> returning f32)
    ~grid
    (fun spec lhs rhs output ->
      build_gemm spec lhs rhs output)

let gemm_kernel =
  D.Kernel.bind gemm_definition
    ~fallback:(fun lhs rhs ->
      Nx.matmul
        (Nx.cast Nx.float32 lhs)
        (Nx.cast Nx.float32 rhs))

let result = gemm_kernel lhs rhs
```

Pointers support typed element offsets and masked loads. Statements support
masked stores and specialization-time assertions. `Value.range` lowers a typed
device loop with one loop-carried value; `Dsl.static_range` unrolls a
construction-time loop.

The pointwise example processes `block_size` contiguous flattened elements per
program. Its generated TTIR forms
`program_id * block_size + [0, block_size)`, masks loads and stores against the
static element count, and launches `ceil_div(numel, block_size)` programs.
Empty tensors and failed specialization guards use the Raven fallback;
incompatible tensor dtypes are rejected by the OCaml type checker.

All DSL kernels share one TTIR emitter and specialization cache. CUDA tests
cover masked tails, general pointer programs, comparisons and selection,
extended math, integer buffers, mixed f16/f32 casts, block reductions, softmax,
transpose, scalar and block loop-carried values, and tensor-core dot. A tiled
32x32 GEMM test combines two-dimensional pointer blocks, a K-loop, f16 loads,
and f32 accumulation.

Multiple results, atomics, scans and sorting, random numbers, FP8 scaled dot,
block pointers, tensor descriptors/TMA, cache modifiers, device diagnostics,
target capability guards, configuration families/autotuning, and a PPX for
statement-level control flow remain unimplemented. Raw CUDA remains the
lower-level path for kernels requiring those target-specific facilities.

## Design position

The smallest useful design has one high-level DSL and one whole-kernel escape
hatch:

```text
ordinary Raven function
  |
  +-- anything outside the PJRT CUDA custom-kernel path
  |     -> execute or transform the Raven body
  |
  `-- PJRT CUDA
        -> select a kernel-family variant
             |
             +-- blocked OCaml DSL variant
             |     -> TTIR in the XLA/PJRT executable compilation
             |
             `-- raw CUDA variant
                   -> AOT-compile with NVCC in the same pipeline
```

The generated kernels are part of PJRT executables. Rune/Tolk JIT, eager
execution, other backends, and transformations such as `vmap` never execute
them; those contexts see only the ordinary Raven body.

The raw CUDA variant is not an error fallback. Selection happens from declared
capabilities and problem guards before launch. DSL selection happens while
lowering a statically described PJRT call, and XLA compiles the selected TTIR
with the containing executable. A raw-CUDA handler may instead dispatch over
host-visible buffer descriptors. Compilation, registration, or launch failure
remains an error.

The abstraction boundary is whole-kernel at first. Calling arbitrary CUDA
device functions or embedding PTX inside a DSL kernel can be considered later,
but neither is required to make the first design useful.

## Concrete target: packed grouped GEMM

The current grouped GEMM has this contract:

```text
lhs         : T[rows, k]
rhs         : T[groups, k, n]
group_sizes : i32[groups]
output      : T[rows, n]

offset[g] = exclusive_sum(group_sizes)[g]

output[offset[g] : offset[g] + group_sizes[g], :]
  = lhs[offset[g] : offset[g] + group_sizes[g], :] @ rhs[g, :, :]
```

`T` is `f16`, `bf16`, or `f32`, and accumulation is `f32`. Group sizes may be
zero, must be non-negative, and sum to `rows`.

Rows are already packed in expert-major order. Routing remains a separate
histogram, prefix-sum, and scatter operation:

```text
expert IDs
  -> histogram
  -> exclusive prefix sum
  -> scatter tokens into expert-major order
  -> grouped GEMM
  -> inverse scatter
```

Sorting and routing should not be hidden inside the GEMM kernel. The routing
permutation can then be reused by both sparse feed-forward projections.

## Algorithm before API

The blocked algorithm, independent of any proposed syntax, is:

```text
kernel grouped_gemm<T, BLOCK_M, BLOCK_N, BLOCK_K>(
    lhs, rhs, group_sizes, output):

  ragged_row_tile = program_id(x)
  column_tile     = program_id(y)

  owner = locate_segmented_tile(
    sizes = group_sizes,
    tile_size = BLOCK_M,
    tile_id = ragged_row_tile)

  if owner is invalid:
    return

  row_offsets = [0 .. BLOCK_M)
  row_indices = owner.group_row_start + owner.local_row_start + row_offsets
  cols = column_tile * BLOCK_N + [0 .. BLOCK_N)

  accumulator = zeros([BLOCK_M, BLOCK_N], f32)

  for inner_start in 0 .. k step BLOCK_K:
    inner = inner_start + [0 .. BLOCK_K)

    a = masked_load(
      lhs[row_indices[:, none], inner[none, :]],
      mask = row_offsets < owner.valid_rows
             and row_indices < rows
             and inner < k,
      other = 0)

    b = masked_load(
      rhs[owner.group, inner[:, none], cols[none, :]],
      mask = inner < k and cols < n,
      other = 0)

    accumulator = dot(a, b, accumulator)

  masked_store(
    output[row_indices[:, none], cols[none, :]],
    cast<T>(accumulator),
    mask = row_offsets < owner.valid_rows
           and row_indices < rows
           and cols < n)
```

For non-empty problems, the launch grid uses a safe upper bound for the number
of non-empty row tiles:

```text
row_programs    = ceil_div(rows, BLOCK_M) + groups - 1
column_programs = ceil_div(n, BLOCK_N)
```

The two dimensions may be assigned to `grid.x` and `grid.y` in either order.
`groups`, `k`, and `n` must be positive. If `rows = 0`, the host adapter emits
an empty grid and does not launch the kernel. Some row programs may otherwise
be inactive.
`locate_segmented_tile` maps the compact ragged tile ordinal back to a group,
the group's global row start, and the tile's local row offset. `valid_rows` is
the smaller of `BLOCK_M` and the number of rows remaining in that group.

This form intentionally says nothing about:

- threads or warps;
- shared memory;
- `cp.async`;
- WMMA fragment ownership;
- shared-memory skew or swizzle;
- barriers;
- vector width;
- register allocation.

Those are compiler decisions in a DSL variant and author decisions in a raw
CUDA variant.

## Staging model

A kernel definition runs as ordinary OCaml while constructing device IR.
Configuration values are ordinary OCaml values:

```ocaml
type config = private {
  block_m : int;
  block_n : int;
  block_k : int;
  num_warps : int;
  num_stages : int;
  grid_order : [ `Rows_first | `Columns_first ];
}

Config.make
  ~block_m
  ~block_n
  ~block_k
  ~num_warps
  ~num_stages
  ~grid_order
```

They select variants, construct static block shapes, and unroll generation-time
code. They are not device values. `Config.make` rejects non-positive block
dimensions, warp counts, and stage counts. Target-specific limits and resource
use are checked before a specialization enters the artifact manifest.

Tensor dimensions, program IDs, loaded scalars, masks, and blocks are staged
device values. Ordinary OCaml cannot branch on them. Device control flow uses
DSL combinators:

```ocaml
Control.when_ predicate (fun () -> ...)

Control.if_ predicate
  ~then_:(fun () -> ...)
  ~else_:(fun () -> ...)

Control.fold
  ~from:start
  ~until:stop
  ~step
  ~init
  (fun index state -> ...)

Control.while_
  ~init
  ~cond:(fun state -> ...)
  ~body:(fun state -> ...)
```

This keeps the core library a normal deep embedding. The implemented
`fun%rune.kernel` PPX translates operators throughout one kernel builder; a
later statement-level frontend could translate control flow into these
combinators, but correctness must not depend on a large source-to-source
transformation.

## Core values

The useful public distinctions are small:

```ocaml
type f16
type bf16
type f32
type i32
type index

type uniform
type varying
type scalar_predicate
type block_predicate

type ('a, 'uniformity) Scalar.t
type 'a Block.t
type ('a, 'rank, 'access) Tensor.t
type ('shape, 'uniformity) Predicate.t
type Dim.t
```

`Block.t` is an immutable, compile-time-shaped block of staged values. Scalar
and block operators broadcast in the usual way.

`index` is a defined signed 64-bit device integer used by dimensions, program
IDs, coordinates, and prefix sums. `Dim.t` is a program-uniform `index` scalar.
The adapter rejects descriptors whose allocation size or indexing arithmetic
cannot fit that domain; the compiler may narrow an index only after proving the
range. Storage dtypes remain explicit: `group_sizes` is externally `i32`, so
its values are widened to `index` immediately after each load.

Comparisons produce a `Predicate.t`, not an OCaml `bool`. Scalar comparisons
produce `scalar_predicate`; block comparisons produce `block_predicate`.
`Mask.and_`, `Mask.or_`, and `Mask.not_` operate on predicates and broadcast a
uniform scalar predicate over a block when needed. `Control.when_`,
`Control.if_`, and `Control.while_` require a program-uniform scalar predicate;
masked tensor operations require a broadcast-compatible block predicate.

The type system should track:

- element dtype;
- tensor read/write capability;
- scalar uniformity where it protects collective control flow;
- result arity and argument order in the kernel signature.

It should not try to encode every dimension and block shape in phantom types.
Shapes remain values verified while constructing or compiling the kernel.
Fully type-level shapes would dominate the OCaml API while still not proving
dynamic buffer bounds.

## Proposed modules

### `Signature`

Describes the external handler ABI:

```ocaml
let element = Dtype.var "t"
let rows_dim = Shape.var "rows"
let inner_dim = Shape.var "inner"
let groups_dim = Shape.var "groups"
let columns_dim = Shape.var "columns"

let lhs_arg =
  Signature.tensor
    ~name:"lhs"
    ~dtype:element
    ~rank:Rank.two
    ~access:`Read

let rhs_arg =
  Signature.tensor
    ~name:"rhs"
    ~dtype:(Dtype.same_as lhs_arg)
    ~rank:Rank.three
    ~access:`Read

let group_sizes_arg =
  Signature.tensor
    ~name:"group_sizes"
    ~dtype:Dtype.i32
    ~rank:Rank.one
    ~access:`Read

let output_result =
  Signature.result
    ~name:"output"
    ~dtype:(Dtype.same_as lhs_arg)
    ~rank:Rank.two

let grouped_gemm_signature =
  let open Signature.Constraint in
  Signature.make
    ~arguments:
      [ Signature.pack lhs_arg;
        Signature.pack rhs_arg;
        Signature.pack group_sizes_arg ]
    ~results:[ Signature.pack output_result ]
    ~constraints:
      [
        dtype element (one_of [ f16; bf16; f32 ]);
        shape lhs_arg [ rows_dim; inner_dim ];
        shape rhs_arg [ groups_dim; inner_dim; columns_dim ];
        shape group_sizes_arg [ groups_dim ];
        shape output_result [ rows_dim; columns_dim ];
        positive groups_dim;
        positive inner_dim;
        positive columns_dim;
      ]
```

The signature is shared by DSL and raw CUDA variants. A generated XLA FFI
adapter validates it and supplies canonical contiguous buffers, result
destinations, and XLA's CUDA stream.

Each tensor/result declaration returns a typed handle. `Signature.pack` erases
only enough type information to put handles in ABI order; `Kernel.get bindings
handle` recovers the tensor with the handle's dtype, rank, and access type.
This makes the core callback implementable as a normal GADT-based embedding.
A generated module or later syntax PPX may add labeled arguments as convenience,
but the core API does not derive OCaml labels from a runtime signature value.

This is an artifact ABI contract, not a second public shape function. The
ordinary Raven body remains the source of concrete output shapes and dtypes
during tracing; binding a kernel family cross-checks those inferred results
against the signature.

The manifest also records the trusted value contract that `group_sizes` is
non-negative and sums to `rows`, separately from host-checkable shape
constraints. Every implementation must meet the stronger defensive
memory-safety rule described under `Segment`, even when that semantic value
contract is not validated.

### `Kernel`

Defines a blocked kernel:

```ocaml
Kernel.define
  ~name:"grouped_gemm"
  ~signature
  ~grid
  ~num_warps:config.num_warps
  ~num_stages:config.num_stages
  (fun program bindings ->
    let lhs = Kernel.get bindings lhs_arg in
    let output = Kernel.get bindings output_result in
    ...)
```

The body stores every result and returns `unit`. XLA owns result allocation.
Static configuration captured by the builder becomes part of the specialization
and artifact identity.

`Kernel.dim argument axis` reads a symbolic buffer extent. `Kernel.dtype`
reads a statically specialized dtype.

### `Grid` and `Program`

The host-side grid function sees problem descriptors and static configuration:

```ocaml
let grouped_gemm_grid problem config =
  let rows = Problem.dim problem rows_dim in
  let groups = Problem.dim problem groups_dim in
  let columns = Problem.dim problem columns_dim in
  if rows = 0 then Grid.empty
  else
    let row_programs =
      ceil_div rows config.block_m + groups - 1
    in
    let column_programs =
      ceil_div columns config.block_n
    in
    match config.grid_order with
    | `Rows_first ->
        Grid.xy ~x:row_programs ~y:column_programs
    | `Columns_first ->
        Grid.xy ~x:column_programs ~y:row_programs
```

`Problem.dim` reads a host-visible dimension bound to a `Shape.var` in the
signature. Grid construction runs only after signature constraints have been
validated, so `groups > 0` here. A problem descriptor also exposes concrete
dtype, target, layout, and alignment facts to variant guards; it never exposes
device buffer contents.

Inside a program:

```ocaml
let row_tile, column_tile =
  match config.grid_order with
  | `Rows_first ->
      Program.id program `X, Program.id program `Y
  | `Columns_first ->
      Program.id program `Y, Program.id program `X
```

`Program.id` is uniform within the program instance.

### `Scalar` and operators

Constructs staged scalar constants and arithmetic:

```ocaml
Scalar.i32 0l
Scalar.index 0
Scalar.f32 0.
Scalar.bool false
Scalar.zero dtype
Scalar.min left right
Scalar.max left right
Scalar.ceil_div numerator denominator

let open Kernel.O in
let tile_start = row_tile * Scalar.index config.block_m in
let in_bounds = tile_start < rows in
```

`Scalar.bool` constructs a uniform scalar `Predicate.t`; the numeric
constructors produce `Scalar.t`.

The same local operators work on compatible scalars and blocks, with explicit
broadcasting where ranks differ. They build IR; they do not invoke OCaml
integer or floating-point arithmetic. `Scalar.cast Dtype.index value` widens a
stored integer into the coordinate type. Conversions that may narrow or change
floating-point behavior use `Scalar.cast` or `Block.cast`. Mask composition is
explicit through `Mask.and_`, `Mask.or_`, and `Mask.not_`; it never uses
OCaml's short-circuit booleans.

### `Block`

Creates and computes on blocked values:

```ocaml
Block.iota config.block_m
Block.zeros Dtype.f32 [ config.block_m; config.block_n ]
Block.broadcast ~axis:0 rows
Block.broadcast ~axis:1 columns
Block.as_row values
Block.as_column values
Block.scalar scalar
Block.add_scalar values scalar
Block.where mask then_ else_
Block.cast dtype values
Block.silu values
Block.dot ~accumulator a b
Block.sum ~axis values
Block.max ~axis values
```

`Block.iota length` produces a one-dimensional `index` block. Static `length`
fixes its block shape; only the resulting elements are staged values.

`Block.dot` is semantic. The compiler chooses a SIMT or tensor-core lowering,
thread decomposition, shared staging, and software pipeline. `num_warps` and
`num_stages` constrain that choice but do not expose its mechanics.

Elementwise block expressions around a dot remain in the same program, so a
fused epilogue is ordinary code:

```ocaml
let open Kernel.O in
let projected = Block.dot ~accumulator a weights in
let gated = Block.silu gate * projected in
Tensor.store output ~coords ~mask (Block.cast output_dtype gated)
```

### `Tensor`

Loads and stores blocks by logical coordinates:

```ocaml
Tensor.load tensor
  ~coords:[ row_coords; inner_coords ]
  ~mask
  ~other:(Scalar.zero (Kernel.dtype tensor))

Tensor.store tensor
  ~coords:[ row_coords; column_coords ]
  ~mask
  values

Tensor.load_scalar tensor
  ~coords:[ group ]
```

Coordinates are broadcast-compatible blocks. Loads outside the mask do not
dereference their address and produce `other`. Stores outside the mask do
nothing. CUDA lowering must predicate address formation itself, or keep offsets
as integers until the predicated instruction, so an inactive lane never forms
an out-of-bounds C++ pointer. A scalar load with uniform coordinates produces a
uniform scalar.

The first version can require canonical contiguous external buffers because
the current PJRT FFI path already enforces that layout. Strided external
arguments can be added only when a concrete kernel needs them.

### `Control`

Represents device-side structured control flow. `Control.fold` must accept a
dynamic upper bound and a static step. Its loop-carried state may be a scalar,
block, tuple, or record whose structure and component types are fixed while
building the kernel. Static OCaml loops remain available for generation-time
repetition.

The verifier rejects a program-wide collective operation under varying
control flow. Most elementwise conditions should use masks or `Block.where`,
not divergent branches.

### `Segment`

Ragged work mapping is common enough to be a library helper but does not need
to be a compiler primitive:

```ocaml
type owner = {
  valid : (scalar_predicate, uniform) Predicate.t;
  group : (index, uniform) Scalar.t;
  group_row_start : (index, uniform) Scalar.t;
  local_row_start : (index, uniform) Scalar.t;
  valid_rows : (index, uniform) Scalar.t;
}

Segment.owner_of_tile
  ~sizes:group_sizes
  ~rows
  ~tile_size:config.block_m
  ~tile_id:row_tile
```

`valid_rows` is the non-negative minimum of `tile_size` and
`group_size - local_row_start`. It is zero when the tile has no owner.

The helper can itself be ordinary DSL code. A simple first lowering is:

```ocaml
let owner_of_tile ~sizes ~rows ~tile_size ~tile_id =
  let open Kernel.O in
  let groups = Kernel.dim sizes 0 in
  let zero = Scalar.index 0 in
  let one = Scalar.index 1 in
  let tile_size = Scalar.index tile_size in
  let initial =
    {
      group = zero;
      row_base = zero;
      tile_base = zero;
      group_size = zero;
      found = Scalar.bool false;
    }
  in
  let state =
    Control.while_
      ~init:initial
      ~cond:(fun state ->
        Mask.and_ (Mask.not_ state.found) (state.group < groups))
      ~body:(fun state ->
        let encoded_size =
          Tensor.load_scalar sizes ~coords:[ state.group ]
        in
        let remaining =
          Scalar.max zero (rows - state.row_base)
        in
        let size =
          Scalar.min remaining
            (Scalar.max zero
               (Scalar.cast Dtype.index encoded_size))
        in
        let tiles = Scalar.ceil_div size tile_size in
        let found =
          Mask.and_
            (tile_id >= state.tile_base)
            (tile_id < state.tile_base + tiles)
        in
        Control.if_ found
          ~then_:(fun () ->
            {
              state with
              group_size = size;
              found = Scalar.bool true;
            })
          ~else_:(fun () ->
            {
              group = state.group + one;
              row_base = state.row_base + size;
              tile_base = state.tile_base + tiles;
              group_size = zero;
              found = Scalar.bool false;
            }))
  in
  let local_row_start =
    (tile_id - state.tile_base) * tile_size
  in
  let valid_rows =
    Control.if_ state.found
      ~then_:(fun () ->
        Scalar.min tile_size
          (Scalar.max zero
             (state.group_size - local_row_start)))
      ~else_:(fun () -> zero)
  in
  {
    valid = state.found;
    group = state.group;
    group_row_start = state.row_base;
    local_row_start;
    valid_rows;
  }
```

All state is program-uniform, so this emits one scalar search per program
instance rather than one search per output element.

The clamp makes malformed metadata unable to produce a row address outside
`[0, rows)`, and the caller still masks final row coordinates against `rows`.
It does not make malformed metadata semantically valid. Non-negative sizes
whose sum is exactly `rows` remain a contract of the router or caller.
Because `group_sizes` is device-resident and may change on every execution, a
signature check or validation during an earlier trace cannot establish that
value invariant without a synchronization.

That distinction applies to the entire kernel family. Every DSL and raw CUDA
variant must defensively clamp segmented address calculation so arbitrary
`i32` contents cannot cause out-of-bounds memory access. Correct numerical
semantics still require the value invariant. A high-level binding may either
carry provenance from a checked router or request an explicit validation;
selection must never insert a surprise device-to-host synchronization. Without
either, the invariant is a trusted precondition and malformed metadata has an
unspecified result, but remains memory-safe.

The initial implementation may emit a uniform scalar loop over groups. The
helper can later improve without changing kernel source, for example by using
an offsets argument or compiler-supported subgroup scan.

If the exact warp scan, ballot, and broadcast sequence is performance-critical
and the DSL cannot produce it, that is a reason to select a raw CUDA variant,
not to immediately expose warp intrinsics throughout the high-level API.

### `Hint`

A very small hint API is reasonable:

```ocaml
Hint.multiple_of inner 8
Hint.multiple_of columns 8
Hint.aligned lhs 16
Hint.max_contiguous inner_coords config.block_k
```

Hints may guide vectorization and tensor-core selection, but cannot change
semantics. Evidence may come from a checked variant guard, such as alignment or
divisibility, or from compiler analysis of the coordinate IR, such as
contiguity. A hint that neither source proves is rejected.

## Grouped GEMM in the proposed API

The following is intentionally pseudocode. Index broadcasting notation is
illustrative.

```ocaml
let grouped_gemm config =
  Kernel.define
    ~name:"grouped_gemm"
    ~signature:grouped_gemm_signature
    ~grid:(fun problem -> grouped_gemm_grid problem config)
    ~num_warps:config.num_warps
    ~num_stages:config.num_stages
    (fun program bindings ->
      let open Kernel.O in
      let lhs = Kernel.get bindings lhs_arg in
      let rhs = Kernel.get bindings rhs_arg in
      let group_sizes = Kernel.get bindings group_sizes_arg in
      let output = Kernel.get bindings output_result in
      let rows = Kernel.dim lhs 0 in
      let inner = Kernel.dim lhs 1 in
      let columns = Kernel.dim rhs 2 in
      let dtype = Kernel.dtype lhs in
      let zero = Scalar.zero dtype in

      let row_tile, column_tile =
        match config.grid_order with
        | `Rows_first ->
            Program.id program `X, Program.id program `Y
        | `Columns_first ->
            Program.id program `Y, Program.id program `X
      in

      let owner =
        Segment.owner_of_tile
          ~sizes:group_sizes
          ~rows
          ~tile_size:config.block_m
          ~tile_id:row_tile
      in

      Control.when_ owner.valid (fun () ->
        let row_offsets = Block.iota config.block_m in
        let column_offsets = Block.iota config.block_n in

        let row_indices =
          Block.add_scalar
            row_offsets
            (owner.group_row_start + owner.local_row_start)
        in
        let column_indices =
          Block.add_scalar
            column_offsets
            (column_tile * Scalar.index config.block_n)
        in

        let accumulator =
          Control.fold
            ~from:(Scalar.index 0)
            ~until:inner
            ~step:config.block_k
            ~init:
              (Block.zeros Dtype.f32
                 [ config.block_m; config.block_n ])
            (fun inner_start accumulator ->
              let inner_offsets = Block.iota config.block_k in
              let inner_indices =
                Block.add_scalar inner_offsets inner_start
              in

              let a_rows = Block.as_column row_indices in
              let a_inner = Block.as_row inner_indices in
              let a_mask =
                Mask.and_
                  (Block.as_column row_offsets < owner.valid_rows)
                  (Mask.and_
                     (a_rows < rows)
                     (a_inner < inner))
              in
              let a =
                Tensor.load lhs
                  ~coords:[ a_rows; a_inner ]
                  ~mask:a_mask
                  ~other:zero
              in

              let b_inner = Block.as_column inner_indices in
              let b_columns = Block.as_row column_indices in
              let b_mask =
                Mask.and_
                  (b_inner < inner)
                  (b_columns < columns)
              in
              let b =
                Tensor.load rhs
                  ~coords:
                    [ Block.scalar owner.group;
                      b_inner;
                      b_columns ]
                  ~mask:b_mask
                  ~other:zero
              in

              Block.dot ~accumulator a b)
        in

        let output_rows = Block.as_column row_indices in
        let output_columns = Block.as_row column_indices in
        let output_mask =
          Mask.and_
            (Block.as_column row_offsets < owner.valid_rows)
            (Mask.and_
               (output_rows < rows)
               (output_columns < columns))
        in
        Tensor.store output
          ~coords:[ output_rows; output_columns ]
          ~mask:output_mask
          (Block.cast dtype accumulator)))
```

This is the target level of detail: the author controls the blocked algorithm,
not its implementation in threads.

Although the signature contains a dtype variable, each generated artifact is
specialized to one concrete dtype. `Kernel.dtype lhs` therefore constructs a
static dtype witness; it is not a device-side dtype branch.

## Line-by-line API mapping

| Algorithm part | Proposed API | Meaning |
| --- | --- | --- |
| Declare tensors and results | `Signature.tensor`, `Signature.result` | One typed ABI shared by every implementation variant. |
| Bind declared tensors in the body | `Kernel.get bindings handle` | Recover a tensor with the handle's static dtype, rank, and access type. |
| Name `rows`, `k`, `n`, and `groups` | `Shape.var`, signature constraints | Relate dimensions and validate shapes before launch. |
| Choose block sizes | ordinary OCaml `config` | Generate statically specialized variants. |
| Compute launch dimensions | `Problem.dim`, `Grid.xy`, `ceil_div` | Host-side launch geometry from bound shapes and config. |
| Read program coordinates | `Program.id` | Select one blocked program instance. |
| Map ragged tile to expert | `Segment.owner_of_tile` | Produce uniform group and row metadata. |
| Skip an overlaunched tile | `Control.when_ owner.valid` | Uniform device branch. |
| Form row, column, and K ranges | `Block.iota`, `Block.add_scalar` | Construct logical coordinate blocks. |
| Add singleton axes | `Block.as_row`, `Block.as_column` | Make broadcasting explicit. |
| Express edge conditions | block comparisons, `Mask.and_` | Produce masks without divergent elementwise control flow. |
| Read A and B tiles | `Tensor.load ~coords ~mask ~other` | Masked global load with zero fill. |
| Create f32 accumulator | `Block.zeros Dtype.f32` | Fix accumulation precision independently of input dtype. |
| Iterate over K tiles | `Control.fold` | Dynamic device loop carrying the accumulator. |
| Multiply-accumulate tiles | `Block.dot ~accumulator` | Semantic block GEMM; compiler selects implementation. |
| Convert output dtype | `Block.cast` | Explicit mixed-precision epilogue. |
| Write the output tile | `Tensor.store ~coords ~mask` | Destination-passed, masked result store. |
| State alignment facts | `Hint.aligned`, `Hint.multiple_of` | Verified assumptions used during lowering. |
| Select schedules | `Kernel.variant` or tuning configs | Choose block sizes, warps, stages, and grid order. |

## What the compiler owns

For a DSL `Block.dot`, the compiler is responsible for:

- mapping block elements to threads and warps;
- coalescing global loads and stores;
- choosing vector widths;
- allocating and laying out shared memory;
- selecting SIMT, WMMA, or another supported dot implementation;
- inserting synchronization;
- generating asynchronous staging when profitable;
- honoring `num_warps` and `num_stages`;
- checking shared memory and register limits;
- predicating boundary tiles correctly.

The compiler may report that it cannot lower a requested configuration. It
must not silently reinterpret the algorithm or drop a verified guard.

The first implementation need not perform heroic automatic scheduling. A
small set of known dot schedules selected from dtype, target, block shape,
warps, and stages is enough to validate the API. Better scheduling can arrive
without changing kernel source.

## When raw CUDA is the right API

The existing grouped GEMM tensor-core path explicitly controls:

- 32- or 64-row by 64-column CTA tiles;
- four threads per tile row;
- warp ownership of two `16 x 16` WMMA fragments;
- two shared-memory stages;
- 16-byte predicated `cp.async` copies;
- an eight-element shared-memory skew;
- `cp.async` commit and wait groups;
- exact CTA barriers;
- reuse of input shared storage as f32 epilogue scratch;
- paired `half2` or `bfloat162` output conversion and stores;
- a warp prefix scan and ballot for many-group tile lookup;
- grid-axis orientation selected by a hand-written heuristic.

That is a good raw CUDA kernel. The high-level DSL should not need equivalent
public calls for every item in this list.

An initial kernel family could deliberately use:

```text
f16/bf16, K % 8 = 0, N % 8 = 0, sm80+
  -> current hand-tuned raw CUDA tensor-core variant

everything else
  -> blocked DSL variant

no eligible CUDA variant
  -> selection error before launch
```

The ordinary Raven body remains the fallback in contexts where no CUDA handler
is selected. A broken selected CUDA variant is still an error.

## One family, multiple implementations

A kernel family binds a semantic signature to guarded implementations:

```ocaml
let grouped_gemm_cuda =
  Kernel_family.define
    ~name:"grouped_gemm"
    ~signature:grouped_gemm_signature
    [
      Raw_cuda.variant
        ~name:"sm80_tensor_core"
        ~source:"grouped_gemm_sm80.cu"
        ~entries:
          (Raw_cuda.by_dtype
             [ Dtype.f16, "GroupedGemmSm80F16";
               Dtype.bf16, "GroupedGemmSm80Bf16" ])
        ~priority:100
        ~requires:[ Cuda.sm 80 ]
        ~when_:
          (Guard.all
             [ Guard.dtype lhs_arg (Guard.one_of [ f16; bf16 ]);
               Guard.multiple_of (Guard.dim lhs_arg 1) 8;
               Guard.multiple_of (Guard.dim rhs_arg 2) 8 ]);

      Kernel.variant
        ~name:"blocked"
        ~specializations:grouped_gemm_specializations
        ~build:grouped_gemm;
    ]
```

The specialization value is explicit dispatch data, typically generated by the
AOT tuning rule:

```ocaml
let grouped_gemm_specializations =
  [
    Kernel.specialize
      ~config:tuned_config
      ~when_:
        (Guard.all
           [ Guard.multiple_of (Guard.dim lhs_arg 1) 32;
             Guard.multiple_of (Guard.dim rhs_arg 2) 64 ])
      ~priority:20;
    Kernel.specialize
      ~config:generic_config
      ~when_:Guard.true_
      ~priority:0;
  ]
```

The tuner may emit several guarded problem buckets. The final generic case is
deliberate rather than implicit. Builder and target checks remove an illegal
configuration before manifest generation; if that would remove the only
generic case for a claimed dtype/target, the build fails.

The exact syntax is open, but these properties are important:

- guards are declarative and checked against the signature;
- every specialization declares a static config, eligibility guard, and
  deterministic priority;
- every selected variant has the same arguments and results;
- raw CUDA and DSL artifacts appear in one manifest;
- the selected variant, or the embedded dispatch-table identity, enters the
  executable and artifact cache keys;
- selection never catches a launch error and tries another implementation.

Whole-kernel variants keep the boundary understandable. There is no hidden
cost or semantic question about moving values between generated DSL code and
an externally compiled CUDA function halfway through a kernel.

## Raw CUDA author experience

Raw CUDA should remain genuinely raw, but Raven can generate the repetitive XLA
FFI adapter from the shared signature. The CUDA author should ideally implement
a launch function resembling:

```cpp
template <typename T>
raven::Status GroupedGemmSm80(
    cudaStream_t stream,
    raven::TensorView<const T, 2> lhs,
    raven::TensorView<const T, 3> rhs,
    raven::TensorView<const int32_t, 1> group_sizes,
    raven::TensorView<T, 2> output) {
  // Arbitrary CUDA, inline PTX, dynamic shared memory, and additional
  // stream-ordered launches are allowed here.
  GroupedGemmKernel<<<grid, block, shared_bytes, stream>>>(...);
  return raven::cuda_status(cudaPeekAtLastError());
}
```

The templated function is an authoring helper, not an ABI entry point.
`GroupedGemmSm80F16` and `GroupedGemmSm80Bf16` are thin concrete wrappers
around its `half` and `nv_bfloat16` instantiations. The declared `entries`
mapping tells the generated adapter exactly which launch function belongs to
each concrete signature dtype; a raw implementation never receives an
unresolved runtime dtype variable.

The generated adapter should:

- validate rank, dtype, and declared shape relationships;
- obtain XLA's CUDA stream;
- wrap input and destination buffers as typed views;
- invoke the launch function;
- turn immediate CUDA errors into XLA FFI errors;
- publish architecture, alignment, and divisibility requirements in the
  manifest.

Until such an adapter exists, exporting the current XLA FFI C handler directly
remains valid.

Raw handlers must not allocate result buffers or synchronize the supplied
stream. They may use registers and dynamic shared memory and may enqueue
multiple operations. They must not allocate a global temporary until the
artifact manifest can declare a workspace buffer that PJRT passes explicitly;
scratch must not become an untracked allocation hidden from PJRT.

## Specialization and tuning

Triton-like kernels depend on static meta-parameters. OCaml can generate
candidates directly:

```ocaml
let grouped_gemm_configs =
  List.concat_map
    (fun block_m ->
      List.concat_map
        (fun block_n ->
          [
            Config.make
              ~block_m
              ~block_n
              ~block_k:32
              ~num_warps:(if block_m >= 64 then 8 else 4)
              ~num_stages:2
              ~grid_order:`Rows_first;
            Config.make
              ~block_m
              ~block_n
              ~block_k:32
              ~num_warps:4
              ~num_stages:3
              ~grid_order:`Columns_first;
          ])
        [ 32; 64; 128 ])
    [ 16; 32; 64 ]
```

Each candidate, concrete signature dtype, and target combination is a separate
static specialization. No tile size or stage count is a dynamic device value.

An AOT tuning flow can:

1. construct candidates with ordinary OCaml;
2. reject configurations unsupported by the target;
3. compile them and record registers and shared memory;
4. run numerical checks against the Raven reference;
5. benchmark representative problem buckets;
6. emit the selected specializations and a dispatch table.

Selection proceeds in three stages:

1. An offline tuning command asks PJRT/XLA to compile and benchmark candidate
   TTIR configurations for the requested device class.
2. The tuning result emits retained configurations and deterministic guards as
   OCaml build output.
3. PJRT lowering selects a configuration from static dtype and shape
   descriptors, then embeds its TTIR and launch metadata in the StableHLO
   custom call. XLA compiles that selection with the containing executable.

Either static selection or the generated dispatch may key on:

- target architecture;
- dtype;
- `rows`, `groups`, `k`, and `n`;
- declared alignment;
- divisibility classes.

Selection must not read device-resident `group_sizes`, because that would
introduce a host synchronization. Tuning workloads should instead cover
representative balanced, empty, and skewed group distributions.

Always retain a safe generic DSL or CUDA variant. PJRT executable compilation
may compile the selected TTIR, but execution never benchmarks or compiles on
the first kernel launch.

Raw CUDA variants may participate in the same tuning run. This gives a direct
answer to whether a compiler-generated DSL schedule is good enough for a
problem bucket without forcing the two implementations into one source
language.

## Build and PPX integration

The PPX should not discover source files or run a compiler. Dune owns source
dependencies, TTIR generation, offline tuning workloads, and generated OCaml
descriptors. XLA owns Triton compilation as part of PJRT executable
compilation. NVCC remains relevant only to the separate raw-CUDA path.

A possible Dune-facing shape is:

```lisp
(triton_kernel
 (name grouped_gemm_cuda)
 (dsl_modules grouped_gemm_kernel)
 (targets sm_80 sm_86)
 (tuning grouped_gemm_workloads))
```

It would produce:

```text
Grouped_gemm_cuda.ml
Grouped_gemm_cuda.mli
grouped_gemm_cuda.tuning
```

The generated OCaml module exposes a typed kernel descriptor. An ideal future
annotation references that value rather than repeating paths and symbol
strings:

```ocaml
let grouped_gemm ~lhs ~rhs ~group_sizes =
  Grouped_gemm.reference ~lhs ~rhs ~group_sizes
[@@rune.kernel.cuda { fwd = Grouped_gemm_cuda.forward }]
```

The current PPX accepts only literal `library`, `fwd`, and `bwd` strings and
one tensor argument. Supporting this form requires deliberate PPX work, but
the kernel DSL should generate the artifact and manifest independently of that
syntax.

The PJRT executable and tuning cache keys must include:

- DSL source and transitive kernel-module dependencies;
- static configuration;
- compiler and renderer revisions;
- target compute capabilities;
- XLA/PJRT plugin build identity;
- generated signature and selected TTIR.

## PJRT-owned compilation and runtime

The kernel DSL belongs to the PJRT infrastructure. The compiler pipeline is:

```text
staged OCaml DSL builder
  -> PJRT blocked-kernel IR
  -> schedule and legalize blocks
  -> TTIR
  -> StableHLO XLA Triton custom call
  -> XLA's Triton/LLVM GPU compiler
  -> PJRT executable
```

The blocked IR should preserve `Block.dot`, masked block loads, and reductions
until scheduling has chosen a valid CUDA implementation. It is not part of
Rune's general JIT lowering, and its artifacts have no non-PJRT execution path.
This also requires no new Nx backend operation: the ordinary Raven function is
the tensor operation, while the kernel family is a PJRT-specific replacement.

The Raven-to-TTIR compiler may eventually need internal IR support for structured async
copies and pipelines to generate competitive `Block.dot` implementations.
Those nodes can remain compiler-internal rather than becoming public OCaml
operations.

## Verification

The DSL compiler should reject:

- loads or stores whose block shapes do not broadcast;
- `Block.dot` with incompatible M, N, or K dimensions;
- collective operations under varying control flow;
- hints justified by neither the selected guard nor coordinate analysis;
- configurations exceeding target launch, shared-memory, or register limits;
- dtype or target combinations with no legal dot lowering;
- dynamic values used where a static block shape or loop step is required.

Masked coordinates need not all be statically in bounds, but every lane with a
true load or store mask must be valid. The compiler should prove this for
recognized coordinate forms; otherwise it remains an explicit author
obligation exercised by bounds-checking builds. The compiler can check local
consistency of store value, coordinate, and mask shapes, but cannot generally
prove that an arbitrary grid and segmented helper cover every logical output
exactly once. Global coverage remains an author obligation tested against the
Raven reference, with an optional debug mode that poison-fills destinations
before launch. The compiler should retain source locations so errors point to
the kernel expression rather than generated CUDA.

Generated variants require:

- comparison against the Raven reference;
- awkward boundary dimensions;
- zero-sized and highly unbalanced groups;
- dynamic changes to `group_sizes` across calls;
- all claimed dtypes and architecture guards;
- sanitizer or bounds checking in a debug compilation mode.

Performance tests should separately report isolated GPU time and end-to-end
PJRT time. A faster kernel launch does not imply a faster application when
compile, registration, or host overhead dominates.

## Forward and backward kernels

Kernel authoring remains separate from Rune transformation semantics:

```ocaml
let kernels =
  Kernel_binding.make
    ?primal
    ?vjp
    ()
```

Here `primal` and `vjp` are optional kernel-family descriptors and at least one
must be present. `primal` corresponds to the current `fwd` handler; it is not a
JVP. If no VJP kernel is supplied, Rune differentiates the ordinary Raven body.
If only a VJP kernel is supplied, the Raven body computes the primal.

Automatic differentiation of the blocked DSL is not required initially. It is
better to retain a correct Raven fallback than to promise differentiation
through compiler scheduling, raw CUDA, barriers, or external device code.

## Deliberate omissions

The first design should not expose:

- explicit shared-memory allocation or layout;
- thread, lane, or warp IDs;
- warp shuffle and ballot;
- barriers;
- asynchronous copy groups;
- MMA instruction names or fragment layouts;
- inline PTX;
- arbitrary external device calls;
- dynamic result allocation;
- hidden saved residuals;
- first-launch compilation or benchmarking.

Each omission has a clear answer: use a raw CUDA variant while a recurring
high-level need is evaluated. A primitive should move into the DSL only after
multiple kernels show that it belongs at the blocked-program level.

## Smallest useful prototype

A disciplined prototype can proceed in this order:

1. Keep the implemented TTIR-to-XLA compiler bridge covered by a CUDA
   execution test.
2. Define typed signature handles, `Shape`, `Config`, `Kernel`, `Grid`,
   `Problem`, `Program`, `Scalar`, `Predicate`, `Block`, `Tensor`, and
   `Control`.
3. Generate a simple elementwise TTIR kernel and compare it with the Raven
   fallback.
4. Add masked block loads/stores and reductions.
5. Add `Block.dot` and lower it to Triton's semantic dot operation.
6. Express the generic grouped GEMM entirely in the DSL.
7. Put the current tensor-core CUDA kernel and the DSL kernel behind one
   signature and deterministic variant selection.
8. Measure both across balanced and skewed MoE workloads before expanding the
   public API.

The grouped GEMM is a strong design target because the concise algorithm needs
only blocked coordinates, masks, a loop, and `dot`, while the tuned
implementation demonstrates exactly why raw CUDA must remain available.

## Open questions

1. Should the core embedding remain combinator-only, or should a later PPX
   translate more natural kernel control-flow syntax?
2. Should `Segment.owner_of_tile` be a normal library helper, a compiler-known
   operation, or should routing provide an explicit tile-to-group table?
3. How many `Block.dot` schedule families are sufficient before automatic
   search is needed?
4. What is the smallest generated raw-CUDA adapter that removes XLA FFI
   boilerplate without hiding CUDA?
5. How should scratch buffers and multiple results appear in the generated
   manifest?
6. Should tuning produce one checked-in target profile, a local build artifact,
   or a distributable device-class database?
7. Which problem should follow grouped GEMM: fused softmax, normalization, or a
   fused gated feed-forward epilogue?
8. Should trusted segmented metadata carry a router-produced provenance value,
   or should validation always be an explicit user-selected operation?

The key decision is already fairly clear: keep the common path blocked and
small, and keep exact GPU control in CUDA until the same need appears often
enough to justify a principled high-level primitive.
