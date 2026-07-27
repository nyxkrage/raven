# Triton language coverage

This document inventories the public Python `triton.language` module and maps
it to `Rune_pjrt.Triton.Dsl`.

The upstream snapshot is Triton commit
[`a59d32fd`](https://github.com/triton-lang/triton/blob/a59d32fd4fcd9a50d8361356d415402d753fe0e3/python/triton/language/__init__.py),
from 2026-07-25. Its `__all__` contains 144 names. This is the definition of
"full module" used below: imported implementation details that are absent from
`__all__` are excluded, while public names that do not have an individual
documentation page are included. The
[operation reference](https://triton-lang.org/main/python-api/triton.language.html)
is useful context but is not itself a complete export list.

The comparison is about author-visible semantics rather than internal compiler
implementation details.

## Complete public inventory

| Area | Public names | Raven today | Missing or intentionally different |
| --- | --- | --- | --- |
| Frontend and metadata | `PropagateNan`, `TRITON_MAX_TENSOR_NUMEL`, `condition`, `const`, `constexpr`, `constexpr_type`, `extra`, `math`, `target_info` | Ordinary OCaml values, `Spec`, specialization guards, and `Dsl.Config` provide static staging. | These Python frontend objects should not be copied one-for-one. Raven still needs target capability queries and reduction NaN policy where those semantics become relevant. `extra` and `math` are namespace modules, not kernel operations. |
| Types | `bfloat16`, `block_type`, `dtype`, `float16`, `float32`, `float64`, `float8e4b15`, `float8e4nv`, `float8e4b8`, `float8e5`, `float8e5b16`, `int1`, `int16`, `int32`, `int64`, `int8`, `pi32_t`, `pointer_type`, `slice`, `tensor`, `tuple`, `uint16`, `uint32`, `uint64`, `uint8`, `void` | `Dsl.Dtype` has `f16`, `bf16`, `f32`, `i1`, `i32`, and `i64`. `'a Value.t` and `'a Pointer.t` distinguish typed values and pointers; values carry a validated static block shape. | No unsigned, narrow integer, f64, or FP8 values and no tuple or slice values. Several upstream names are Python type-system machinery and intentionally become OCaml types rather than matching values. |
| Programming model | `program_id`, `num_programs`, `map_elementwise` | `Value.program_id`, `Value.num_programs`, and the general `Kernel.define` body are public. | There is no separate block-valued `map_elementwise` operation; ordinary typed combinators construct its equivalent inside a kernel body. |
| Creation and conversion | `arange`, `cast`, `full`, `to_tensor`, `zeros`, `zeros_like` | `Value.arange`, `cast`, `full`, `zeros`, and typed `float`, `int`, and `bool` scalar constructors are public. | `zeros_like` is unnecessary when dtype and shape are directly available. `to_tensor` is Python scalar-conversion machinery rather than a needed OCaml operation. |
| Shape and indexing | `broadcast`, `broadcast_to`, `cat`, `expand_dims`, `flip`, `gather`, `interleave`, `join`, `permute`, `ravel`, `reshape`, `split`, `squeeze`, `swizzle2d`, `trans`, `unsqueeze`, `view`, `where` | Automatic broadcasting plus `broadcast_to`, `expand_dims`, `permute`, `reshape`, and `where` are public. Pointer offsets provide arbitrary gather/scatter-style global-memory coordinates. | Block-local cat, flip, gather, interleave, join, split, and swizzle helpers are absent. Ravel, transpose, unsqueeze, squeeze, and view are covered by reshape/permute/expand operations without aliases. |
| Math | `abs`, `add`, `cdiv`, `ceil`, `clamp`, `cos`, `div_rn`, `erf`, `exp`, `exp2`, `fdiv`, `floor`, `fma`, `log`, `log2`, `maximum`, `minimum`, `mul`, `rsqrt`, `sigmoid`, `sin`, `softmax`, `sqrt`, `sqrt_rn`, `sub`, `umulhi` | All except unsigned high-half multiply are available. `div` is the fast/default form and `div_rn` and `sqrt_rn` select Triton's precise operations. | `umulhi` remains absent pending unsigned dtypes. Raven deliberately exposes one name per operation rather than mirroring aliases. |
| Linear algebra | `dot`, `dot_scaled` | `Value.dot` supports rank-two f16, bf16, or f32 blocks with an f32 accumulator. | FP8/microscaling dtypes and `dot_scaled` remain absent. |
| Memory and pointers | `load_tensor_descriptor`, `store_tensor_descriptor`, `make_tensor_descriptor`, `tensor_descriptor`, `load`, `make_block_ptr`, `store` | Typed `Signature` inputs, `Pointer.offset`, masked `Pointer.load`, and masked `Statement.store` provide typed scalar and blocked memory access. | Cache/eviction policies, boundary-check sugar, block pointers, tensor descriptors, and TMA remain absent. |
| Reductions | `argmax`, `argmin`, `max`, `min`, `reduce`, `reduce_or`, `sum`, `xor_sum` | Typed block `sum`, `max`, and `min` reductions are public and lower to `tt.reduce`. | Arg reductions, Boolean/XOR reductions, and a public custom combine region remain absent. |
| Scans and sorting | `associative_scan`, `bitonic_merge`, `cumprod`, `cumsum`, `histogram`, `sort`, `topk` | None. | Needed later for routing, selection, and prefix algorithms. `bitonic_merge` is an implementation building block and does not need to be a first-class Raven API merely because upstream exports it. |
| Atomics | `atomic_add`, `atomic_and`, `atomic_cas`, `atomic_max`, `atomic_min`, `atomic_or`, `atomic_poll`, `atomic_xchg`, `atomic_xor` | None. | Integer and pointer prerequisites now exist. Add, max, exchange, and CAS are the likely first additions; polling should wait for a concrete need. |
| Random numbers | `pair_uniform_to_normal`, `philox`, `philox_impl`, `rand`, `rand4x`, `randint`, `randint4x`, `randn`, `randn4x`, `uint_to_uniform_float` | None. | Random generation and explicit counter/seed semantics are absent. Raven should prefer a small public stateless RNG surface; upstream's Philox helpers need not all be public. |
| Iteration | `range`, `static_range` | `Value.range` emits a typed device loop with one loop-carried value. `Dsl.static_range` performs construction-time unrolling. | Multiple simultaneous loop-carried values and while-style control remain absent. |
| Inline assembly | `inline_asm_elementwise` | None. | Raw CUDA is Raven's whole-kernel escape hatch. Inline PTX can remain absent until a kernel demonstrates that whole-kernel CUDA is too coarse. |
| Compiler hints and synchronization | `assume`, `debug_barrier`, `expect_zero`, `max_constancy`, `max_contiguous`, `multiple_of` | `Dsl.Config` selects block size, warp count, and pipeline stages. | No value facts, alignment/contiguity hints, expected-underflow marker, or explicit barrier. Alignment and contiguity facts belong in the next performance-contract layer. |
| Debugging | `device_assert`, `device_print`, `static_assert`, `static_print` | Construction-time validation, `Statement.static_assert`, and `Kernel.to_ttir_for` provide specialization diagnostics. | Device print/assert and a static print helper remain absent. |

Every name in the pinned module's `__all__` appears exactly once in the table.

## Language semantics outside `__all__`

The export list is not the whole Triton authoring model. A Python
`tl.tensor` also supplies operators and indexing syntax, while `@triton.jit`
compiles Python statements and control flow. Raven still lacks:

- shifts and implicit
  [Triton type promotion](https://triton-lang.org/main/python-api/triton-semantics.html);
- statement-level conditionals, early returns, and multiple results;
- functions callable from other kernel functions;
- runtime scalar kernel parameters that are not buffers.

Raven deliberately uses explicit `cast` rather than Triton's implicit
promotion. Mixed f16/f32 programs and f32 dot accumulation are supported.
Within `fun%rune.kernel`, a scalar literal adopts the dtype of its neighboring
staged value rather than promoting or otherwise changing the expression dtype.

The following adjacent APIs live in the root `triton` module rather than
`triton.language`, so they are not part of the 144-name inventory:

- `jit`;
- `autotune`;
- `heuristics`;
- `Config`.

Raven has `Kernel.define`, a fixed `Dsl.Config`, specialization guards, static
grid functions, and a shape-specialized TTIR cache, but no
kernel-function frontend, configuration families, selection keys, benchmark
harness, or persistent autotuning cache. Those are supporting compiler/runtime
work rather than value-level language operations.

Target-specific APIs under `triton.language.extra.cuda`, including
programmatic dependent launch, are also outside the base module. They should
not be part of the portable Raven DSL initially; raw CUDA remains available
when a kernel needs CUDA-specific control.

## What blocks useful Raven kernels

Counting missing names is less useful than identifying semantic cut lines.
Raven now has the general blocked and tiled-compute core. The remaining work is
performance metadata and advanced algorithm families.

### 1. General blocked core — implemented

The implemented `Kernel.define` IR provides:

- scalar and block values;
- `i1`, `i32`, and `i64` in addition to the current floating dtypes;
- explicit casts rather than a large implicit-promotion system;
- kernel pointer arguments and specialization-time scalar parameters;
- `program_id`, `num_programs`, and `arange`;
- comparison, Boolean, integer, and floating arithmetic;
- broadcasting, dimension insertion, reshape, and `where`;
- pointer arithmetic plus masked `load` and `store`;
- a grid function that may depend on static problem dimensions.

This layer is enough for arbitrary pointwise kernels, gathers, simple
transposes, and tiled copies. `Kernel.define` is the only authoring path.

### 2. Tiled compute — implemented

The implemented tiled layer provides:

- `static_range` and a device `range`;
- `zeros` and typed accumulators;
- `sum`, `max`, and `min` reductions;
- `dot` with f32 accumulation;
- mixed storage and compute dtypes;
- `log`, `log2`, `exp2`, `rsqrt`, `erf`, `fma`, and precise/fast math choices.

This makes matmul, the compute core of grouped GEMM, softmax, layer
normalization, fused activations, and attention components expressible. CUDA
tests exercise softmax and a tiled f16/f32 GEMM rather than only isolated IR
rendering.

### 3. Performance contracts and configuration families

Add alignment, divisibility, contiguity, and constant-range facts to the IR,
then support several `Dsl.Config` candidates per kernel. AOT tuning should:

- specialize candidates from build-known shapes and dtypes;
- compile them through the same XLA-bundled Triton pipeline;
- benchmark representative dynamic-shape buckets when a build machine has the
  requested GPU;
- persist the selected configuration with its target capability and compiler
  identity;
- retain a principled heuristic when build-time measurement is unavailable.

These are the pieces that let the DSL communicate enough structure for
Triton's optimizer to outperform a generic fused XLA graph.

### 4. Advanced algorithms

Add atomics, scans, sorting, stateless RNG, FP8/scaled dot, block pointers, and
tensor descriptors only as concrete kernels require them. They expand coverage
to routing, sampling, dropout, quantized matmul, and descriptor/TMA-based
pipelines, but none should delay the general blocked core.

## Coverage by target kernel

| Kernel family | Current DSL | First missing capability |
| --- | --- | --- |
| Fused bias/activation/scale | Expressible | Application-specific schedule selection |
| GELU using `erf` | Expressible | Application-specific schedule selection |
| Softmax | Expressible and CUDA-tested | Configuration families for varying row widths |
| Layer/RMS normalization | Expressible | Configuration families for varying row widths |
| Dense GEMM | Expressible; tiled GEMM is CUDA-tested | Autotuned tile and warp configurations |
| Packed grouped GEMM | Core arithmetic is expressible | Efficient dynamic group/tile lookup and tuning |
| MoE routing | Partially expressible | Atomics, scan, and block-local sorting |
| Dropout | Not expressible | Stateless RNG |
| Flash-style attention | Core operations are expressible | Multiple loop-carried values and tuned schedules |

The practical goal is not 144 matching OCaml names. It is a compact,
orthogonal core from which the high-level wrappers can be defined. The first
two stages provide that core; later stages should be pulled by measured kernel
needs.
