# PJRT custom kernels

Status: exploratory implementation. The prebuilt-library API, unary-function
PPX, forward and reverse-mode custom calls, and CUDA loading path described
below are implemented, but none of them is stable. CUDA source compilation,
multiple arguments, and multiple results remain design work.

This document describes how Raven functions can use custom CUDA kernels inside
PJRT programs without giving up a normal Raven implementation. The regular
function body remains the reference and fallback implementation. When the
function is traced for the PJRT CUDA backend, Rune can replace its primal
computation, reverse-mode VJP, or both with typed XLA FFI custom calls.

The intended properties are:

- calling a kernel looks like calling an ordinary OCaml function;
- the function continues to work eagerly and on non-CUDA backends;
- output shapes and dtypes come from the fallback, rather than a second shape
  API;
- forward and backward CUDA handlers can be supplied independently;
- CUDA code receives XLA's stream and destination buffers directly;
- loading, registration, and StableHLO are hidden from application code;
- a broken CUDA kernel is reported as an error instead of silently changing the
  execution path.

## Current user experience

An annotated function contains its complete Raven implementation:

```ocaml
let scale_two x =
  Nx.mul_s x 2.
[@@rune.kernel.cuda
  {
    library = "libscale_two.so";
    fwd = "raven_scale_two_fwd";
    bwd = "raven_scale_two_bwd";
  }]
```

`fwd` names the primal handler; it does not mean a forward-mode JVP. `bwd`
names a reverse-mode VJP handler. Both fields are optional, but at least one
must be present:

| Fields | Primal computation | Reverse-mode VJP |
| --- | --- | --- |
| `fwd` and `bwd` | Forward CUDA handler | Backward CUDA handler |
| only `fwd` | Forward CUDA handler | Differentiate the Raven body |
| only `bwd` | Execute the Raven body | Backward CUDA handler |

This lets a kernel author optimize one direction without having to implement
the other one first.

It remains an ordinary function at call sites:

```ocaml
let predict =
  Rune_pjrt.jit ~backend:`Cuda (fun x ->
      Nx.sin (scale_two x))

let x =
  Nx.create Nx.float32 [| 8 |]
    [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]

let y = predict x

let gradient =
  Rune_pjrt.jit ~backend:`Cuda
    (Rune.grad (fun x -> Nx.sum (scale_two x)))

let dx = gradient x
```

No explicit plugin loading, handler registration, output allocation, or
StableHLO should appear in application code.

The application enables the rewriter in the usual way:

```lisp
(executable
 (name main)
 (libraries nx rune rune-pjrt)
 (preprocess (pps ppx_rune_kernel)))
```

`ppx_rune_kernel` is a companion PPX package. The program still links normally
when CUDA is unavailable because the generated wrapper retains the original
OCaml body. The shared library is first opened by `Runtime.execute` when the
traced CUDA program actually contains its handler.

A relative `library` path is resolved from the final executable's directory,
not the process working directory or the annotated source file. Dune and
installation rules must preserve that executable-relative layout. Absolute
paths are also accepted. The file need not exist for eager or CPU fallback,
but must exist when a CUDA trace selects the handler.

The attribute uses a CUDA-specific namespace deliberately. A future backend
should get its own attribute instead of adding backend switches to the function
body, for example `rune.kernel.rocm`.

### CUDA source form

The attribute parser recognizes `source` so that it can give a direct error,
but source compilation is not implemented. Kernel authors must currently build
a shared library and use `library`. A future source form may look like:

```ocaml
let scale_two x =
  Nx.mul_s x 2.
[@@rune.kernel.cuda
  {
    source = "scale_two.cu";
    fwd = "raven_scale_two_fwd";
    bwd = "raven_scale_two_bwd";
  }]
```

The PPX currently rejects this with `source compilation is not implemented`.
Adding it requires Dune dependency tracking, compiler configuration, and an
artifact cache rather than a PPX-only change.

## CUDA handlers

The referenced source exports typed XLA FFI handlers. Each kernel launches on
the stream supplied by XLA and writes into result buffers supplied by XLA:

```cpp
#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"

using F32Vector =
    xla::ffi::BufferR1<xla::ffi::DataType::F32>;

__global__ void ScaleTwoKernel(
    const float* input,
    float* output,
    size_t count) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = input[i] * 2.0f;
}

xla::ffi::Error ScaleTwoForward(
    cudaStream_t stream,
    F32Vector input,
    xla::ffi::Result<F32Vector> output) {
  size_t count = input.element_count();
  int threads = 128;
  int blocks = static_cast<int>((count + threads - 1) / threads);

  ScaleTwoKernel<<<blocks, threads, 0, stream>>>(
      input.typed_data(), output->typed_data(), count);

  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(error));
  }
  return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    raven_scale_two_fwd,
    ScaleTwoForward,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32Vector>()
        .Ret<F32Vector>());

xla::ffi::Error ScaleTwoBackward(
    cudaStream_t stream,
    F32Vector input,
    F32Vector output,
    F32Vector output_cotangent,
    xla::ffi::Result<F32Vector> input_cotangent) {
  size_t count = input.element_count();
  int threads = 128;
  int blocks = static_cast<int>((count + threads - 1) / threads);

  // The common VJP ABI includes the primal output for kernels that need it.
  (void)output;
  ScaleTwoKernel<<<blocks, threads, 0, stream>>>(
      output_cotangent.typed_data(), input_cotangent->typed_data(), count);

  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(error));
  }
  return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    raven_scale_two_bwd,
    ScaleTwoBackward,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32Vector>()
        .Arg<F32Vector>()
        .Arg<F32Vector>()
        .Ret<F32Vector>());
```

This typed C++ form is the intended API, but the current upstream `ffi.h`
emits warnings under the measured NVCC 13.3 toolchain. The warning-free fused
softmax experiment exports the same typed handler ABI directly through
`xla/ffi/api/c_api.h`; see `kernels/causal_scaled_softmax.cu`.

For a unary, single-result function, the current backward ABI is:

```text
bwd(primal input, primal output, output cotangent) -> input cotangent
```

For future multiple inputs and outputs, the general order would be all primal
inputs, all primal outputs, then all output cotangents, returning one cotangent
for each differentiable input. Passing primal outputs avoids forcing kernels
such as softmax to recompute useful forward values. Arbitrary saved residuals
are out of scope for the first version.

Handlers must not allocate their results or synchronize the stream. They may
enqueue more than one CUDA operation on the supplied stream. Immediate launch
errors should be returned through XLA FFI.

## PPX expansion

Conceptually, `ppx_rune_kernel` retains the body in a local fallback, creates a
kernel descriptor, and wraps the function in a custom VJP:

```ocaml
let scale_two x =
  let fallback x = Nx.mul_s x 2. in
  let kernel =
    Rune_pjrt.Ffi.Kernel.create
      ~library:"libscale_two.so"
      ~fwd:"raven_scale_two_fwd"
      ~bwd:"raven_scale_two_bwd"
      ()
  in
  Rune.custom_vjp
    ~fwd:(fun x ->
      let y =
        Rune_pjrt.Ffi.call_fwd kernel
          ~inputs:[ Rune_pjrt.Ffi.Tensor x ]
          ~fallback:(fun () -> fallback x)
      in
      y, (x, y))
    ~bwd:(fun (x, y) dy ->
      Rune_pjrt.Ffi.call_bwd kernel
        ~inputs:
          [ Rune_pjrt.Ffi.Tensor x;
            Rune_pjrt.Ffi.Tensor y;
            Rune_pjrt.Ffi.Tensor dy ]
        ~fallback:(fun () -> Stdlib.snd (Rune.vjp fallback x dy)))
    x
```

The real expansion uses generated local names to avoid adding top-level values.
When `fwd` is absent, `call_fwd` directly invokes its fallback and records no
forward custom call. When `bwd` is absent, `call_bwd` differentiates the
unannotated body and records its ordinary Raven operations.

The generated local names are illustrative; the `Rune_pjrt.Ffi` calls shown are
the current runtime API. The wrapper is type-transparent to its caller: the
inferred type of `scale_two` is the same as the type of the unannotated
definition.

The typed one-result internal interface and packed input list avoid public
`call1`, `call2`, and similar arity-specific APIs. The PPX knows the syntactic
arguments and hides input packing from the user.

The current PPX supports only:

- non-recursive functions;
- one positional tensor argument bound to a simple name;
- one tensor result;
- a literal `library` path;
- literal `fwd` and `bwd` symbol names, with at least one present;
- custom handlers for primal and first-order reverse-mode computation.

Tuples, multiple results, labelled arguments, recursive definitions, and
arbitrary pytrees should wait until concrete kernels require them.

## Dispatch semantics

The fallback is selected by execution context, not by catching arbitrary CUDA
errors.

| Context | Behaviour |
| --- | --- |
| Ordinary eager execution | Execute the Raven body. |
| Rune/Tolk JIT | Trace the Raven body into normal Rune kernels. |
| PJRT CPU | Lower the Raven body to ordinary StableHLO operations. |
| PJRT CUDA primal, with `fwd` | Emit the forward FFI custom call. |
| PJRT CUDA primal, without `fwd` | Lower the Raven body. |
| PJRT CUDA reverse mode, with `bwd` | Emit the backward FFI custom call. |
| PJRT CUDA reverse mode, without `bwd` | Differentiate and lower the Raven body. |
| Forward-mode AD, `vmap`, or higher-order AD | Transform the Raven body. |
| CUDA handler compilation, registration, or launch fails | Report the error. |

Falling back after a CUDA failure would hide incorrect kernels and make
performance unpredictable. The fallback is for unsupported execution contexts,
not error recovery.

The fallback body must therefore be pure tensor code. It may run eagerly during
tracing, so observable OCaml side effects in the body are invalid.

## Trace and lowering flow

During a PJRT CUDA trace, the wrapper performs a custom-kernel effect carrying:

- the kernel artifact and selected forward or backward symbol;
- packed input tensors;
- a fallback thunk.

The trace handler then:

1. runs the fallback outside the custom-kernel handler;
2. uses its concrete result to obtain output shapes and dtypes;
3. records a custom-call IR node using the inputs and result descriptors;
4. returns the eager result so tracing of subsequent operations can continue.

This follows the current PJRT tracer's eager shape-discovery model. It also
keeps shape logic in one place: the Raven implementation.

The forward node lowers to StableHLO similar to:

```mlir
%result = "stablehlo.custom_call"(%input) {
  api_version = 4 : i32,
  backend_config = {},
  call_target_name = "raven_cuda_7dd0...",
  operand_layouts = [dense<[0]> : tensor<1xindex>],
  result_layouts = [dense<[0]> : tensor<1xindex>]
} : (tensor<8xf32>) -> tensor<8xf32>
```

The backward node receives the primal input, primal output, and output
cotangent:

```mlir
%dinput = "stablehlo.custom_call"(%input, %result, %dresult) {
  api_version = 4 : i32,
  backend_config = {},
  call_target_name = "raven_cuda_451f...",
  operand_layouts = [
    dense<[0]> : tensor<1xindex>,
    dense<[0]> : tensor<1xindex>,
    dense<[0]> : tensor<1xindex>
  ],
  result_layouts = [dense<[0]> : tensor<1xindex>]
} : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
```

Raven constrains every custom-call buffer to the canonical contiguous layout
(last tensor axis minor-most). XLA therefore inserts any required layout copy
before or after the call; raw FFI handlers can safely interpret their buffers
as packed row-major arrays even when a neighboring graph operation is a
transpose.

The target is `raven_cuda_` followed by a digest of the shared-library contents
and handler symbol. This prevents two libraries from registering incompatible
handlers under the same PJRT target and ensures that changing either the
library or symbol invalidates executable caches.

The library is canonicalized and hashed during CUDA tracing, then rehashed
before its first execution. A change between those points is rejected. Once a
handler is registered, its library remains loaded and is treated as immutable;
rebuilding the same path in a live process is rejected on the next registration.
Use versioned filenames or restart the process when iterating on a handler.

## Compilation and loading

The PPX is responsible only for rewriting OCaml. It does not define Dune rules
for external CUDA files. Applications currently build the shared library with
their own rule and put that path in the attribute. The fused softmax experiment
shows one such rule in `kernels/dune`.

That experiment is deliberately opt-in: set
`RUNE_PJRT_CUDA_KERNELS=enabled` when building its CUDA library, benchmark, or
integration test. Normal builds do not require `nvcc` or the local XLA FFI
header tree.

The implemented ahead-of-time pipeline is:

```text
scale_two.cu
    -> CUDA compiler and XLA FFI headers
    -> prebuilt shared library at an executable-relative or absolute path
    -> PPX-generated descriptor and content-digested call target
    -> dlopen and dlsym once per process
    -> PJRT FFI registration before PJRT_Client_Compile
    -> StableHLO custom call
    -> CUDA launch on XLA's stream
```

The current implementation follows the ahead-of-time path, so applications do
not invoke `nvcc` at runtime. Source compilation and its artifact cache remain
future work.

Any future source artifact cache key must include at least:

- CUDA source contents;
- included project headers;
- exported forward and backward symbols;
- CUDA compiler and flags;
- target compute capabilities;
- XLA FFI header or ABI version.

The implemented runtime loads the PJRT plugin, walks its extension chain to
`PJRT_Extension_Type_FFI`, resolves each supplied handler with `dlsym`, and
calls `register_handler` for platform `CUDA`. Both the PJRT plugin and handler
library must remain loaded while any cached executable can call either handler.

Registration happens before PJRT compilation. A missing library, symbol, FFI
extension, or failed registration is reported as an error; the runtime does
not silently execute the fallback after a CUDA failure.

## Transformations

An FFI custom call is opaque to XLA and Rune. The PPX must connect `fwd` and
`bwd` using a custom VJP rather than expecting XLA to differentiate the forward
call.

For first-order reverse mode:

- when `fwd` is present, the primal computation uses that handler;
- when `fwd` is absent, the primal computation uses the Raven body;
- when `bwd` is present, the VJP uses that handler;
- when `bwd` is absent, Rune differentiates the Raven body and uses the resulting
  VJP.

If `fwd` is present but `bwd` is absent, constructing the fallback VJP may
re-evaluate the Raven body during the backward trace to recover intermediates.
That is the conservative correctness path; supplying `bwd` avoids this
fallback computation.

Forward-mode AD is separate: `fwd` means the primal handler, not a JVP handler.
The current `jvp`, `vmap`, nested AD, and Rune/Tolk JIT handlers explicitly
select the Raven body because neither FFI call carries JVP, batching, or
higher-order derivative semantics. Dedicated JVP or batching handlers remain
future design work.

This keeps an annotated function semantically equivalent to its body while
allowing either half of a first-order reverse-mode computation to be
accelerated independently. It does not imply that XLA can differentiate, batch,
fuse, or shard either custom call itself.

## Layouts, aliasing, and results

XLA owns result allocation. The FFI handler receives destination-passed result
buffers and must write all result elements.

The current prototype emits canonical dense layouts for every custom-call
operand and result and supports one result. It does not expose layout or
aliasing controls. Later versions may add carefully designed support for:

- input/output aliasing for in-place kernels;
- multiple results;
- scratch buffers;
- dynamic dimensions;
- command-buffer compatibility.

None of these should become PPX flags initially. They affect correctness and
belong in a typed kernel descriptor or generated manifest.

## Failure reporting

Runtime errors currently distinguish shared-library loading, symbol lookup,
PJRT FFI extension discovery, registration, PJRT compilation, and CUDA handler
errors. Associating every error with the annotated OCaml function is still a
diagnostic improvement to make.

- kernel source compilation, once source form exists;
- shared-library loading;
- exported-symbol lookup;
- PJRT FFI extension discovery;
- handler registration;
- StableHLO/PJRT compilation;
- CUDA launch.

Future debug output should also say whether the FFI kernel or Raven fallback
was selected. This should be diagnostic output, not a public execution-policy
knob.

## Evidence from the feasibility probes

Four CUDA paths have been exercised successfully with the current PJRT plugin
on an NVIDIA RTX 3090:

- an embedded PTX custom call computed `3 + 4 = 7`;
- an embedded Triton/TTIR custom call computed `41 + 1 = 42`;
- a typed XLA FFI handler doubled an eight-element float32 vector;
- the PPX-generated fused causal scaled-softmax forward and VJP handlers
  matched the Raven body through PJRT CUDA up to `[1,12,1024,1024]`.

Typed FFI is the preferred basis for this design because it supplies typed
buffers, destination results, and XLA's CUDA stream through an extension
exported by the loaded PJRT plugin. Embedded PTX remains useful for experiments,
but its `__gpu$xla.gpu.ptx` call target is an internal XLA mechanism.

The fused softmax profile in `bench/SOFTMAX_PROFILE.md` also shows that the
custom forward handler is 11-31% faster at the kernel level than XLA's existing
fused Triton result for the measured GPT-2 shapes.

Relevant upstream interfaces:

- [XLA custom calls](https://openxla.org/xla/custom_call)
- [StableHLO `custom_call`](https://openxla.org/stablehlo/spec#custom_call)
- [PJRT FFI
  extension](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_ffi_extension.h)
- [XLA FFI C++ bindings](../../vendor/xla/xla/ffi/api/ffi.h)

## Open decisions

Before promoting the prototype to a public feature, decide:

1. whether and how to add CUDA source alongside the implemented prebuilt-library
   form;
2. how Dune learns that a CUDA source file is a dependency of an annotated
   function;
3. whether handler signatures get a generated manifest or rely initially on
   XLA FFI's runtime validation;
4. whether the fixed backward ABI is sufficient or a later manifest should
   describe saved residuals;
5. which function shapes the first PPX supports beyond one tensor result;
6. whether to expose JVP or batching handlers instead of always selecting the
   implemented Raven fallback for those transformations;
7. where future compiled source artifacts live without placing unmanaged files
   inside Dune-owned build directories.

The current prototype is one annotated unary function, a prebuilt CUDA library,
and internal forward and backward custom-call IR nodes. Its tests cover `fwd`
only, `bwd` only, both directions, eager, CPU, and Rune JIT graph fallbacks,
JVP, `vmap`, nested AD, content cache invalidation, StableHLO execution, every
CUDA launch regime, and finite-mask numerical agreement. The next step should
be chosen from a concrete kernel need rather than expanding the API
speculatively.
