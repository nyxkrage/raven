# PJRT custom kernels

Status: exploratory design. None of the APIs or PPX syntax in this document is
implemented or stable.

This document sketches how Raven functions could use custom CUDA kernels inside
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

## Proposed user experience

An annotated function contains its complete Raven implementation:

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

`ppx_rune_kernel` is a proposed companion package, not an existing library.
The program still links normally when CUDA is unavailable because the generated
wrapper retains the original OCaml body.

The attribute uses a CUDA-specific namespace deliberately. A future backend
should get its own attribute instead of adding backend switches to the function
body, for example `rune.kernel.rocm`.

### Prebuilt-library form

Build systems and deployed applications may prefer to supply a shared library
instead of CUDA source:

```ocaml
let scale_two x =
  Nx.mul_s x 2.
[@@rune.kernel.cuda
  {
    library = "libexample_kernels.so";
    fwd = "raven_scale_two_fwd";
    bwd = "raven_scale_two_bwd";
  }]
```

Exactly one of `source` and `library` would be accepted. The source form is the
convenient development interface; the library form is the reproducible
deployment interface.

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

For a unary, single-result function, the proposed backward ABI is:

```text
bwd(primal input, primal output, output cotangent) -> input cotangent
```

For multiple inputs and outputs, the general order would be all primal inputs,
all primal outputs, then all output cotangents, returning one cotangent for each
differentiable input. Passing primal outputs avoids forcing kernels such as
softmax to recompute useful forward values. Arbitrary saved residuals are out of
scope for the first version.

Handlers must not allocate their results or synchronize the stream. They may
enqueue more than one CUDA operation on the supplied stream. Immediate launch
errors should be returned through XLA FFI.

## PPX expansion

Conceptually, `ppx_rune_kernel` turns the annotated definition into three
values:

```ocaml
let scale_two__raven_fallback x =
  Nx.mul_s x 2.

let scale_two__raven_kernel =
  Rune_pjrt.Ffi.Kernel.cuda_source
    ~source:"scale_two.cu"
    ~fwd:(Some "raven_scale_two_fwd")
    ~bwd:(Some "raven_scale_two_bwd")

let scale_two x =
  Rune.custom_vjp
    ~fwd:(fun x ->
      let y =
        Rune_pjrt.Ffi.dispatch_fwd
          scale_two__raven_kernel
          ~inputs:[ Rune_pjrt.Tensor x ]
          ~fallback:(fun () ->
            [ Rune_pjrt.Tensor (scale_two__raven_fallback x) ])
        |> Rune_pjrt.Ffi.single_result
      in
      y, (x, y))
    ~bwd:(fun (x, y) dy ->
      Rune_pjrt.Ffi.dispatch_bwd
        scale_two__raven_kernel
        ~inputs:
          [ Rune_pjrt.Tensor x;
            Rune_pjrt.Tensor y;
            Rune_pjrt.Tensor dy ]
        ~fallback:(fun () ->
          let _, dx = Rune.vjp scale_two__raven_fallback x dy in
          [ Rune_pjrt.Tensor dx ])
      |> Rune_pjrt.Ffi.single_result)
    x
```

This is a semantic expansion, not a commitment to these internal functions.
When `fwd` is absent, `dispatch_fwd` directly invokes its fallback and records
no forward custom call. When `bwd` is absent, `dispatch_bwd` differentiates the
unannotated body and records its ordinary Raven operations.

These exact internal names are illustrative. The generated wrapper should be
type-transparent to its caller: the inferred type of `scale_two` must be the
same as the type of the unannotated definition.

The packed internal interface avoids public `call1`, `call2`, and similar
arity-specific APIs. The PPX knows the syntactic arguments and hides packing
and unpacking from the user.

An initial PPX can reasonably support only:

- non-recursive functions;
- positional tensor arguments bound to simple names;
- one tensor result;
- a literal `source` or `library` path;
- literal `fwd` and `bwd` symbol names, with at least one present;
- first-order reverse-mode differentiation.

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
%result = stablehlo.custom_call @raven_scale_two_fwd(%input) {
  api_version = 4 : i32,
  backend_config = {}
} : (tensor<8xf32>) -> tensor<8xf32>
```

The backward node receives the primal input, primal output, and output
cotangent:

```mlir
%dinput = stablehlo.custom_call @raven_scale_two_bwd(
    %input, %result, %dresult) {
  api_version = 4 : i32,
  backend_config = {}
} : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
```

The real target names should include an artifact digest and direction, for
example `raven_cuda_7dd0..._raven_scale_two_fwd`. This prevents two libraries
from registering incompatible handlers under the same PJRT target and ensures
that a changed kernel invalidates executable caches.

## Compilation and loading

The PPX is responsible for rewriting OCaml. It cannot, by itself, define Dune
rules for external CUDA files. Kernel compilation therefore needs a companion
build component.

The desired pipeline is:

```text
scale_two.cu
    -> CUDA compiler and XLA FFI headers
    -> content-addressed shared library
    -> PPX-generated kernel descriptor
    -> dlopen and dlsym once per process
    -> PJRT FFI registration before PJRT_Client_Compile
    -> StableHLO custom call
    -> CUDA launch on XLA's stream
```

For an exploratory implementation, `cuda_source` could compile and cache the
library on first CUDA use. A durable implementation should compile ahead of
time so applications do not require `nvcc` at runtime. The same kernel
descriptor can support both paths.

The artifact cache key must include at least:

- CUDA source contents;
- included project headers;
- exported forward and backward symbols;
- CUDA compiler and flags;
- target compute capabilities;
- XLA FFI header or ABI version.

The runtime loads the PJRT plugin, walks its extension chain to
`PJRT_Extension_Type_FFI`, resolves each supplied handler with `dlsym`, and
calls `register_handler` for platform `CUDA`. Both the PJRT plugin and handler
library must remain loaded while any cached executable can call either handler.

Registration must happen before PJRT compilation. A missing handler is a
compilation error, which is preferable to discovering it during a model run.

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
Until a `jvp` design exists, `jvp` should differentiate the Raven body. `vmap`
and higher-order derivatives should also use the body because neither the
forward nor backward FFI call carries batching or derivative semantics.

This keeps an annotated function semantically equivalent to its body while
allowing either half of a first-order reverse-mode computation to be
accelerated independently. It does not imply that XLA can differentiate, batch,
fuse, or shard either custom call itself.

## Layouts, aliasing, and results

XLA owns result allocation. The FFI handler receives destination-passed result
buffers and must write all result elements.

The first version should require contiguous layouts and emit explicit operand
and result layouts. Later versions may add carefully designed support for:

- input/output aliasing for in-place kernels;
- multiple results;
- scratch buffers;
- dynamic dimensions;
- command-buffer compatibility.

None of these should become PPX flags initially. They affect correctness and
belong in a typed kernel descriptor or generated manifest.

## Failure reporting

Errors should identify the annotated OCaml function, direction, and CUDA
symbol. The following phases need distinct messages:

- kernel source compilation;
- shared-library loading;
- exported-symbol lookup;
- PJRT FFI extension discovery;
- handler registration;
- StableHLO/PJRT compilation;
- CUDA launch.

Debug output should also say whether the FFI kernel or Raven fallback was
selected. This should be diagnostic output, not a public execution-policy knob.

## Evidence from the feasibility probes

Three CUDA paths have been exercised successfully with the current PJRT plugin
on an NVIDIA RTX 3090:

- an embedded PTX custom call computed `3 + 4 = 7`;
- an embedded Triton/TTIR custom call computed `41 + 1 = 42`;
- a typed XLA FFI handler doubled an eight-element float32 vector.

Typed FFI is the preferred basis for this design because it supplies typed
buffers, destination results, and XLA's CUDA stream through an extension
exported by the loaded PJRT plugin. Embedded PTX remains useful for experiments,
but its `__gpu$xla.gpu.ptx` call target is an internal XLA mechanism.

Relevant upstream interfaces:

- [XLA custom calls](https://openxla.org/xla/custom_call)
- [StableHLO `custom_call`](https://openxla.org/stablehlo/spec#custom_call)
- [PJRT FFI
  extension](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_ffi_extension.h)
- [XLA FFI C++ bindings](../../vendor/xla/xla/ffi/api/ffi.h)

## Open decisions

Before implementing a public feature, decide:

1. whether the first prototype accepts CUDA source, only prebuilt libraries, or
   both;
2. how Dune learns that a CUDA source file is a dependency of an annotated
   function;
3. whether handler signatures get a generated manifest or rely initially on
   XLA FFI's runtime validation;
4. whether the fixed backward ABI is sufficient or a later manifest should
   describe saved residuals;
5. which function shapes the first PPX supports beyond one tensor result;
6. how forward-mode, batching, and higher-order handlers force the Raven
   fallback;
7. where compiled kernel artifacts live without placing unmanaged files inside
   Dune-owned build directories.

The smallest next prototype is one annotated unary function, a prebuilt CUDA
library exporting both handlers, and internal forward and backward custom-call
IR nodes. Tests should cover `fwd` only, `bwd` only, and both together. This is
enough to validate PPX expansion and fallback selection without committing to
source compilation, multi-result APIs, or more transformation rules.
