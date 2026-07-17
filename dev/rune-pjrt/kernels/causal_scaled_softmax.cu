// ---------------------------------------------------------------------------
// Copyright (c) 2026 The Raven authors. All rights reserved.
// SPDX-License-Identifier: ISC
// ---------------------------------------------------------------------------

#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>

#include "xla/ffi/api/c_api.h"

namespace {

constexpr float kScale = 0.125f;
constexpr float kMask = -1.0e9f;

template <int Threads>
__device__ float BlockReduceMax(float value, float* scratch) {
  constexpr int kWarpSize = 32;
  constexpr int kWarps = Threads / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  int warp = threadIdx.x / kWarpSize;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  if (lane == 0) scratch[warp] = value;
  __syncthreads();

  value = threadIdx.x < kWarps ? scratch[lane] : -CUDART_INF_F;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    if (warp == 0) {
      value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
  }
  if (threadIdx.x == 0) scratch[0] = value;
  __syncthreads();
  float result = scratch[0];
  __syncthreads();
  return result;
}

template <int Threads>
__device__ float BlockReduceSum(float value, float* scratch) {
  constexpr int kWarpSize = 32;
  constexpr int kWarps = Threads / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  int warp = threadIdx.x / kWarpSize;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  if (lane == 0) scratch[warp] = value;
  __syncthreads();

  value = threadIdx.x < kWarps ? scratch[lane] : 0.0f;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    if (warp == 0) {
      value += __shfl_down_sync(0xffffffff, value, offset);
    }
  }
  if (threadIdx.x == 0) scratch[0] = value;
  __syncthreads();
  float result = scratch[0];
  __syncthreads();
  return result;
}

template <int Threads, int Items>
__global__ void CausalScaledSoftmaxForwardKernel(const float* scores,
                                                  float* probabilities,
                                                  int sequence) {
  __shared__ float scratch[Threads];
  int row = blockIdx.x;
  int query = row % sequence;
  int64_t offset = static_cast<int64_t>(row) * sequence;
  float values[Items];
  float local_max = -CUDART_INF_F;

#pragma unroll
  for (int item = 0; item < Items; ++item) {
    int key = threadIdx.x + (item * Threads);
    float value =
        key >= sequence
            ? -CUDART_INF_F
            : (key <= query ? scores[offset + key] * kScale : kMask);
    values[item] = value;
    local_max = fmaxf(local_max, value);
  }
  float row_max = BlockReduceMax<Threads>(local_max, scratch);

  float local_sum = 0.0f;
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    float value = expf(values[item] - row_max);
    values[item] = value;
    local_sum += value;
  }
  float inverse_sum = 1.0f / BlockReduceSum<Threads>(local_sum, scratch);

#pragma unroll
  for (int item = 0; item < Items; ++item) {
    int key = threadIdx.x + (item * Threads);
    if (key < sequence) {
      probabilities[offset + key] = values[item] * inverse_sum;
    }
  }
}

template <int Threads, int Items>
__global__ void CausalScaledSoftmaxBackwardKernel(
    const float* probabilities, const float* output_cotangents,
    float* input_cotangents, int sequence) {
  __shared__ float scratch[Threads];
  int row = blockIdx.x;
  int query = row % sequence;
  int64_t offset = static_cast<int64_t>(row) * sequence;
  float primal[Items];
  float cotangent[Items];
  float local_dot = 0.0f;

#pragma unroll
  for (int item = 0; item < Items; ++item) {
    int key = threadIdx.x + (item * Threads);
    float probability = key < sequence ? probabilities[offset + key] : 0.0f;
    float output_cotangent =
        key < sequence ? output_cotangents[offset + key] : 0.0f;
    primal[item] = probability;
    cotangent[item] = output_cotangent;
    local_dot += probability * output_cotangent;
  }
  float dot = BlockReduceSum<Threads>(local_dot, scratch);

#pragma unroll
  for (int item = 0; item < Items; ++item) {
    int key = threadIdx.x + (item * Threads);
    if (key < sequence) {
      input_cotangents[offset + key] =
          key <= query
              ? kScale * primal[item] * (cotangent[item] - dot)
              : 0.0f;
    }
  }
}

XLA_FFI_Error* Error(XLA_FFI_CallFrame* call_frame,
                     XLA_FFI_Error_Code code, const char* message) {
  XLA_FFI_Error_Create_Args arguments;
  arguments.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
  arguments.extension_start = nullptr;
  arguments.message = message;
  arguments.errc = code;
  return call_frame->api->XLA_FFI_Error_Create(&arguments);
}

XLA_FFI_Error* InvalidArgument(XLA_FFI_CallFrame* call_frame,
                               const char* message) {
  return Error(call_frame, XLA_FFI_Error_Code_INVALID_ARGUMENT, message);
}

bool SameShape(const XLA_FFI_Buffer* lhs, const XLA_FFI_Buffer* rhs) {
  if (lhs->rank != rhs->rank) return false;
  for (int64_t i = 0; i < lhs->rank; ++i) {
    if (lhs->dims[i] != rhs->dims[i]) return false;
  }
  return true;
}

XLA_FFI_Error* ValidateBuffer(XLA_FFI_CallFrame* call_frame,
                              const XLA_FFI_Buffer* buffer) {
  if (buffer == nullptr || buffer->struct_size < XLA_FFI_Buffer_STRUCT_SIZE) {
    return InvalidArgument(call_frame, "invalid XLA FFI buffer");
  }
  if (buffer->dims == nullptr) {
    return InvalidArgument(call_frame, "XLA FFI buffer has no dimensions");
  }
  if (buffer->dtype != XLA_FFI_DataType_F32 || buffer->rank != 4) {
    return InvalidArgument(call_frame,
                           "causal scaled softmax requires a rank-4 f32 buffer");
  }
  if (buffer->dims[2] != buffer->dims[3]) {
    return InvalidArgument(
        call_frame, "causal scaled softmax requires square final dimensions");
  }
  if (buffer->dims[3] <= 0 || buffer->dims[3] > 1024) {
    return InvalidArgument(
        call_frame,
        "causal scaled softmax requires a sequence length from 1 to 1024");
  }
  uint64_t rows = 1;
  for (int axis = 0; axis < 3; ++axis) {
    if (buffer->dims[axis] < 0) {
      return InvalidArgument(call_frame,
                             "causal scaled softmax has a negative dimension");
    }
    uint64_t dimension = static_cast<uint64_t>(buffer->dims[axis]);
    if (dimension != 0 &&
        rows > std::numeric_limits<unsigned int>::max() / dimension) {
      return InvalidArgument(
          call_frame,
          "causal scaled softmax has too many rows for a CUDA launch");
    }
    rows *= dimension;
  }
  return nullptr;
}

void LaunchForward(unsigned int rows, int sequence, cudaStream_t stream,
                   const float* scores, float* probabilities) {
  if (sequence <= 128) {
    CausalScaledSoftmaxForwardKernel<128, 1>
        <<<rows, 128, 0, stream>>>(scores, probabilities, sequence);
  } else if (sequence <= 256) {
    CausalScaledSoftmaxForwardKernel<128, 2>
        <<<rows, 128, 0, stream>>>(scores, probabilities, sequence);
  } else if (sequence <= 512) {
    CausalScaledSoftmaxForwardKernel<256, 2>
        <<<rows, 256, 0, stream>>>(scores, probabilities, sequence);
  } else {
    CausalScaledSoftmaxForwardKernel<256, 4>
        <<<rows, 256, 0, stream>>>(scores, probabilities, sequence);
  }
}

void LaunchBackward(unsigned int rows, int sequence, cudaStream_t stream,
                    const float* probabilities,
                    const float* output_cotangents,
                    float* input_cotangents) {
  if (sequence <= 128) {
    CausalScaledSoftmaxBackwardKernel<128, 1><<<rows, 128, 0, stream>>>(
        probabilities, output_cotangents, input_cotangents, sequence);
  } else if (sequence <= 256) {
    CausalScaledSoftmaxBackwardKernel<128, 2><<<rows, 128, 0, stream>>>(
        probabilities, output_cotangents, input_cotangents, sequence);
  } else if (sequence <= 512) {
    CausalScaledSoftmaxBackwardKernel<256, 2><<<rows, 256, 0, stream>>>(
        probabilities, output_cotangents, input_cotangents, sequence);
  } else {
    CausalScaledSoftmaxBackwardKernel<256, 4><<<rows, 256, 0, stream>>>(
        probabilities, output_cotangents, input_cotangents, sequence);
  }
}

XLA_FFI_Error* PopulateMetadata(XLA_FFI_CallFrame* call_frame) {
  auto* extension = reinterpret_cast<XLA_FFI_Metadata_Extension*>(
      call_frame->extension_start);
  if (extension->extension_base.struct_size <
          XLA_FFI_Metadata_Extension_STRUCT_SIZE ||
      extension->metadata == nullptr ||
      extension->metadata->struct_size < XLA_FFI_Metadata_STRUCT_SIZE) {
    return InvalidArgument(call_frame, "invalid XLA FFI metadata extension");
  }
  extension->metadata->api_version = XLA_FFI_Api_Version{
      XLA_FFI_Api_Version_STRUCT_SIZE, nullptr, XLA_FFI_API_MAJOR,
      XLA_FFI_API_MINOR};
  extension->metadata->traits = 0;
  extension->metadata->state_type_id = XLA_FFI_UNKNOWN_TYPE_ID;
  return nullptr;
}

XLA_FFI_Error* ValidateCall(XLA_FFI_CallFrame* call_frame,
                            int64_t argument_count) {
  if (call_frame->struct_size < XLA_FFI_CallFrame_STRUCT_SIZE) {
    return InvalidArgument(call_frame, "invalid XLA FFI call frame");
  }
  if (call_frame->extension_start != nullptr &&
      call_frame->extension_start->type == XLA_FFI_Extension_Metadata) {
    return PopulateMetadata(call_frame);
  }
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return InvalidArgument(call_frame,
                           "causal scaled softmax requires execute stage");
  }
  if (call_frame->args.size != argument_count || call_frame->rets.size != 1) {
    return InvalidArgument(call_frame,
                           "causal scaled softmax has an invalid arity");
  }
  for (int64_t i = 0; i < argument_count; ++i) {
    if (call_frame->args.types[i] != XLA_FFI_ArgType_BUFFER) {
      return InvalidArgument(call_frame,
                             "causal scaled softmax arguments must be buffers");
    }
  }
  if (call_frame->rets.types[0] != XLA_FFI_RetType_BUFFER) {
    return InvalidArgument(call_frame,
                           "causal scaled softmax result must be a buffer");
  }
  return nullptr;
}

cudaStream_t Stream(XLA_FFI_CallFrame* call_frame, XLA_FFI_Error** error) {
  XLA_FFI_Stream_Get_Args arguments;
  arguments.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  arguments.extension_start = nullptr;
  arguments.ctx = call_frame->ctx;
  arguments.stream = nullptr;
  *error = call_frame->api->XLA_FFI_Stream_Get(&arguments);
  return reinterpret_cast<cudaStream_t>(arguments.stream);
}

XLA_FFI_Error* CausalScaledSoftmaxForward(XLA_FFI_CallFrame* call_frame) {
  bool metadata = call_frame->extension_start != nullptr &&
                  call_frame->extension_start->type ==
                      XLA_FFI_Extension_Metadata;
  XLA_FFI_Error* error = ValidateCall(call_frame, 1);
  if (error != nullptr || metadata) return error;
  auto* scores = reinterpret_cast<XLA_FFI_Buffer*>(call_frame->args.args[0]);
  auto* probabilities =
      reinterpret_cast<XLA_FFI_Buffer*>(call_frame->rets.rets[0]);
  if ((error = ValidateBuffer(call_frame, scores)) != nullptr) return error;
  if ((error = ValidateBuffer(call_frame, probabilities)) != nullptr) {
    return error;
  }
  if (!SameShape(scores, probabilities)) {
    return InvalidArgument(call_frame, "forward input and output shapes must match");
  }

  int sequence = static_cast<int>(scores->dims[3]);
  unsigned int rows = static_cast<unsigned int>(
      scores->dims[0] * scores->dims[1] * scores->dims[2]);
  if (rows == 0) return nullptr;
  cudaStream_t stream = Stream(call_frame, &error);
  if (error != nullptr) return error;
  LaunchForward(rows, sequence, stream,
                static_cast<const float*>(scores->data),
                static_cast<float*>(probabilities->data));
  cudaError_t cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return Error(call_frame, XLA_FFI_Error_Code_INTERNAL,
                 cudaGetErrorString(cuda_error));
  }
  return nullptr;
}

XLA_FFI_Error* CausalScaledSoftmaxBackward(XLA_FFI_CallFrame* call_frame) {
  bool metadata = call_frame->extension_start != nullptr &&
                  call_frame->extension_start->type ==
                      XLA_FFI_Extension_Metadata;
  XLA_FFI_Error* error = ValidateCall(call_frame, 3);
  if (error != nullptr || metadata) return error;
  auto* scores = reinterpret_cast<XLA_FFI_Buffer*>(call_frame->args.args[0]);
  auto* probabilities =
      reinterpret_cast<XLA_FFI_Buffer*>(call_frame->args.args[1]);
  auto* output_cotangents =
      reinterpret_cast<XLA_FFI_Buffer*>(call_frame->args.args[2]);
  auto* input_cotangents =
      reinterpret_cast<XLA_FFI_Buffer*>(call_frame->rets.rets[0]);
  XLA_FFI_Buffer* buffers[] = {scores, probabilities, output_cotangents,
                               input_cotangents};
  for (XLA_FFI_Buffer* buffer : buffers) {
    if ((error = ValidateBuffer(call_frame, buffer)) != nullptr) return error;
  }
  if (!SameShape(scores, probabilities) ||
      !SameShape(scores, output_cotangents) ||
      !SameShape(scores, input_cotangents)) {
    return InvalidArgument(
        call_frame, "backward primal, cotangent, and result shapes must match");
  }

  int sequence = static_cast<int>(scores->dims[3]);
  unsigned int rows = static_cast<unsigned int>(
      scores->dims[0] * scores->dims[1] * scores->dims[2]);
  if (rows == 0) return nullptr;
  cudaStream_t stream = Stream(call_frame, &error);
  if (error != nullptr) return error;
  LaunchBackward(rows, sequence, stream,
                 static_cast<const float*>(probabilities->data),
                 static_cast<const float*>(output_cotangents->data),
                 static_cast<float*>(input_cotangents->data));
  cudaError_t cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return Error(call_frame, XLA_FFI_Error_Code_INTERNAL,
                 cudaGetErrorString(cuda_error));
  }
  return nullptr;
}

}  // namespace

extern "C" XLA_FFI_Error* raven_causal_scaled_softmax_fwd(
    XLA_FFI_CallFrame* call_frame) {
  return CausalScaledSoftmaxForward(call_frame);
}

extern "C" XLA_FFI_Error* raven_causal_scaled_softmax_bwd(
    XLA_FFI_CallFrame* call_frame) {
  return CausalScaledSoftmaxBackward(call_frame);
}
