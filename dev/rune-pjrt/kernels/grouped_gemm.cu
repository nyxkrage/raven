// ---------------------------------------------------------------------------
// Copyright (c) 2026 The Raven authors. All rights reserved.
// SPDX-License-Identifier: ISC
// ---------------------------------------------------------------------------

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <mma.h>

#include <cstdint>
#include <limits>

#include "xla/ffi/api/c_api.h"

namespace {

constexpr int kScalarTileRows = 16;
constexpr int kScalarTileColumns = 16;
constexpr int kTileInner = 16;
constexpr int kTensorTileInner = 32;
constexpr int kTensorTileColumns = 64;
constexpr int kTensorSharedSkew = 8;

template <typename T>
__device__ float ToFloat(T value);

template <>
__device__ float ToFloat(float value) {
  return value;
}

template <>
__device__ float ToFloat(__half value) {
  return __half2float(value);
}

template <>
__device__ float ToFloat(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ T FromFloat(float value);

template <>
__device__ float FromFloat(float value) {
  return value;
}

template <>
__device__ __half FromFloat(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 FromFloat(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void GroupedGemmKernel(const T* lhs, const T* rhs,
                                  const int32_t* group_sizes, T* output,
                                  int64_t rows, int64_t inner,
                                  int64_t columns, int64_t groups) {
  __shared__ T lhs_tile[kScalarTileRows][kTileInner];
  __shared__ T rhs_tile[kTileInner][kScalarTileColumns];
  __shared__ int64_t group_index;
  __shared__ int64_t group_row_start;
  __shared__ int64_t group_rows;
  __shared__ int64_t local_row_start;

  int thread = threadIdx.y * blockDim.x + threadIdx.x;
  if (thread == 0) {
    int64_t row_start = 0;
    int64_t tile_start = 0;
    group_index = -1;
    for (int64_t group = 0; group < groups; ++group) {
      int64_t size = static_cast<int64_t>(group_sizes[group]);
      if (size < 0) size = 0;
      int64_t tiles = (size + kScalarTileRows - 1) / kScalarTileRows;
      if (static_cast<int64_t>(blockIdx.x) < tile_start + tiles) {
        group_index = group;
        group_row_start = row_start;
        group_rows = size;
        local_row_start =
            (static_cast<int64_t>(blockIdx.x) - tile_start) *
            kScalarTileRows;
        break;
      }
      row_start += size;
      tile_start += tiles;
    }
  }
  __syncthreads();

  if (group_index < 0) return;

  int64_t local_row = local_row_start + threadIdx.y;
  int64_t row = group_row_start + local_row;
  int64_t column =
      static_cast<int64_t>(blockIdx.y) * kScalarTileColumns + threadIdx.x;
  bool valid_row = local_row < group_rows && row < rows;
  float accumulator = 0.0f;

  for (int64_t tile_inner = 0; tile_inner < inner;
       tile_inner += kTileInner) {
    int64_t lhs_inner = tile_inner + threadIdx.x;
    lhs_tile[threadIdx.y][threadIdx.x] =
        valid_row && lhs_inner < inner
            ? lhs[row * inner + lhs_inner]
            : FromFloat<T>(0.0f);

    int64_t rhs_inner = tile_inner + threadIdx.y;
    rhs_tile[threadIdx.y][threadIdx.x] =
        rhs_inner < inner && column < columns
            ? rhs[(group_index * inner + rhs_inner) * columns + column]
            : FromFloat<T>(0.0f);
    __syncthreads();

#pragma unroll
    for (int offset = 0; offset < kTileInner; ++offset) {
      accumulator += ToFloat(lhs_tile[threadIdx.y][offset]) *
                     ToFloat(rhs_tile[offset][threadIdx.x]);
    }
    __syncthreads();
  }

  if (valid_row && column < columns) {
    output[row * columns + column] = FromFloat<T>(accumulator);
  }
}

template <typename T>
__device__ void StorePair(T* output, float first, float second);

template <>
__device__ void StorePair(__half* output, float first, float second) {
  *reinterpret_cast<__half2*>(output) =
      __floats2half2_rn(first, second);
}

template <>
__device__ void StorePair(__nv_bfloat16* output, float first,
                          float second) {
  *reinterpret_cast<__nv_bfloat162*>(output) =
      __floats2bfloat162_rn(first, second);
}

template <typename T, int TileRows>
struct alignas(32) TensorCoreInputStorage {
  static constexpr int kLhsStride =
      kTensorTileInner + kTensorSharedSkew;
  static constexpr int kRhsStride =
      kTensorTileColumns + kTensorSharedSkew;
  T lhs[2][TileRows][kLhsStride];
  T rhs[2][kTensorTileInner][kRhsStride];
};

template <typename T, int TileRows>
union alignas(32) TensorCoreSharedStorage {
  TensorCoreInputStorage<T, TileRows> input;
  float output[TileRows][kTensorTileColumns];
};

template <typename T>
__device__ void CopyAsync(T* destination, const T* source, bool valid) {
  uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(shared_address), "l"(source), "r"(valid ? 16 : 0)
               : "memory");
}

template <typename T, int TileRows>
__device__ void LoadTensorCoreStage(
    TensorCoreInputStorage<T, TileRows>* storage, int stage,
    const T* lhs, const T* group_rhs, int row_start, int local_rows,
    int inner, int columns, int column_start, int inner_start) {
  constexpr int kThreads = TileRows * 4;
  constexpr int kElementsPerCopy = 16 / sizeof(T);
  constexpr int kLhsChunks =
      TileRows * kTensorTileInner / kElementsPerCopy;
  constexpr int kRhsChunks =
      kTensorTileInner * kTensorTileColumns / kElementsPerCopy;
  for (int chunk = static_cast<int>(threadIdx.x);
       chunk < kLhsChunks + kRhsChunks; chunk += kThreads) {
    if (chunk < kLhsChunks) {
      int row = chunk / (kTensorTileInner / kElementsPerCopy);
      int column =
          (chunk % (kTensorTileInner / kElementsPerCopy)) *
          kElementsPerCopy;
      bool valid =
          row < local_rows && inner_start + column < inner;
      const T* source =
          valid
              ? lhs + (static_cast<int64_t>(row_start + row) * inner) +
                    inner_start + column
              : lhs;
      CopyAsync(&storage->lhs[stage][row][column], source, valid);
    } else {
      int rhs_chunk = chunk - kLhsChunks;
      int row =
          rhs_chunk / (kTensorTileColumns / kElementsPerCopy);
      int column =
          (rhs_chunk % (kTensorTileColumns / kElementsPerCopy)) *
          kElementsPerCopy;
      bool valid = inner_start + row < inner &&
                   column_start + column < columns;
      const T* source =
          valid
              ? group_rhs +
                    (static_cast<int64_t>(inner_start + row) * columns) +
                    column_start + column
              : group_rhs;
      CopyAsync(&storage->rhs[stage][row][column], source, valid);
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <typename T, int TileRows, bool ColumnFast>
__global__ void GroupedGemmTensorCoreKernel(
    const T* lhs, const T* rhs, const int32_t* group_sizes, T* output,
    int rows, int inner, int columns, int groups) {
  static_assert(TileRows == 32 || TileRows == 64);
  constexpr int kThreads = TileRows * 4;
  constexpr int kLhsStride =
      TensorCoreInputStorage<T, TileRows>::kLhsStride;
  constexpr int kRhsStride =
      TensorCoreInputStorage<T, TileRows>::kRhsStride;
  __shared__ TensorCoreSharedStorage<T, TileRows> shared;
  __shared__ int group_index;
  __shared__ int group_row_start;
  __shared__ int group_rows;
  __shared__ int local_row_start;

  int row_tile = static_cast<int>(
      ColumnFast ? blockIdx.y : blockIdx.x);
  if (groups < 32) {
    if (threadIdx.x == 0) {
      int row_offset = 0;
      int tile_offset = 0;
      group_index = -1;
      for (int group = 0; group < groups; ++group) {
        int size = group_sizes[group];
        if (size < 0) size = 0;
        if (size > rows - row_offset) size = rows - row_offset;
        int tiles = (size + TileRows - 1) / TileRows;
        if (row_tile < tile_offset + tiles) {
          group_index = group;
          group_row_start = row_offset;
          group_rows = size;
          local_row_start = (row_tile - tile_offset) * TileRows;
          break;
        }
        row_offset += size;
        tile_offset += tiles;
      }
    }
  } else if (threadIdx.x < 32) {
    constexpr unsigned kWarpMask = 0xffffffffu;
    int lane = static_cast<int>(threadIdx.x);
    int64_t row_base = 0;
    int64_t tile_base = 0;
    if (lane == 0) group_index = -1;
    for (int first_group = 0; first_group < groups;
         first_group += 32) {
      int group = first_group + lane;
      int64_t size =
          group < groups ? static_cast<int64_t>(group_sizes[group]) : 0;
      if (size < 0) size = 0;
      if (size > rows) size = rows;
      int64_t tiles = (size + TileRows - 1) / TileRows;
      int64_t row_prefix = size;
      int64_t tile_prefix = tiles;
#pragma unroll
      for (int offset = 1; offset < 32; offset *= 2) {
        int64_t previous_rows =
            __shfl_up_sync(kWarpMask, row_prefix, offset);
        int64_t previous_tiles =
            __shfl_up_sync(kWarpMask, tile_prefix, offset);
        if (lane >= offset) {
          row_prefix += previous_rows;
          tile_prefix += previous_tiles;
        }
      }
      int64_t tile_begin = tile_base + tile_prefix - tiles;
      bool owns_tile =
          group < groups && row_tile >= tile_begin &&
          row_tile < tile_begin + tiles;
      unsigned owners = __ballot_sync(kWarpMask, owns_tile);
      if (owners != 0) {
        int owner = __ffs(owners) - 1;
        if (lane == owner) {
          int64_t row_begin = row_base + row_prefix - size;
          if (row_begin < rows) {
            int64_t remaining = rows - row_begin;
            if (size > remaining) size = remaining;
            group_index = group;
            group_row_start = static_cast<int>(row_begin);
            group_rows = static_cast<int>(size);
            local_row_start =
                static_cast<int>(row_tile - tile_begin) * TileRows;
          }
        }
        break;
      }
      row_base += __shfl_sync(kWarpMask, row_prefix, 31);
      tile_base += __shfl_sync(kWarpMask, tile_prefix, 31);
    }
  }
  __syncthreads();
  if (group_index < 0) return;

  int remaining_rows = group_rows - local_row_start;
  int local_rows =
      remaining_rows < TileRows ? remaining_rows : TileRows;
  int row_start = group_row_start + local_row_start;
  int column_tile = static_cast<int>(
      ColumnFast ? blockIdx.x : blockIdx.y);
  int column_start = column_tile * kTensorTileColumns;
  const T* group_rhs =
      rhs + static_cast<int64_t>(group_index) * inner * columns;
  auto* storage = &shared.input;

  using namespace nvcuda;
  int warp = static_cast<int>(threadIdx.x) / 32;
  int warp_row = warp / 2;
  int warp_column = warp % 2;
  bool active_warp =
      TileRows != 32 || ColumnFast || warp_row * 16 < local_rows;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      accumulators[2];
  if (active_warp) {
    wmma::fill_fragment(accumulators[0], 0.0f);
    wmma::fill_fragment(accumulators[1], 0.0f);
  }

  LoadTensorCoreStage(storage, 0, lhs, group_rhs, row_start,
                      local_rows, inner, columns, column_start, 0);
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();

  int stage = 0;
  int inner_tiles =
      1 + (inner - 1) / kTensorTileInner;
  for (int tile = 0; tile < inner_tiles; ++tile) {
    int next_stage = stage ^ 1;
    if (tile + 1 < inner_tiles) {
      LoadTensorCoreStage(
          storage, next_stage, lhs, group_rhs, row_start, local_rows,
          inner, columns, column_start,
          (tile + 1) * kTensorTileInner);
    }

    if (active_warp) {
#pragma unroll
      for (int offset = 0; offset < kTensorTileInner;
           offset += kTileInner) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, T,
                       wmma::row_major>
            lhs_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T,
                       wmma::row_major>
            rhs_fragment[2];
        wmma::load_matrix_sync(
            lhs_fragment,
            &storage->lhs[stage][warp_row * 16][offset], kLhsStride);
#pragma unroll
        for (int fragment = 0; fragment < 2; ++fragment) {
          wmma::load_matrix_sync(
              rhs_fragment[fragment],
              &storage->rhs[stage][offset]
                           [warp_column * 32 + fragment * 16],
              kRhsStride);
          wmma::mma_sync(accumulators[fragment], lhs_fragment,
                         rhs_fragment[fragment],
                         accumulators[fragment]);
        }
      }
    }

    if (tile + 1 < inner_tiles) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }
    stage = next_stage;
  }

  __syncthreads();
  if (active_warp) {
#pragma unroll
    for (int fragment = 0; fragment < 2; ++fragment) {
      wmma::store_matrix_sync(
          &shared.output[warp_row * 16]
                        [warp_column * 32 + fragment * 16],
          accumulators[fragment], kTensorTileColumns,
          wmma::mem_row_major);
    }
  }
  __syncthreads();

  for (int index = static_cast<int>(threadIdx.x);
       index < TileRows * kTensorTileColumns / 2;
       index += kThreads) {
    int row = index / (kTensorTileColumns / 2);
    int column = index % (kTensorTileColumns / 2) * 2;
    if (row < local_rows && column_start + column < columns) {
      StorePair(
          output +
              (static_cast<int64_t>(row_start + row) * columns) +
              column_start + column,
          shared.output[row][column],
          shared.output[row][column + 1]);
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
  extension->metadata->traits =
      XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
  extension->metadata->state_type_id = XLA_FFI_UNKNOWN_TYPE_ID;
  return nullptr;
}

XLA_FFI_Error* ValidateCall(XLA_FFI_CallFrame* call_frame) {
  if (call_frame->struct_size < XLA_FFI_CallFrame_STRUCT_SIZE) {
    return InvalidArgument(call_frame, "invalid XLA FFI call frame");
  }
  if (call_frame->extension_start != nullptr &&
      call_frame->extension_start->type == XLA_FFI_Extension_Metadata) {
    return PopulateMetadata(call_frame);
  }
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return InvalidArgument(call_frame, "grouped GEMM requires execute stage");
  }
  if (call_frame->args.size != 3 || call_frame->rets.size != 1) {
    return InvalidArgument(call_frame, "grouped GEMM has an invalid arity");
  }
  for (int64_t index = 0; index < 3; ++index) {
    if (call_frame->args.types[index] != XLA_FFI_ArgType_BUFFER) {
      return InvalidArgument(call_frame,
                             "grouped GEMM arguments must be buffers");
    }
  }
  if (call_frame->rets.types[0] != XLA_FFI_RetType_BUFFER) {
    return InvalidArgument(call_frame,
                           "grouped GEMM result must be a buffer");
  }
  return nullptr;
}

bool IsFloatType(XLA_FFI_DataType dtype) {
  return dtype == XLA_FFI_DataType_F16 || dtype == XLA_FFI_DataType_BF16 ||
         dtype == XLA_FFI_DataType_F32;
}

XLA_FFI_Error* ValidateBuffers(XLA_FFI_CallFrame* call_frame,
                               const XLA_FFI_Buffer* lhs,
                               const XLA_FFI_Buffer* rhs,
                               const XLA_FFI_Buffer* group_sizes,
                               const XLA_FFI_Buffer* output) {
  const XLA_FFI_Buffer* buffers[] = {lhs, rhs, group_sizes, output};
  for (const XLA_FFI_Buffer* buffer : buffers) {
    if (buffer == nullptr ||
        buffer->struct_size < XLA_FFI_Buffer_STRUCT_SIZE ||
        buffer->dims == nullptr) {
      return InvalidArgument(call_frame, "invalid grouped GEMM buffer");
    }
  }
  if (lhs->rank != 2 || rhs->rank != 3 || group_sizes->rank != 1 ||
      output->rank != 2) {
    return InvalidArgument(
        call_frame,
        "grouped GEMM expects lhs[rows,k], rhs[groups,k,n], "
        "group_sizes[groups], and output[rows,n]");
  }
  if (!IsFloatType(lhs->dtype) || rhs->dtype != lhs->dtype ||
      output->dtype != lhs->dtype) {
    return InvalidArgument(
        call_frame,
        "grouped GEMM requires matching f16, bf16, or f32 data buffers");
  }
  if (group_sizes->dtype != XLA_FFI_DataType_S32) {
    return InvalidArgument(call_frame,
                           "grouped GEMM group sizes must be int32");
  }

  int64_t rows = lhs->dims[0];
  int64_t inner = lhs->dims[1];
  int64_t groups = rhs->dims[0];
  int64_t columns = rhs->dims[2];
  if (rows < 0 || inner <= 0 || groups <= 0 || columns <= 0) {
    return InvalidArgument(call_frame,
                           "grouped GEMM has invalid dimensions");
  }
  if (rhs->dims[1] != inner || group_sizes->dims[0] != groups ||
      output->dims[0] != rows || output->dims[1] != columns) {
    return InvalidArgument(call_frame,
                           "grouped GEMM buffer shapes are inconsistent");
  }
  if (rows > std::numeric_limits<int>::max() ||
      inner > std::numeric_limits<int>::max() ||
      columns > std::numeric_limits<int>::max() ||
      groups > std::numeric_limits<int>::max()) {
    return InvalidArgument(call_frame,
                           "grouped GEMM dimensions exceed CUDA int limits");
  }
  int64_t column_tiles =
      (columns + kScalarTileColumns - 1) / kScalarTileColumns;
  if (column_tiles > 65535) {
    return InvalidArgument(
        call_frame,
        "grouped GEMM output is too wide for the CUDA launch grid");
  }
  int64_t row_tiles =
      (rows + kScalarTileRows - 1) / kScalarTileRows + groups - 1;
  if (row_tiles > std::numeric_limits<int>::max()) {
    return InvalidArgument(
        call_frame,
        "grouped GEMM has too many row groups for a CUDA launch");
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

template <typename T>
void LaunchScalar(const XLA_FFI_Buffer* lhs, const XLA_FFI_Buffer* rhs,
                  const XLA_FFI_Buffer* group_sizes, XLA_FFI_Buffer* output,
                  cudaStream_t stream) {
  int rows = static_cast<int>(lhs->dims[0]);
  int inner = static_cast<int>(lhs->dims[1]);
  int groups = static_cast<int>(rhs->dims[0]);
  int columns = static_cast<int>(rhs->dims[2]);
  int64_t row_tiles =
      (static_cast<int64_t>(rows) + kScalarTileRows - 1) /
          kScalarTileRows +
      groups - 1;
  if (row_tiles > rows) row_tiles = rows;
  dim3 threads(kScalarTileColumns, kScalarTileRows);
  dim3 blocks(static_cast<unsigned int>(row_tiles),
              static_cast<unsigned int>(
                  (columns + kScalarTileColumns - 1) /
                  kScalarTileColumns));
  GroupedGemmKernel<T><<<blocks, threads, 0, stream>>>(
      static_cast<const T*>(lhs->data), static_cast<const T*>(rhs->data),
      static_cast<const int32_t*>(group_sizes->data),
      static_cast<T*>(output->data), rows, inner, columns, groups);
}

template <typename T, int TileRows, bool ColumnFast>
void LaunchTensorCore(const XLA_FFI_Buffer* lhs,
                      const XLA_FFI_Buffer* rhs,
                      const XLA_FFI_Buffer* group_sizes,
                      XLA_FFI_Buffer* output, cudaStream_t stream) {
  int rows = static_cast<int>(lhs->dims[0]);
  int inner = static_cast<int>(lhs->dims[1]);
  int groups = static_cast<int>(rhs->dims[0]);
  int columns = static_cast<int>(rhs->dims[2]);
  int64_t row_tiles =
      (static_cast<int64_t>(rows) + TileRows - 1) / TileRows + groups - 1;
  if (row_tiles > rows) row_tiles = rows;
  int column_tiles =
      (columns + kTensorTileColumns - 1) / kTensorTileColumns;
  dim3 blocks =
      ColumnFast
          ? dim3(static_cast<unsigned int>(column_tiles),
                 static_cast<unsigned int>(row_tiles))
          : dim3(static_cast<unsigned int>(row_tiles),
                 static_cast<unsigned int>(column_tiles));
  GroupedGemmTensorCoreKernel<T, TileRows, ColumnFast>
      <<<blocks, TileRows * 4, 0, stream>>>(
      static_cast<const T*>(lhs->data), static_cast<const T*>(rhs->data),
      static_cast<const int32_t*>(group_sizes->data),
      static_cast<T*>(output->data), rows, inner, columns, groups);
}

template <typename T>
void LaunchTensorCore(const XLA_FFI_Buffer* lhs,
                      const XLA_FFI_Buffer* rhs,
                      const XLA_FFI_Buffer* group_sizes,
                      XLA_FFI_Buffer* output, cudaStream_t stream) {
  int64_t average_rows = lhs->dims[0] / rhs->dims[0];
  int64_t row_tiles =
      (lhs->dims[0] + 31) / 32 + rhs->dims[0] - 1;
  if (row_tiles > lhs->dims[0]) row_tiles = lhs->dims[0];
  bool column_fast = average_rows >= 16 && row_tiles <= 65535;
  if (average_rows >= 33) {
    LaunchTensorCore<T, 64, false>(lhs, rhs, group_sizes, output, stream);
  } else if (column_fast) {
    LaunchTensorCore<T, 32, true>(lhs, rhs, group_sizes, output, stream);
  } else {
    LaunchTensorCore<T, 32, false>(lhs, rhs, group_sizes, output, stream);
  }
}

XLA_FFI_Error* GroupedGemmForward(XLA_FFI_CallFrame* call_frame) {
  bool metadata = call_frame->extension_start != nullptr &&
                  call_frame->extension_start->type ==
                      XLA_FFI_Extension_Metadata;
  XLA_FFI_Error* error = ValidateCall(call_frame);
  if (error != nullptr || metadata) return error;

  auto* lhs =
      reinterpret_cast<XLA_FFI_Buffer*>(call_frame->args.args[0]);
  auto* rhs =
      reinterpret_cast<XLA_FFI_Buffer*>(call_frame->args.args[1]);
  auto* group_sizes =
      reinterpret_cast<XLA_FFI_Buffer*>(call_frame->args.args[2]);
  auto* output =
      reinterpret_cast<XLA_FFI_Buffer*>(call_frame->rets.rets[0]);
  if ((error = ValidateBuffers(call_frame, lhs, rhs, group_sizes, output)) !=
      nullptr) {
    return error;
  }
  if (lhs->dims[0] == 0) return nullptr;

  cudaStream_t stream = Stream(call_frame, &error);
  if (error != nullptr) return error;
  switch (lhs->dtype) {
    case XLA_FFI_DataType_F16:
      if (lhs->dims[1] % 8 == 0 && rhs->dims[2] % 8 == 0) {
        LaunchTensorCore<__half>(lhs, rhs, group_sizes, output, stream);
      } else {
        LaunchScalar<__half>(lhs, rhs, group_sizes, output, stream);
      }
      break;
    case XLA_FFI_DataType_BF16:
      if (lhs->dims[1] % 8 == 0 && rhs->dims[2] % 8 == 0) {
        LaunchTensorCore<__nv_bfloat16>(lhs, rhs, group_sizes, output,
                                       stream);
      } else {
        LaunchScalar<__nv_bfloat16>(lhs, rhs, group_sizes, output, stream);
      }
      break;
    case XLA_FFI_DataType_F32:
      LaunchScalar<float>(lhs, rhs, group_sizes, output, stream);
      break;
    default:
      return InvalidArgument(call_frame,
                             "grouped GEMM received an unsupported dtype");
  }
  cudaError_t cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return Error(call_frame, XLA_FFI_Error_Code_INTERNAL,
                 cudaGetErrorString(cuda_error));
  }
  return nullptr;
}

}  // namespace

extern "C" XLA_FFI_Error* raven_grouped_gemm_fwd(
    XLA_FFI_CallFrame* call_frame) {
  return GroupedGemmForward(call_frame);
}
