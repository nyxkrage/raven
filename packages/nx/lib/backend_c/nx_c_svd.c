/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <complex.h>
#include <float.h>
#include <lapacke.h>

#include "nx_c_shared.h"

// Helper functions for shape and stride operations
static inline int nx_ndim(value v_shape) { return Wosize_val(v_shape); }

static inline int nx_shape_at(value v_shape, int idx) {
  return Int_val(Field(v_shape, idx));
}

static inline int nx_stride_at(value v_strides, int idx) {
  return Int_val(Field(v_strides, idx));
}

static inline int nx_batch_size(value v_shape) {
  int ndim = Wosize_val(v_shape);
  if (ndim <= 2) return 1;
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) {
    batch_size *= Int_val(Field(v_shape, i));
  }
  return batch_size;
}

static inline size_t nx_batch_offset_elems(int b, value v_shape,
                                           value v_strides) {
  int ndim = Wosize_val(v_shape);
  if (ndim <= 2) return 0;
  size_t offset = 0;
  int remaining = b;
  // Calculate offset for batch dimensions
  for (int i = ndim - 3; i >= 0; i--) {
    int dim_size = Int_val(Field(v_shape, i));
    int coord = remaining % dim_size;
    remaining /= dim_size;
    offset += coord * Int_val(Field(v_strides, i));
  }
  return offset;
}

// Helper functions for packing/unpacking matrices
static void nx_pack_f32(float* dst, const float* src, int m, int n,
                        int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * n + j] = src[i * stride_row + j * stride_col];
    }
  }
}

static void nx_unpack_f32(float* dst, const float* src, int m, int n,
                          int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * stride_row + j * stride_col] = src[i * n + j];
    }
  }
}

static void nx_pack_f64(double* dst, const double* src, int m, int n,
                        int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * n + j] = src[i * stride_row + j * stride_col];
    }
  }
}

static void nx_unpack_f64(double* dst, const double* src, int m, int n,
                          int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * stride_row + j * stride_col] = src[i * n + j];
    }
  }
}

static void nx_pack_c32(complex32* dst, const complex32* src, int m, int n,
                        int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * n + j] = src[i * stride_row + j * stride_col];
    }
  }
}

static void nx_unpack_c32(complex32* dst, const complex32* src, int m, int n,
                          int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * stride_row + j * stride_col] = src[i * n + j];
    }
  }
}

static void nx_pack_c64(complex64* dst, const complex64* src, int m, int n,
                        int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * n + j] = src[i * stride_row + j * stride_col];
    }
  }
}

static void nx_unpack_c64(complex64* dst, const complex64* src, int m, int n,
                          int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * stride_row + j * stride_col] = src[i * n + j];
    }
  }
}

// SVD implementations
static lapack_int svd_float32(float* a, float* u, float* s, float* vt, int m,
                              int n, int full_matrices) {
  // LAPACK destroys the input matrix, so we need to make a copy
  float* a_copy = (float*)malloc(m * n * sizeof(float));
  if (!a_copy) return -1010;
  memcpy(a_copy, a, m * n * sizeof(float));

  char jobu = full_matrices ? 'A' : 'S';
  char jobvt = full_matrices ? 'A' : 'S';
  int minmn = m < n ? m : n;
  // ldu: U is [m, m] (full) or [m, minmn] (econ), leading dim is # cols
  lapack_int ldu = full_matrices ? m : minmn;
  // ldvt: VT is [n, n] (full) or [minmn, n] (econ), leading dim is # cols = n
  lapack_int ldvt = n;

  // Allocate space for superbidiagonal elements (not used in our interface)
  float* superb = (float*)malloc((minmn - 1) * sizeof(float));
  if (!superb) {
    free(a_copy);
    return -1010;
  }

  lapack_int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a_copy, n, s, u, ldu, vt, ldvt, superb);
  free(a_copy);
  free(superb);
  // Note: LAPACK returns singular values in descending order, which matches our expectation
  return info;
}

static lapack_int svd_float64(double* a, double* u, double* s, double* vt,
                              int m, int n, int full_matrices) {
  // LAPACK destroys the input matrix, so we need to make a copy
  double* a_copy = (double*)malloc(m * n * sizeof(double));
  if (!a_copy) return -1010;
  memcpy(a_copy, a, m * n * sizeof(double));

  char jobu = full_matrices ? 'A' : 'S';
  char jobvt = full_matrices ? 'A' : 'S';
  int minmn = m < n ? m : n;
  // ldu: U is [m, m] (full) or [m, minmn] (econ), leading dim is # cols
  lapack_int ldu = full_matrices ? m : minmn;
  // ldvt: VT is [n, n] (full) or [minmn, n] (econ), leading dim is # cols = n
  lapack_int ldvt = n;

  // Allocate space for superbidiagonal elements (not used in our interface)
  double* superb = (double*)malloc((minmn - 1) * sizeof(double));
  if (!superb) {
    free(a_copy);
    return -1010;
  }

  lapack_int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a_copy, n, s, u, ldu, vt, ldvt, superb);
  free(a_copy);
  free(superb);
  // Note: LAPACK returns singular values in descending order, which matches our expectation
  return info;
}

static lapack_int svd_complex32(complex32* a, complex32* u, float* s,
                                complex32* vt, int m, int n,
                                int full_matrices) {
  // LAPACK destroys the input matrix, so we need to make a copy
  complex32* a_copy = (complex32*)malloc(m * n * sizeof(complex32));
  if (!a_copy) return -1010;
  memcpy(a_copy, a, m * n * sizeof(complex32));

  char jobu = full_matrices ? 'A' : 'S';
  char jobvt = full_matrices ? 'A' : 'S';
  int minmn = m < n ? m : n;
  lapack_int ldu = full_matrices ? m : minmn;
  lapack_int ldvt = full_matrices ? n : minmn;

  // Allocate space for superbidiagonal elements (not used in our interface)
  float* superb = (float*)malloc((minmn - 1) * sizeof(float));
  if (!superb) {
    free(a_copy);
    return -1010;
  }

  lapack_int info = LAPACKE_cgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a_copy, n, s, u, ldu, vt, ldvt, superb);
  free(a_copy);
  free(superb);
  if (info != 0) return info;
  // Note: LAPACK returns singular values in descending order, which matches our expectation
  // Note: For complex SVD, LAPACK returns V^H (conjugate transpose), but our interface expects V^T
  // We need to conjugate the result to match our expected output
  if (full_matrices) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = conj(vt[i * n + j]);
      }
    }
  } else {
    for (int i = 0; i < minmn; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = conj(vt[i * n + j]);
      }
    }
  }
  return 0;
}

static lapack_int svd_complex64(complex64* a, complex64* u, double* s,
                                complex64* vt, int m, int n,
                                int full_matrices) {
  // LAPACK destroys the input matrix, so we need to make a copy
  complex64* a_copy = (complex64*)malloc(m * n * sizeof(complex64));
  if (!a_copy) return -1010;
  memcpy(a_copy, a, m * n * sizeof(complex64));

  char jobu = full_matrices ? 'A' : 'S';
  char jobvt = full_matrices ? 'A' : 'S';
  int minmn = m < n ? m : n;
  lapack_int ldu = full_matrices ? m : minmn;
  lapack_int ldvt = full_matrices ? n : minmn;

  // Allocate space for superbidiagonal elements (not used in our interface)
  double* superb = (double*)malloc((minmn - 1) * sizeof(double));
  if (!superb) {
    free(a_copy);
    return -1010;
  }

  lapack_int info = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a_copy, n, s, u, ldu, vt, ldvt, superb);
  free(a_copy);
  free(superb);
  if (info != 0) return info;
  // Note: LAPACK returns singular values in descending order, which matches our expectation
  // Note: For complex SVD, LAPACK returns V^H (conjugate transpose), but our interface expects V^T
  // We need to conjugate the result to match our expected output
  if (full_matrices) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = conj(vt[i * n + j]);
      }
    }
  } else {
    for (int i = 0; i < minmn; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = conj(vt[i * n + j]);
      }
    }
  }
  return 0;
}

static lapack_int svd_float16(uint16_t* a, uint16_t* u, uint16_t* s,
                              uint16_t* vt, int m, int n,
                              int full_matrices) {
  int minmn = m < n ? m : n;
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int u_cols = full_matrices ? m : minmn;
  float* u_float = (float*)malloc(m * u_cols * sizeof(float));
  float* s_float = (float*)malloc(minmn * sizeof(float));
  int vt_rows = full_matrices ? n : minmn;
  float* vt_float = (float*)malloc(vt_rows * n * sizeof(float));
  if (!a_float || !u_float || !s_float || !vt_float) {
    free(a_float);
    free(u_float);
    free(s_float);
    free(vt_float);
    return -1010;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = half_to_float(a[i]);
  lapack_int info =
      svd_float32(a_float, u_float, s_float, vt_float, m, n, full_matrices);
  if (info == 0) {
    for (int i = 0; i < m * u_cols; i++) u[i] = float_to_half(u_float[i]);
    for (int i = 0; i < minmn; i++) s[i] = float_to_half(s_float[i]);
    for (int i = 0; i < vt_rows * n; i++) vt[i] = float_to_half(vt_float[i]);
  }
  free(a_float);
  free(u_float);
  free(s_float);
  free(vt_float);
  return info;
}

static lapack_int svd_bfloat16(caml_ba_bfloat16* a, caml_ba_bfloat16* u,
                               caml_ba_bfloat16* s, caml_ba_bfloat16* vt,
                               int m, int n, int full_matrices) {
  int minmn = m < n ? m : n;
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int u_cols = full_matrices ? m : minmn;
  float* u_float = (float*)malloc(m * u_cols * sizeof(float));
  float* s_float = (float*)malloc(minmn * sizeof(float));
  int vt_rows = full_matrices ? n : minmn;
  float* vt_float = (float*)malloc(vt_rows * n * sizeof(float));
  if (!a_float || !u_float || !s_float || !vt_float) {
    free(a_float);
    free(u_float);
    free(s_float);
    free(vt_float);
    return -1010;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = bfloat16_to_float(a[i]);
  lapack_int info =
      svd_float32(a_float, u_float, s_float, vt_float, m, n, full_matrices);
  if (info == 0) {
    for (int i = 0; i < m * u_cols; i++)
      u[i] = float_to_bfloat16(u_float[i]);
    for (int i = 0; i < minmn; i++) s[i] = float_to_bfloat16(s_float[i]);
    for (int i = 0; i < vt_rows * n; i++)
      vt[i] = float_to_bfloat16(vt_float[i]);
  }
  free(a_float);
  free(u_float);
  free(s_float);
  free(vt_float);
  return info;
}

static lapack_int svd_f8e4m3(caml_ba_fp8_e4m3* a, caml_ba_fp8_e4m3* u,
                             caml_ba_fp8_e4m3* s, caml_ba_fp8_e4m3* vt, int m,
                             int n, int full_matrices) {
  int minmn = m < n ? m : n;
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int u_cols = full_matrices ? m : minmn;
  float* u_float = (float*)malloc(m * u_cols * sizeof(float));
  float* s_float = (float*)malloc(minmn * sizeof(float));
  int vt_rows = full_matrices ? n : minmn;
  float* vt_float = (float*)malloc(vt_rows * n * sizeof(float));
  if (!a_float || !u_float || !s_float || !vt_float) {
    free(a_float);
    free(u_float);
    free(s_float);
    free(vt_float);
    return -1010;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = fp8_e4m3_to_float(a[i]);
  lapack_int info =
      svd_float32(a_float, u_float, s_float, vt_float, m, n, full_matrices);
  if (info == 0) {
    for (int i = 0; i < m * u_cols; i++) u[i] = float_to_fp8_e4m3(u_float[i]);
    for (int i = 0; i < minmn; i++) s[i] = float_to_fp8_e4m3(s_float[i]);
    for (int i = 0; i < vt_rows * n; i++)
      vt[i] = float_to_fp8_e4m3(vt_float[i]);
  }
  free(a_float);
  free(u_float);
  free(s_float);
  free(vt_float);
  return info;
}

static lapack_int svd_f8e5m2(caml_ba_fp8_e5m2* a, caml_ba_fp8_e5m2* u,
                             caml_ba_fp8_e5m2* s, caml_ba_fp8_e5m2* vt, int m,
                             int n, int full_matrices) {
  int minmn = m < n ? m : n;
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int u_cols = full_matrices ? m : minmn;
  float* u_float = (float*)malloc(m * u_cols * sizeof(float));
  float* s_float = (float*)malloc(minmn * sizeof(float));
  int vt_rows = full_matrices ? n : minmn;
  float* vt_float = (float*)malloc(vt_rows * n * sizeof(float));
  if (!a_float || !u_float || !s_float || !vt_float) {
    free(a_float);
    free(u_float);
    free(s_float);
    free(vt_float);
    return -1010;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = fp8_e5m2_to_float(a[i]);
  lapack_int info =
      svd_float32(a_float, u_float, s_float, vt_float, m, n, full_matrices);
  if (info == 0) {
    for (int i = 0; i < m * u_cols; i++) u[i] = float_to_fp8_e5m2(u_float[i]);
    for (int i = 0; i < minmn; i++) s[i] = float_to_fp8_e5m2(s_float[i]);
    for (int i = 0; i < vt_rows * n; i++)
      vt[i] = float_to_fp8_e5m2(vt_float[i]);
  }
  free(a_float);
  free(u_float);
  free(s_float);
  free(vt_float);
  return info;
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

CAMLprim value caml_nx_op_svd(value v_in, value v_u, value v_s, value v_vt,
                              value v_full_matrices) {
  CAMLparam5(v_in, v_u, v_s, v_vt, v_full_matrices);
  int full_matrices = Int_val(v_full_matrices);
  ndarray_t in = extract_ndarray(v_in);
  ndarray_t u_nd = extract_ndarray(v_u);
  ndarray_t s_nd = extract_ndarray(v_s);
  ndarray_t vt_nd = extract_ndarray(v_vt);
  struct caml_ba_array* ba_in = Caml_ba_array_val(Field(v_in, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_u = Caml_ba_array_val(Field(v_u, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_s = Caml_ba_array_val(Field(v_s, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_vt = Caml_ba_array_val(Field(v_vt, FFI_TENSOR_DATA));
  int kind = nx_buffer_get_kind(ba_in);
  if (in.ndim < 2) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&u_nd);
    cleanup_ndarray(&s_nd);
    cleanup_ndarray(&vt_nd);
    caml_failwith("svd: input must have at least 2 dimensions");
  }
  int m = in.shape[in.ndim - 2];
  int n = in.shape[in.ndim - 1];
  int minmn = m < n ? m : n;
  int u_cols = full_matrices ? m : minmn;
  int vt_rows = full_matrices ? n : minmn;
  if (u_nd.shape[u_nd.ndim - 1] != u_cols || u_nd.shape[u_nd.ndim - 2] != m ||
      vt_nd.shape[vt_nd.ndim - 1] != n ||
      vt_nd.shape[vt_nd.ndim - 2] != vt_rows ||
      s_nd.shape[s_nd.ndim - 1] != minmn) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&u_nd);
    cleanup_ndarray(&s_nd);
    cleanup_ndarray(&vt_nd);
    caml_failwith("svd: output shapes mismatch");
  }
  int batch_size = 1;
  for (int i = 0; i < in.ndim - 2; i++) {
    batch_size *= in.shape[i];
  }
  int s_in_row = in.strides[in.ndim - 2];
  int s_in_col = in.strides[in.ndim - 1];
  int s_u_row = u_nd.strides[u_nd.ndim - 2];
  int s_u_col = u_nd.strides[u_nd.ndim - 1];
  int s_s_stride = s_nd.strides[s_nd.ndim - 1];
  int s_vt_row = vt_nd.strides[vt_nd.ndim - 2];
  int s_vt_col = vt_nd.strides[vt_nd.ndim - 1];
  caml_enter_blocking_section();
  for (int b = 0; b < batch_size; b++) {
    size_t off_in = in.offset;
    size_t off_u = u_nd.offset;
    size_t off_s = s_nd.offset;
    size_t off_vt = vt_nd.offset;
    if (in.ndim > 2) {
      int remaining = b;
      for (int i = in.ndim - 3; i >= 0; i--) {
        int coord = remaining % in.shape[i];
        remaining /= in.shape[i];
        off_in += coord * in.strides[i];
        off_u += coord * u_nd.strides[i];
        off_s += coord * s_nd.strides[i];
        off_vt += coord * vt_nd.strides[i];
      }
    }
    switch (kind) {
      case CAML_BA_FLOAT32: {
        float* base_in = (float*)ba_in->data + off_in;
        float* base_u = (float*)ba_u->data + off_u;
        double* base_s = (double*)ba_s->data + off_s;  // S is always float64
        float* base_vt = (float*)ba_vt->data + off_vt;
        float* A = (float*)malloc((size_t)m * n * sizeof(float));
        float* U = (float*)malloc((size_t)m * u_cols * sizeof(float));
        float* S = (float*)malloc((size_t)minmn * sizeof(float));
        float* VT = (float*)malloc((size_t)vt_rows * n * sizeof(float));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        nx_pack_f32(A, base_in, m, n, s_in_row, s_in_col);
        svd_float32(A, U, S, VT, m, n, full_matrices);
        nx_unpack_f32(base_u, U, m, u_cols, s_u_row, s_u_col);
        // Convert S from float32 to float64
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = (double)S[i];
        nx_unpack_f32(base_vt, VT, vt_rows, n, s_vt_row, s_vt_col);
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case CAML_BA_FLOAT64: {
        double* base_in = (double*)ba_in->data + off_in;
        double* base_u = (double*)ba_u->data + off_u;
        double* base_s = (double*)ba_s->data + off_s;
        double* base_vt = (double*)ba_vt->data + off_vt;
        double* A = (double*)malloc((size_t)m * n * sizeof(double));
        double* U = (double*)malloc((size_t)m * u_cols * sizeof(double));
        double* S = (double*)malloc((size_t)minmn * sizeof(double));
        double* VT = (double*)malloc((size_t)vt_rows * n * sizeof(double));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        nx_pack_f64(A, base_in, m, n, s_in_row, s_in_col);
        svd_float64(A, U, S, VT, m, n, full_matrices);
        nx_unpack_f64(base_u, U, m, u_cols, s_u_row, s_u_col);
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        nx_unpack_f64(base_vt, VT, vt_rows, n, s_vt_row, s_vt_col);
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case CAML_BA_COMPLEX32: {
        complex32* base_in = (complex32*)ba_in->data + off_in;
        complex32* base_u = (complex32*)ba_u->data + off_u;
        double* base_s = (double*)ba_s->data + off_s;  // S is always float64
        complex32* base_vt = (complex32*)ba_vt->data + off_vt;
        complex32* A = (complex32*)malloc((size_t)m * n * sizeof(complex32));
        complex32* U =
            (complex32*)malloc((size_t)m * u_cols * sizeof(complex32));
        float* S = (float*)malloc((size_t)minmn * sizeof(float));
        complex32* VT =
            (complex32*)malloc((size_t)vt_rows * n * sizeof(complex32));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        nx_pack_c32(A, base_in, m, n, s_in_row, s_in_col);
        svd_complex32(A, U, S, VT, m, n, full_matrices);
        nx_unpack_c32(base_u, U, m, u_cols, s_u_row, s_u_col);
        // Convert S from float32 to float64
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = (double)S[i];
        nx_unpack_c32(base_vt, VT, vt_rows, n, s_vt_row, s_vt_col);
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case CAML_BA_COMPLEX64: {
        complex64* base_in = (complex64*)ba_in->data + off_in;
        complex64* base_u = (complex64*)ba_u->data + off_u;
        double* base_s = (double*)ba_s->data + off_s;
        complex64* base_vt = (complex64*)ba_vt->data + off_vt;
        complex64* A = (complex64*)malloc((size_t)m * n * sizeof(complex64));
        complex64* U =
            (complex64*)malloc((size_t)m * u_cols * sizeof(complex64));
        double* S = (double*)malloc((size_t)minmn * sizeof(double));
        complex64* VT =
            (complex64*)malloc((size_t)vt_rows * n * sizeof(complex64));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        nx_pack_c64(A, base_in, m, n, s_in_row, s_in_col);
        svd_complex64(A, U, S, VT, m, n, full_matrices);
        nx_unpack_c64(base_u, U, m, u_cols, s_u_row, s_u_col);
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        nx_unpack_c64(base_vt, VT, vt_rows, n, s_vt_row, s_vt_col);
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case CAML_BA_FLOAT16: {
        uint16_t* base_in = (uint16_t*)ba_in->data + off_in;
        uint16_t* base_u = (uint16_t*)ba_u->data + off_u;
        uint16_t* base_s = (uint16_t*)ba_s->data + off_s;
        uint16_t* base_vt = (uint16_t*)ba_vt->data + off_vt;
        uint16_t* A = (uint16_t*)malloc((size_t)m * n * sizeof(uint16_t));
        uint16_t* U = (uint16_t*)malloc((size_t)m * u_cols * sizeof(uint16_t));
        uint16_t* S = (uint16_t*)malloc((size_t)minmn * sizeof(uint16_t));
        uint16_t* VT =
            (uint16_t*)malloc((size_t)vt_rows * n * sizeof(uint16_t));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        svd_float16(A, U, S, VT, m, n, full_matrices);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < u_cols; j++) {
            base_u[i * s_u_row + j * s_u_col] = U[i * u_cols + j];
          }
        }
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        for (int i = 0; i < vt_rows; i++) {
          for (int j = 0; j < n; j++) {
            base_vt[i * s_vt_row + j * s_vt_col] = VT[i * n + j];
          }
        }
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case NX_BA_BFLOAT16: {
        caml_ba_bfloat16* base_in = (caml_ba_bfloat16*)ba_in->data + off_in;
        caml_ba_bfloat16* base_u = (caml_ba_bfloat16*)ba_u->data + off_u;
        caml_ba_bfloat16* base_s = (caml_ba_bfloat16*)ba_s->data + off_s;
        caml_ba_bfloat16* base_vt = (caml_ba_bfloat16*)ba_vt->data + off_vt;
        caml_ba_bfloat16* A =
            (caml_ba_bfloat16*)malloc((size_t)m * n * sizeof(caml_ba_bfloat16));
        caml_ba_bfloat16* U = (caml_ba_bfloat16*)malloc(
            (size_t)m * u_cols * sizeof(caml_ba_bfloat16));
        caml_ba_bfloat16* S =
            (caml_ba_bfloat16*)malloc((size_t)minmn * sizeof(caml_ba_bfloat16));
        caml_ba_bfloat16* VT = (caml_ba_bfloat16*)malloc(
            (size_t)vt_rows * n * sizeof(caml_ba_bfloat16));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        svd_bfloat16(A, U, S, VT, m, n, full_matrices);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < u_cols; j++) {
            base_u[i * s_u_row + j * s_u_col] = U[i * u_cols + j];
          }
        }
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        for (int i = 0; i < vt_rows; i++) {
          for (int j = 0; j < n; j++) {
            base_vt[i * s_vt_row + j * s_vt_col] = VT[i * n + j];
          }
        }
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case NX_BA_FP8_E4M3: {
        caml_ba_fp8_e4m3* base_in = (caml_ba_fp8_e4m3*)ba_in->data + off_in;
        caml_ba_fp8_e4m3* base_u = (caml_ba_fp8_e4m3*)ba_u->data + off_u;
        caml_ba_fp8_e4m3* base_s = (caml_ba_fp8_e4m3*)ba_s->data + off_s;
        caml_ba_fp8_e4m3* base_vt = (caml_ba_fp8_e4m3*)ba_vt->data + off_vt;
        caml_ba_fp8_e4m3* A =
            (caml_ba_fp8_e4m3*)malloc((size_t)m * n * sizeof(caml_ba_fp8_e4m3));
        caml_ba_fp8_e4m3* U = (caml_ba_fp8_e4m3*)malloc(
            (size_t)m * u_cols * sizeof(caml_ba_fp8_e4m3));
        caml_ba_fp8_e4m3* S =
            (caml_ba_fp8_e4m3*)malloc((size_t)minmn * sizeof(caml_ba_fp8_e4m3));
        caml_ba_fp8_e4m3* VT = (caml_ba_fp8_e4m3*)malloc(
            (size_t)vt_rows * n * sizeof(caml_ba_fp8_e4m3));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        svd_f8e4m3(A, U, S, VT, m, n, full_matrices);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < u_cols; j++) {
            base_u[i * s_u_row + j * s_u_col] = U[i * u_cols + j];
          }
        }
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        for (int i = 0; i < vt_rows; i++) {
          for (int j = 0; j < n; j++) {
            base_vt[i * s_vt_row + j * s_vt_col] = VT[i * n + j];
          }
        }
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case NX_BA_FP8_E5M2: {
        caml_ba_fp8_e5m2* base_in = (caml_ba_fp8_e5m2*)ba_in->data + off_in;
        caml_ba_fp8_e5m2* base_u = (caml_ba_fp8_e5m2*)ba_u->data + off_u;
        caml_ba_fp8_e5m2* base_s = (caml_ba_fp8_e5m2*)ba_s->data + off_s;
        caml_ba_fp8_e5m2* base_vt = (caml_ba_fp8_e5m2*)ba_vt->data + off_vt;
        caml_ba_fp8_e5m2* A =
            (caml_ba_fp8_e5m2*)malloc((size_t)m * n * sizeof(caml_ba_fp8_e5m2));
        caml_ba_fp8_e5m2* U = (caml_ba_fp8_e5m2*)malloc(
            (size_t)m * u_cols * sizeof(caml_ba_fp8_e5m2));
        caml_ba_fp8_e5m2* S =
            (caml_ba_fp8_e5m2*)malloc((size_t)minmn * sizeof(caml_ba_fp8_e5m2));
        caml_ba_fp8_e5m2* VT = (caml_ba_fp8_e5m2*)malloc(
            (size_t)vt_rows * n * sizeof(caml_ba_fp8_e5m2));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        svd_f8e5m2(A, U, S, VT, m, n, full_matrices);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < u_cols; j++) {
            base_u[i * s_u_row + j * s_u_col] = U[i * u_cols + j];
          }
        }
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        for (int i = 0; i < vt_rows; i++) {
          for (int j = 0; j < n; j++) {
            base_vt[i * s_vt_row + j * s_vt_col] = VT[i * n + j];
          }
        }
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      default:
        caml_leave_blocking_section();
        cleanup_ndarray(&in);
        cleanup_ndarray(&u_nd);
        cleanup_ndarray(&s_nd);
        cleanup_ndarray(&vt_nd);
        caml_failwith("svd: unsupported dtype");
    }
  }
  caml_leave_blocking_section();
  cleanup_ndarray(&in);
  cleanup_ndarray(&u_nd);
  cleanup_ndarray(&s_nd);
  cleanup_ndarray(&vt_nd);
  CAMLreturn(Val_unit);
}
