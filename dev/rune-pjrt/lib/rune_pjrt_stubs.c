/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef RUNE_PJRT_VENDOR_XLA
#include "xla/pjrt/c/pjrt_c_api.h"

typedef const PJRT_Api* (*rune_get_pjrt_api_fn)(void);

typedef struct rune_exec_cache {
  char* cache_key;
  char* plugin_path;
  int device_id;
  void* handle;
  const PJRT_Api* api;
  PJRT_Client* client;
  PJRT_Device* device;
  PJRT_LoadedExecutable* executable;
  PJRT_Executable* executable_view;
  size_t output_count;
  size_t constant_count;
  PJRT_Buffer** constant_buffers;
  struct rune_exec_cache* next;
} rune_exec_cache;

static rune_exec_cache* rune_exec_cache_head = NULL;

static char* rune_dup_bytes(const char* src, size_t len) {
  char* dst = malloc(len + 1);
  if (dst == NULL) return NULL;
  if (len > 0) memcpy(dst, src, len);
  dst[len] = '\0';
  return dst;
}

static char* rune_dup_cstr(const char* src) {
  return rune_dup_bytes(src, strlen(src));
}

static void rune_pjrt_destroy_error(const PJRT_Api* api, PJRT_Error* error) {
  if (api == NULL || error == NULL) return;
  PJRT_Error_Destroy_Args args;
  memset(&args, 0, sizeof(args));
  args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  args.error = error;
  api->PJRT_Error_Destroy(&args);
}

static char* rune_pjrt_error_message(const PJRT_Api* api, PJRT_Error* error) {
  if (error == NULL) return NULL;
  PJRT_Error_Message_Args args;
  char* message;
  memset(&args, 0, sizeof(args));
  args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  args.error = error;
  api->PJRT_Error_Message(&args);
  message = rune_dup_bytes(args.message, args.message_size);
  rune_pjrt_destroy_error(api, error);
  return message;
}

static void rune_pjrt_destroy_event(const PJRT_Api* api, PJRT_Event* event) {
  if (api == NULL || event == NULL) return;
  PJRT_Event_Destroy_Args args;
  memset(&args, 0, sizeof(args));
  args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  args.event = event;
  api->PJRT_Event_Destroy(&args);
}

static void rune_pjrt_destroy_buffer(const PJRT_Api* api, PJRT_Buffer* buffer) {
  if (api == NULL || buffer == NULL) return;
  PJRT_Buffer_Destroy_Args args;
  memset(&args, 0, sizeof(args));
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = buffer;
  api->PJRT_Buffer_Destroy(&args);
}

static void rune_pjrt_destroy_executable(const PJRT_Api* api,
                                         PJRT_LoadedExecutable* executable) {
  if (api == NULL || executable == NULL) return;
  PJRT_LoadedExecutable_Destroy_Args args;
  memset(&args, 0, sizeof(args));
  args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  args.executable = executable;
  api->PJRT_LoadedExecutable_Destroy(&args);
}

static void rune_pjrt_destroy_plain_executable(const PJRT_Api* api,
                                               PJRT_Executable* executable) {
  if (api == NULL || executable == NULL) return;
  PJRT_Executable_Destroy_Args args;
  memset(&args, 0, sizeof(args));
  args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  args.executable = executable;
  api->PJRT_Executable_Destroy(&args);
}

static void rune_pjrt_destroy_client(const PJRT_Api* api, PJRT_Client* client) {
  if (api == NULL || client == NULL) return;
  PJRT_Client_Destroy_Args args;
  memset(&args, 0, sizeof(args));
  args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  args.client = client;
  api->PJRT_Client_Destroy(&args);
}

static void rune_pjrt_destroy_buffer_array(const PJRT_Api* api,
                                           PJRT_Buffer** buffers,
                                           size_t count) {
  size_t i;
  if (buffers == NULL) return;
  for (i = 0; i < count; ++i) rune_pjrt_destroy_buffer(api, buffers[i]);
  free(buffers);
}

static int rune_dtype_size(const char* dtype) {
  if (strcmp(dtype, "float32") == 0) return 4;
  if (strcmp(dtype, "float64") == 0) return 8;
  if (strcmp(dtype, "float16") == 0) return 2;
  if (strcmp(dtype, "int8") == 0) return 1;
  if (strcmp(dtype, "uint8") == 0) return 1;
  if (strcmp(dtype, "int16") == 0) return 2;
  if (strcmp(dtype, "uint16") == 0) return 2;
  if (strcmp(dtype, "int32") == 0) return 4;
  if (strcmp(dtype, "uint32") == 0) return 4;
  if (strcmp(dtype, "int64") == 0) return 8;
  if (strcmp(dtype, "uint64") == 0) return 8;
  if (strcmp(dtype, "bool") == 0) return 1;
  return -1;
}

static PJRT_Buffer_Type rune_pjrt_type_of_dtype(const char* dtype) {
  if (strcmp(dtype, "float32") == 0) return PJRT_Buffer_Type_F32;
  if (strcmp(dtype, "float64") == 0) return PJRT_Buffer_Type_F64;
  if (strcmp(dtype, "float16") == 0) return PJRT_Buffer_Type_F16;
  if (strcmp(dtype, "int8") == 0) return PJRT_Buffer_Type_S8;
  if (strcmp(dtype, "uint8") == 0) return PJRT_Buffer_Type_U8;
  if (strcmp(dtype, "int16") == 0) return PJRT_Buffer_Type_S16;
  if (strcmp(dtype, "uint16") == 0) return PJRT_Buffer_Type_U16;
  if (strcmp(dtype, "int32") == 0) return PJRT_Buffer_Type_S32;
  if (strcmp(dtype, "uint32") == 0) return PJRT_Buffer_Type_U32;
  if (strcmp(dtype, "int64") == 0) return PJRT_Buffer_Type_S64;
  if (strcmp(dtype, "uint64") == 0) return PJRT_Buffer_Type_U64;
  if (strcmp(dtype, "bool") == 0) return PJRT_Buffer_Type_PRED;
  return PJRT_Buffer_Type_INVALID;
}

static size_t rune_shape_numel(value v_shape) {
  mlsize_t i;
  size_t numel = 1;
  for (i = 0; i < Wosize_val(v_shape); ++i) {
    numel *= (size_t)Long_val(Field(v_shape, i));
  }
  return numel;
}

static int64_t* rune_copy_shape(value v_shape) {
  mlsize_t i;
  mlsize_t rank = Wosize_val(v_shape);
  int64_t* dims = malloc(sizeof(int64_t) * (rank == 0 ? 1 : rank));
  if (dims == NULL) return NULL;
  for (i = 0; i < rank; ++i) {
    dims[i] = (int64_t)Long_val(Field(v_shape, i));
  }
  return dims;
}

static int64_t* rune_make_minor_to_major(size_t rank) {
  size_t i;
  int64_t* dims = malloc(sizeof(int64_t) * (rank == 0 ? 1 : rank));
  if (dims == NULL) return NULL;
  for (i = 0; i < rank; ++i) dims[i] = (int64_t)(rank - 1 - i);
  return dims;
}

static void rune_free_shapes(int64_t** shapes, size_t count) {
  size_t i;
  if (shapes == NULL) return;
  for (i = 0; i < count; ++i) free(shapes[i]);
  free(shapes);
}

static void rune_free_byte_buffers(char** buffers, size_t count) {
  size_t i;
  if (buffers == NULL) return;
  for (i = 0; i < count; ++i) free(buffers[i]);
  free(buffers);
}

static rune_exec_cache* rune_find_exec_cache(const char* cache_key) {
  rune_exec_cache* entry = rune_exec_cache_head;
  while (entry != NULL) {
    if (strcmp(entry->cache_key, cache_key) == 0) return entry;
    entry = entry->next;
  }
  return NULL;
}

static void rune_free_exec_cache_entry(rune_exec_cache* entry) {
  if (entry == NULL) return;
  rune_pjrt_destroy_buffer_array(entry->api, entry->constant_buffers,
                                 entry->constant_count);
  rune_pjrt_destroy_executable(entry->api, entry->executable);
  rune_pjrt_destroy_plain_executable(entry->api, entry->executable_view);
  rune_pjrt_destroy_client(entry->api, entry->client);
  if (entry->handle != NULL) dlclose(entry->handle);
  free(entry->cache_key);
  free(entry->plugin_path);
  free(entry);
}

static char* rune_load_pjrt_error(const char* prefix, const char* detail) {
  size_t a = strlen(prefix);
  size_t b = strlen(detail);
  char* msg = malloc(a + 2 + b + 1);
  if (msg == NULL) return NULL;
  memcpy(msg, prefix, a);
  msg[a] = ':';
  msg[a + 1] = ' ';
  memcpy(msg + a + 2, detail, b);
  msg[a + 2 + b] = '\0';
  return msg;
}

static char* rune_await_event(const PJRT_Api* api, PJRT_Event* event) {
  PJRT_Event_Await_Args args;
  PJRT_Error* error;
  if (event == NULL) return NULL;
  memset(&args, 0, sizeof(args));
  args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  args.event = event;
  error = api->PJRT_Event_Await(&args);
  rune_pjrt_destroy_event(api, event);
  return rune_pjrt_error_message(api, error);
}

static char* rune_upload_buffer_from_ocaml(const PJRT_Api* api,
                                           PJRT_Client* client,
                                           PJRT_Device* device, value v_dtype,
                                           value v_shape, value v_data,
                                           PJRT_Buffer** out_buffer,
                                           int64_t** out_dims) {
  PJRT_Client_BufferFromHostBuffer_Args buffer_args;
  PJRT_Error* error;
  PJRT_Event* event = NULL;
  const char* dtype = String_val(v_dtype);
  int itemsize = rune_dtype_size(dtype);
  size_t expected_bytes;
  PJRT_Buffer_Type pjrt_type = rune_pjrt_type_of_dtype(dtype);
  int64_t* dims = NULL;

  if (itemsize <= 0 || pjrt_type == PJRT_Buffer_Type_INVALID) {
    return rune_load_pjrt_error("unsupported input dtype", dtype);
  }

  expected_bytes = rune_shape_numel(v_shape) * (size_t)itemsize;
  if (caml_string_length(v_data) != expected_bytes) {
    char detail[128];
    snprintf(detail, sizeof(detail), "%s buffer has %zu bytes, expected %zu",
             dtype, caml_string_length(v_data), expected_bytes);
    return rune_load_pjrt_error("input byte size mismatch", detail);
  }

  dims = rune_copy_shape(v_shape);
  if (dims == NULL) return rune_dup_cstr("out of memory");

  memset(&buffer_args, 0, sizeof(buffer_args));
  buffer_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  buffer_args.client = client;
  buffer_args.data = String_val(v_data);
  buffer_args.type = pjrt_type;
  buffer_args.dims = dims;
  buffer_args.num_dims = Wosize_val(v_shape);
  buffer_args.host_buffer_semantics =
      PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  buffer_args.device = device;
  error = api->PJRT_Client_BufferFromHostBuffer(&buffer_args);
  if (error != NULL) {
    free(dims);
    return rune_pjrt_error_message(api, error);
  }

  event = buffer_args.done_with_host_buffer;
  if (event != NULL) {
    char* event_error = rune_await_event(api, event);
    if (event_error != NULL) {
      rune_pjrt_destroy_buffer(api, buffer_args.buffer);
      free(dims);
      return event_error;
    }
  }

  *out_buffer = buffer_args.buffer;
  if (out_dims != NULL)
    *out_dims = dims;
  else
    free(dims);
  return NULL;
}

static value rune_copy_outputs_to_ocaml(char** output_bytes, size_t* output_sizes,
                                        size_t output_count) {
  CAMLparam0();
  CAMLlocal2(v_result, v_item);
  size_t i;
  v_result = caml_alloc(output_count, 0);
  for (i = 0; i < output_count; ++i) {
    v_item = caml_alloc_string(output_sizes[i]);
    memcpy((char*)String_val(v_item), output_bytes[i], output_sizes[i]);
    Store_field(v_result, i, v_item);
  }
  CAMLreturn(v_result);
}

static value rune_execute(value v_plugin_path, value v_cache_key, value v_device_id,
                          value v_stablehlo, value v_dynamic_input_dtypes,
                          value v_dynamic_input_shapes,
                          value v_dynamic_input_data,
                          value v_constant_input_dtypes,
                          value v_constant_input_shapes,
                          value v_constant_input_data, value v_output_dtypes,
                          value v_output_shapes) {
  CAMLparam5(v_plugin_path, v_cache_key, v_device_id, v_stablehlo,
             v_dynamic_input_dtypes);
  CAMLxparam5(v_dynamic_input_shapes, v_dynamic_input_data,
              v_constant_input_dtypes, v_constant_input_shapes,
              v_constant_input_data);
  CAMLxparam2(v_output_dtypes, v_output_shapes);
  rune_exec_cache* cache = NULL;
  rune_exec_cache* new_cache = NULL;
  void* handle = NULL;
  const PJRT_Api* api = NULL;
  rune_get_pjrt_api_fn get_api = NULL;
  PJRT_Client* client = NULL;
  PJRT_Device* device = NULL;
  PJRT_LoadedExecutable* executable = NULL;
  PJRT_Executable* executable_view = NULL;
  PJRT_Buffer** dynamic_input_buffers = NULL;
  PJRT_Buffer** argument_buffers = NULL;
  PJRT_Buffer** output_buffers = NULL;
  int64_t** dynamic_input_dims = NULL;
  PJRT_Buffer** constant_buffers = NULL;
  char** output_bytes = NULL;
  size_t* output_sizes = NULL;
  char* error_message = NULL;
  size_t dynamic_input_count = Wosize_val(v_dynamic_input_dtypes);
  size_t constant_input_count = Wosize_val(v_constant_input_dtypes);
  size_t output_count = Wosize_val(v_output_dtypes);
  size_t executable_output_count = 0;
  size_t total_input_count = 0;
  size_t i;
  value result = Val_unit;

  if (Wosize_val(v_dynamic_input_shapes) != dynamic_input_count ||
      Wosize_val(v_dynamic_input_data) != dynamic_input_count) {
    caml_invalid_argument("rune_pjrt_execute: mismatched dynamic input metadata");
  }
  if (Wosize_val(v_constant_input_shapes) != constant_input_count ||
      Wosize_val(v_constant_input_data) != constant_input_count) {
    caml_invalid_argument("rune_pjrt_execute: mismatched constant input metadata");
  }
  if (Wosize_val(v_output_shapes) != output_count) {
    caml_invalid_argument("rune_pjrt_execute: mismatched output metadata");
  }

  cache = rune_find_exec_cache(String_val(v_cache_key));
  if (cache != NULL) {
    api = cache->api;
    client = cache->client;
    device = cache->device;
    executable = cache->executable;
    executable_view = cache->executable_view;
    executable_output_count = cache->output_count;
    if (cache->constant_count != constant_input_count) {
      error_message = rune_dup_cstr("cached executable constant arity mismatch");
      goto cleanup;
    }
  } else {
    handle = dlopen(String_val(v_plugin_path), RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
      error_message = rune_load_pjrt_error("dlopen failed", dlerror());
      goto cleanup;
    }

    get_api = (rune_get_pjrt_api_fn)dlsym(handle, "GetPjrtApi");
    if (get_api == NULL) {
      error_message = rune_load_pjrt_error("dlsym(GetPjrtApi) failed", dlerror());
      goto cleanup;
    }

    api = get_api();
    if (api == NULL) {
      error_message = rune_dup_cstr("GetPjrtApi returned null");
      goto cleanup;
    }

    if (api->PJRT_Plugin_Initialize != NULL) {
      PJRT_Plugin_Initialize_Args init_args;
      PJRT_Error* error;
      memset(&init_args, 0, sizeof(init_args));
      init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
      error = api->PJRT_Plugin_Initialize(&init_args);
      if (error != NULL) {
        error_message = rune_pjrt_error_message(api, error);
        goto cleanup;
      }
    }

    {
      PJRT_Client_Create_Args create_args;
      PJRT_NamedValue create_options[3];
      int64_t visible_devices[1];
      size_t num_options = 0;
      PJRT_Error* error;
      memset(&create_args, 0, sizeof(create_args));
      create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
      if (strstr(String_val(v_plugin_path), "gpu_plugin") != NULL) {
        memset(&create_options[num_options], 0, sizeof(PJRT_NamedValue));
        create_options[num_options].struct_size = PJRT_NamedValue_STRUCT_SIZE;
        create_options[num_options].name = "allocator";
        create_options[num_options].name_size = strlen("allocator");
        create_options[num_options].type = PJRT_NamedValue_kString;
        create_options[num_options].string_value = "bfc";
        create_options[num_options].value_size = strlen("bfc");
        ++num_options;

        visible_devices[0] = Int_val(v_device_id);
        memset(&create_options[num_options], 0, sizeof(PJRT_NamedValue));
        create_options[num_options].struct_size = PJRT_NamedValue_STRUCT_SIZE;
        create_options[num_options].name = "visible_devices";
        create_options[num_options].name_size = strlen("visible_devices");
        create_options[num_options].type = PJRT_NamedValue_kInt64List;
        create_options[num_options].int64_array_value = visible_devices;
        create_options[num_options].value_size = 1;
        ++num_options;

        memset(&create_options[num_options], 0, sizeof(PJRT_NamedValue));
        create_options[num_options].struct_size = PJRT_NamedValue_STRUCT_SIZE;
        create_options[num_options].name = "preallocate";
        create_options[num_options].name_size = strlen("preallocate");
        create_options[num_options].type = PJRT_NamedValue_kBool;
        create_options[num_options].bool_value = false;
        create_options[num_options].value_size = 1;
        ++num_options;

        create_args.create_options = create_options;
        create_args.num_options = num_options;
      }
      error = api->PJRT_Client_Create(&create_args);
      if (error != NULL) {
        error_message = rune_pjrt_error_message(api, error);
        goto cleanup;
      }
      client = create_args.client;
    }

    {
      PJRT_Client_AddressableDevices_Args addr_args;
      int device_id = Int_val(v_device_id);
      PJRT_Error* error;
      memset(&addr_args, 0, sizeof(addr_args));
      addr_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
      addr_args.client = client;
      error = api->PJRT_Client_AddressableDevices(&addr_args);
      if (error != NULL) {
        error_message = rune_pjrt_error_message(api, error);
        goto cleanup;
      }
      if (device_id < 0 || (size_t)device_id >= addr_args.num_addressable_devices) {
        error_message = rune_dup_bytes("device_id out of range", 22);
        goto cleanup;
      }
      if (device_id != 0) {
        error_message =
            rune_dup_cstr("device_id > 0 is not supported until compile options are wired");
        goto cleanup;
      }
      device = addr_args.addressable_devices[device_id];
    }

    {
      static const char k_mlir[] = "mlir";
      static const char k_compile_options[] = {
          0x1a, 0x06, 0x08, 0x00, 0x20, 0x01, 0x28, 0x01};
      PJRT_Program program;
      PJRT_Client_Compile_Args compile_args;
      PJRT_Error* error;
      memset(&program, 0, sizeof(program));
      program.struct_size = PJRT_Program_STRUCT_SIZE;
      program.code = (char*)String_val(v_stablehlo);
      program.code_size = caml_string_length(v_stablehlo);
      program.format = k_mlir;
      program.format_size = sizeof(k_mlir) - 1;

      memset(&compile_args, 0, sizeof(compile_args));
      compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
      compile_args.client = client;
      compile_args.program = &program;
      compile_args.compile_options = k_compile_options;
      compile_args.compile_options_size = sizeof(k_compile_options);
      error = api->PJRT_Client_Compile(&compile_args);
      if (error != NULL) {
        error_message = rune_pjrt_error_message(api, error);
        goto cleanup;
      }
      executable = compile_args.executable;
    }

    {
      PJRT_LoadedExecutable_GetExecutable_Args get_exec_args;
      PJRT_Executable_NumOutputs_Args num_outputs_args;
      PJRT_Error* error;
      memset(&get_exec_args, 0, sizeof(get_exec_args));
      get_exec_args.struct_size =
          PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
      get_exec_args.loaded_executable = executable;
      error = api->PJRT_LoadedExecutable_GetExecutable(&get_exec_args);
      if (error != NULL) {
        error_message = rune_pjrt_error_message(api, error);
        goto cleanup;
      }
      executable_view = get_exec_args.executable;

      memset(&num_outputs_args, 0, sizeof(num_outputs_args));
      num_outputs_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
      num_outputs_args.executable = executable_view;
      error = api->PJRT_Executable_NumOutputs(&num_outputs_args);
      if (error != NULL) {
        error_message = rune_pjrt_error_message(api, error);
        goto cleanup;
      }
      executable_output_count = num_outputs_args.num_outputs;
      if (executable_output_count != output_count) {
        char detail[128];
        snprintf(detail, sizeof(detail),
                 "expected %zu outputs, executable returns %zu", output_count,
                 executable_output_count);
        error_message = rune_load_pjrt_error("output arity mismatch", detail);
        goto cleanup;
      }
    }

    constant_buffers =
        calloc(constant_input_count == 0 ? 1 : constant_input_count,
               sizeof(PJRT_Buffer*));
    if (constant_buffers == NULL) {
      error_message = rune_dup_cstr("out of memory");
      goto cleanup;
    }
    for (i = 0; i < constant_input_count; ++i) {
      error_message = rune_upload_buffer_from_ocaml(
          api, client, device, Field(v_constant_input_dtypes, i),
          Field(v_constant_input_shapes, i), Field(v_constant_input_data, i),
          &constant_buffers[i], NULL);
      if (error_message != NULL) goto cleanup;
    }

    new_cache = malloc(sizeof(rune_exec_cache));
    if (new_cache == NULL) {
      error_message = rune_dup_cstr("out of memory");
      goto cleanup;
    }
    memset(new_cache, 0, sizeof(*new_cache));
    new_cache->cache_key = rune_dup_cstr(String_val(v_cache_key));
    new_cache->plugin_path = rune_dup_cstr(String_val(v_plugin_path));
    if (new_cache->cache_key == NULL || new_cache->plugin_path == NULL) {
      error_message = rune_dup_cstr("out of memory");
      goto cleanup;
    }
    new_cache->device_id = Int_val(v_device_id);
    new_cache->handle = handle;
    new_cache->api = api;
    new_cache->client = client;
    new_cache->device = device;
    new_cache->executable = executable;
    new_cache->executable_view = executable_view;
    new_cache->output_count = executable_output_count;
    new_cache->constant_count = constant_input_count;
    new_cache->constant_buffers = constant_buffers;
    new_cache->next = rune_exec_cache_head;
    rune_exec_cache_head = new_cache;
    cache = new_cache;

    handle = NULL;
    client = NULL;
    executable = NULL;
    executable_view = NULL;
    constant_buffers = NULL;
    new_cache = NULL;
  }

  if (cache->output_count != output_count) {
    error_message = rune_dup_cstr("cached executable output arity mismatch");
    goto cleanup;
  }

  executable_output_count = cache->output_count;
  total_input_count = dynamic_input_count + cache->constant_count;
  dynamic_input_buffers =
      calloc(dynamic_input_count == 0 ? 1 : dynamic_input_count,
             sizeof(PJRT_Buffer*));
  dynamic_input_dims =
      calloc(dynamic_input_count == 0 ? 1 : dynamic_input_count,
             sizeof(int64_t*));
  argument_buffers =
      calloc(total_input_count == 0 ? 1 : total_input_count,
             sizeof(PJRT_Buffer*));
  output_buffers =
      calloc(executable_output_count == 0 ? 1 : executable_output_count,
             sizeof(PJRT_Buffer*));
  output_bytes =
      calloc(output_count == 0 ? 1 : output_count, sizeof(char*));
  output_sizes =
      calloc(output_count == 0 ? 1 : output_count, sizeof(size_t));
  if (dynamic_input_buffers == NULL || dynamic_input_dims == NULL ||
      argument_buffers == NULL || output_buffers == NULL ||
      output_bytes == NULL || output_sizes == NULL) {
    error_message = rune_dup_cstr("out of memory");
    goto cleanup;
  }

  for (i = 0; i < dynamic_input_count; ++i) {
    error_message = rune_upload_buffer_from_ocaml(
        cache->api, cache->client, cache->device, Field(v_dynamic_input_dtypes, i),
        Field(v_dynamic_input_shapes, i), Field(v_dynamic_input_data, i),
        &dynamic_input_buffers[i], &dynamic_input_dims[i]);
    if (error_message != NULL) goto cleanup;
    argument_buffers[i] = dynamic_input_buffers[i];
  }
  for (i = 0; i < cache->constant_count; ++i) {
    argument_buffers[dynamic_input_count + i] = cache->constant_buffers[i];
  }

  {
    PJRT_ExecuteOptions execute_options;
    PJRT_LoadedExecutable_Execute_Args execute_args;
    PJRT_Buffer* const* argument_lists[1];
    PJRT_Buffer** output_lists[1];
    PJRT_Error* error;
    memset(&execute_options, 0, sizeof(execute_options));
    execute_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    memset(&execute_args, 0, sizeof(execute_args));
    execute_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = cache->executable;
    execute_args.options = &execute_options;
    argument_lists[0] = argument_buffers;
    output_lists[0] = output_buffers;
    execute_args.argument_lists = argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = total_input_count;
    execute_args.output_lists = output_lists;
    execute_args.execute_device = NULL;
    error = cache->api->PJRT_LoadedExecutable_Execute(&execute_args);
    if (error != NULL) {
      error_message = rune_pjrt_error_message(cache->api, error);
      goto cleanup;
    }
  }

  for (i = 0; i < output_count; ++i) {
    value v_dtype = Field(v_output_dtypes, i);
    value v_shape = Field(v_output_shapes, i);
    PJRT_Buffer_ToHostBuffer_Args copy_args;
    PJRT_Buffer_ToHostBuffer_Args size_args;
    PJRT_Error* error;
    int itemsize = rune_dtype_size(String_val(v_dtype));
    size_t numel = rune_shape_numel(v_shape);
    size_t rank = Wosize_val(v_shape);
    int64_t* minor_to_major = NULL;
    PJRT_Buffer_MemoryLayout host_layout;
    PJRT_Event* event = NULL;
    if (itemsize <= 0) {
      error_message = rune_load_pjrt_error("unsupported output dtype",
                                           String_val(v_dtype));
      goto cleanup;
    }
    minor_to_major = rune_make_minor_to_major(rank);
    if (minor_to_major == NULL) {
      error_message = rune_dup_cstr("out of memory");
      goto cleanup;
    }
    memset(&host_layout, 0, sizeof(host_layout));
    host_layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
    host_layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
    host_layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
    host_layout.tiled.minor_to_major = minor_to_major;
    host_layout.tiled.minor_to_major_size = rank;

    memset(&size_args, 0, sizeof(size_args));
    size_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    size_args.src = output_buffers[i];
    size_args.host_layout = &host_layout;
    size_args.dst = NULL;
    size_args.dst_size = 0;
    error = cache->api->PJRT_Buffer_ToHostBuffer(&size_args);
    if (error != NULL) {
      free(minor_to_major);
      error_message = rune_pjrt_error_message(cache->api, error);
      goto cleanup;
    }
    output_sizes[i] = size_args.dst_size;
    if (output_sizes[i] != numel * (size_t)itemsize) {
      char detail[160];
      snprintf(detail, sizeof(detail),
               "%s output requires %zu bytes, expected %zu from traced metadata",
               String_val(v_dtype), output_sizes[i], numel * (size_t)itemsize);
      free(minor_to_major);
      error_message = rune_load_pjrt_error("output byte size mismatch", detail);
      goto cleanup;
    }
    output_bytes[i] = malloc(output_sizes[i] == 0 ? 1 : output_sizes[i]);
    if (output_bytes[i] == NULL) {
      free(minor_to_major);
      error_message = rune_dup_cstr("out of memory");
      goto cleanup;
    }

    memset(&copy_args, 0, sizeof(copy_args));
    copy_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    copy_args.src = output_buffers[i];
    copy_args.host_layout = &host_layout;
    copy_args.dst = output_bytes[i];
    copy_args.dst_size = output_sizes[i];
    error = cache->api->PJRT_Buffer_ToHostBuffer(&copy_args);
    if (error != NULL) {
      free(minor_to_major);
      error_message = rune_pjrt_error_message(cache->api, error);
      goto cleanup;
    }
    free(minor_to_major);
    event = copy_args.event;
    if (event != NULL) {
      error_message = rune_await_event(cache->api, event);
      if (error_message != NULL) goto cleanup;
    }
  }

  result = rune_copy_outputs_to_ocaml(output_bytes, output_sizes, output_count);

cleanup:
  if (output_buffers != NULL) {
    for (i = 0; i < executable_output_count; ++i) {
      rune_pjrt_destroy_buffer(cache != NULL ? cache->api : api, output_buffers[i]);
    }
  }
  if (dynamic_input_buffers != NULL) {
    for (i = 0; i < dynamic_input_count; ++i) {
      rune_pjrt_destroy_buffer(cache != NULL ? cache->api : api,
                               dynamic_input_buffers[i]);
    }
  }
  rune_free_shapes(dynamic_input_dims, dynamic_input_count);

  if (error_message != NULL) {
    char* message = error_message;
    rune_free_byte_buffers(output_bytes, output_count);
    free(output_sizes);
    free(dynamic_input_buffers);
    free(argument_buffers);
    free(output_buffers);
    if (new_cache != NULL)
      rune_free_exec_cache_entry(new_cache);
    else {
      rune_pjrt_destroy_buffer_array(api, constant_buffers, constant_input_count);
      rune_pjrt_destroy_executable(api, executable);
      rune_pjrt_destroy_plain_executable(api, executable_view);
      rune_pjrt_destroy_client(api, client);
      if (handle != NULL) dlclose(handle);
    }
    caml_failwith(message);
  }

  rune_free_byte_buffers(output_bytes, output_count);
  free(output_sizes);
  free(dynamic_input_buffers);
  free(argument_buffers);
  free(output_buffers);
  CAMLreturn(result);
}

CAMLprim value caml_rune_pjrt_execute(
    value v_plugin_path, value v_cache_key, value v_device_id,
    value v_stablehlo, value v_dynamic_input_dtypes,
    value v_dynamic_input_shapes, value v_dynamic_input_data,
    value v_constant_input_dtypes, value v_constant_input_shapes,
    value v_constant_input_data, value v_output_dtypes,
    value v_output_shapes) {
  return rune_execute(
      v_plugin_path, v_cache_key, v_device_id, v_stablehlo,
      v_dynamic_input_dtypes, v_dynamic_input_shapes, v_dynamic_input_data,
      v_constant_input_dtypes, v_constant_input_shapes, v_constant_input_data,
      v_output_dtypes, v_output_shapes);
}

CAMLprim value caml_rune_pjrt_execute_bc(value* argv, int argn) {
  if (argn != 12) caml_invalid_argument("rune_pjrt_execute: arity");
  return rune_execute(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                      argv[6], argv[7], argv[8], argv[9], argv[10], argv[11]);
}

#else

CAMLprim value caml_rune_pjrt_execute(value v_plugin_path, value v_device_id,
                                      value v_stablehlo, value v_input_dtypes,
                                      value v_input_shapes, value v_input_data,
                                      value v_output_dtypes,
                                      value v_output_shapes) {
  (void)v_plugin_path;
  (void)v_device_id;
  (void)v_stablehlo;
  (void)v_input_dtypes;
  (void)v_input_shapes;
  (void)v_input_data;
  (void)v_output_dtypes;
  (void)v_output_shapes;
  caml_failwith("rune-pjrt was built without vendor/xla available");
}

CAMLprim value caml_rune_pjrt_execute_bc(value* argv, int argn) {
  (void)argv;
  (void)argn;
  caml_failwith("rune-pjrt was built without vendor/xla available");
}

#endif
