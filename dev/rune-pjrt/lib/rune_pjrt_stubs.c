/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
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
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"

typedef const PJRT_Api* (*rune_get_pjrt_api_fn)(void);

typedef struct rune_client_cache {
  char* plugin_path;
  int device_id;
  void* handle;
  const PJRT_Api* api;
  PJRT_Client* client;
  PJRT_Device* device;
  struct rune_client_cache* next;
} rune_client_cache;

typedef struct rune_exec_cache {
  char* cache_key;
  char* plugin_path;
  int device_id;
  PJRT_LoadedExecutable* executable;
  PJRT_Executable* executable_view;
  size_t output_count;
  size_t constant_count;
  PJRT_Buffer** constant_buffers;
  rune_client_cache* runtime;
  struct rune_exec_cache* next;
} rune_exec_cache;

typedef struct rune_device_buffer {
  rune_client_cache* runtime;
  PJRT_Buffer* buffer;
} rune_device_buffer;

static rune_client_cache* rune_client_cache_head = NULL;
static rune_exec_cache* rune_exec_cache_head = NULL;

typedef struct rune_ffi_registration {
  char* plugin_path;
  char* library_path;
  char* library_digest;
  char* symbol;
  char* target;
  void* plugin_handle;
  void* library_handle;
  struct rune_ffi_registration* next;
} rune_ffi_registration;

static rune_ffi_registration* rune_ffi_registration_head = NULL;

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

#define Rune_device_buffer_val(v) \
  ((rune_device_buffer*)Data_custom_val(v))

static void rune_finalize_device_buffer(value v) {
  rune_device_buffer* device_buffer = Rune_device_buffer_val(v);
  if (device_buffer->runtime != NULL && device_buffer->buffer != NULL) {
    rune_pjrt_destroy_buffer(device_buffer->runtime->api,
                             device_buffer->buffer);
    device_buffer->buffer = NULL;
  }
}

static struct custom_operations rune_device_buffer_ops = {
    "raven.rune_pjrt.device_buffer",
    rune_finalize_device_buffer,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static value rune_alloc_device_buffer(rune_client_cache* runtime,
                                      PJRT_Buffer* buffer,
                                      size_t dependent_memory) {
  value result = caml_alloc_custom_mem(
      &rune_device_buffer_ops, sizeof(rune_device_buffer), dependent_memory);
  rune_device_buffer* device_buffer = Rune_device_buffer_val(result);
  device_buffer->runtime = runtime;
  device_buffer->buffer = buffer;
  return result;
}

static rune_device_buffer* rune_get_device_buffer(value v_buffer) {
  if (!Is_block(v_buffer) || Tag_val(v_buffer) != Custom_tag ||
      Custom_ops_val(v_buffer) != &rune_device_buffer_ops) {
    caml_invalid_argument("rune-pjrt: expected a device buffer");
  }
  return Rune_device_buffer_val(v_buffer);
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
  if (strcmp(dtype, "bfloat16") == 0) return 2;
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
  if (strcmp(dtype, "bfloat16") == 0) return PJRT_Buffer_Type_BF16;
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

static rune_exec_cache* rune_find_exec_cache(const char* cache_key) {
  rune_exec_cache* entry = rune_exec_cache_head;
  while (entry != NULL) {
    if (strcmp(entry->cache_key, cache_key) == 0) return entry;
    entry = entry->next;
  }
  return NULL;
}

static void rune_free_exec_cache_entry(rune_exec_cache* entry) {
  const PJRT_Api* api;
  if (entry == NULL) return;
  api = entry->runtime->api;
  rune_pjrt_destroy_buffer_array(api, entry->constant_buffers,
                                 entry->constant_count);
  rune_pjrt_destroy_executable(api, entry->executable);
  rune_pjrt_destroy_plain_executable(api, entry->executable_view);
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

static rune_client_cache* rune_find_client_cache(const char* plugin_path,
                                                 int device_id) {
  rune_client_cache* runtime = rune_client_cache_head;
  while (runtime != NULL) {
    if (runtime->device_id == device_id &&
        strcmp(runtime->plugin_path, plugin_path) == 0) {
      return runtime;
    }
    runtime = runtime->next;
  }
  return NULL;
}

static char* rune_get_client(const char* plugin_path, int device_id,
                             rune_client_cache** out_runtime) {
  rune_client_cache* runtime =
      rune_find_client_cache(plugin_path, device_id);
  rune_get_pjrt_api_fn get_api = NULL;
  void* handle = NULL;
  const PJRT_Api* api = NULL;
  PJRT_Client* client = NULL;
  PJRT_Device* device = NULL;
  char* error_message = NULL;
  int is_gpu = strstr(plugin_path, "gpu_plugin") != NULL;

  if (runtime != NULL) {
    *out_runtime = runtime;
    return NULL;
  }

  handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    return rune_load_pjrt_error("dlopen failed", dlerror());
  }

  get_api = (rune_get_pjrt_api_fn)dlsym(handle, "GetPjrtApi");
  if (get_api == NULL) {
    error_message =
        rune_load_pjrt_error("dlsym(GetPjrtApi) failed", dlerror());
    goto fail;
  }

  api = get_api();
  if (api == NULL) {
    error_message = rune_dup_cstr("GetPjrtApi returned null");
    goto fail;
  }

  if (api->PJRT_Plugin_Initialize != NULL) {
    PJRT_Plugin_Initialize_Args init_args;
    PJRT_Error* error;
    memset(&init_args, 0, sizeof(init_args));
    init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    error = api->PJRT_Plugin_Initialize(&init_args);
    if (error != NULL) {
      error_message = rune_pjrt_error_message(api, error);
      goto fail;
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
    if (is_gpu) {
      memset(&create_options[num_options], 0, sizeof(PJRT_NamedValue));
      create_options[num_options].struct_size = PJRT_NamedValue_STRUCT_SIZE;
      create_options[num_options].name = "allocator";
      create_options[num_options].name_size = strlen("allocator");
      create_options[num_options].type = PJRT_NamedValue_kString;
      create_options[num_options].string_value = "bfc";
      create_options[num_options].value_size = strlen("bfc");
      ++num_options;

      visible_devices[0] = device_id;
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
      goto fail;
    }
    client = create_args.client;
  }

  {
    PJRT_Client_AddressableDevices_Args addr_args;
    size_t addressable_index = is_gpu ? 0 : (size_t)device_id;
    PJRT_Error* error;
    memset(&addr_args, 0, sizeof(addr_args));
    addr_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    addr_args.client = client;
    error = api->PJRT_Client_AddressableDevices(&addr_args);
    if (error != NULL) {
      error_message = rune_pjrt_error_message(api, error);
      goto fail;
    }
    if (device_id < 0 ||
        addressable_index >= addr_args.num_addressable_devices) {
      error_message = rune_dup_cstr("device_id out of range");
      goto fail;
    }
    device = addr_args.addressable_devices[addressable_index];
  }

  runtime = malloc(sizeof(*runtime));
  if (runtime == NULL) {
    error_message = rune_dup_cstr("out of memory");
    goto fail;
  }
  memset(runtime, 0, sizeof(*runtime));
  runtime->plugin_path = rune_dup_cstr(plugin_path);
  if (runtime->plugin_path == NULL) {
    error_message = rune_dup_cstr("out of memory");
    goto fail;
  }
  runtime->device_id = device_id;
  runtime->handle = handle;
  runtime->api = api;
  runtime->client = client;
  runtime->device = device;
  runtime->next = rune_client_cache_head;
  rune_client_cache_head = runtime;
  *out_runtime = runtime;
  return NULL;

fail:
  if (runtime != NULL) {
    free(runtime->plugin_path);
    free(runtime);
  }
  rune_pjrt_destroy_client(api, client);
  if (handle != NULL) dlclose(handle);
  return error_message != NULL ? error_message
                               : rune_dup_cstr("PJRT client creation failed");
}

CAMLprim value caml_rune_pjrt_register_ffi_handler(
    value v_plugin_path, value v_library_path, value v_library_digest,
    value v_symbol, value v_target) {
  CAMLparam5(v_plugin_path, v_library_path, v_library_digest, v_symbol,
             v_target);
  const char* plugin_path = String_val(v_plugin_path);
  const char* library_path = String_val(v_library_path);
  const char* library_digest = String_val(v_library_digest);
  const char* symbol = String_val(v_symbol);
  const char* target = String_val(v_target);
  rune_ffi_registration* registration = rune_ffi_registration_head;
  void* plugin_handle = NULL;
  void* library_handle = NULL;
  rune_get_pjrt_api_fn get_api = NULL;
  const PJRT_Api* api = NULL;
  const PJRT_Extension_Base* extension = NULL;
  const PJRT_FFI* ffi = NULL;
  PJRT_FFI_Register_Handler_Args args;
  PJRT_Error* error = NULL;
  char* error_message = NULL;
  void* handler = NULL;

  while (registration != NULL) {
    if (strcmp(registration->plugin_path, plugin_path) == 0 &&
        strcmp(registration->target, target) == 0) {
      if (strcmp(registration->library_digest, library_digest) != 0 ||
          strcmp(registration->symbol, symbol) != 0) {
        caml_failwith(
            "PJRT FFI target is already registered to a different handler");
      }
      CAMLreturn(Val_unit);
    }
    if (strcmp(registration->plugin_path, plugin_path) == 0 &&
        strcmp(registration->library_path, library_path) == 0 &&
        strcmp(registration->library_digest, library_digest) != 0) {
      caml_failwith(
          "PJRT FFI library changed after it was loaded; restart the process "
          "or use a content-addressed filename");
    }
    registration = registration->next;
  }

  plugin_handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
  if (plugin_handle == NULL) {
    error_message = rune_load_pjrt_error("PJRT plugin dlopen failed", dlerror());
    goto fail;
  }

  dlerror();
  get_api = (rune_get_pjrt_api_fn)dlsym(plugin_handle, "GetPjrtApi");
  {
    const char* detail = dlerror();
    if (detail != NULL || get_api == NULL) {
      error_message = rune_load_pjrt_error(
          "PJRT plugin does not export GetPjrtApi",
          detail != NULL ? detail : "symbol not found");
      goto fail;
    }
  }

  api = get_api();
  if (api == NULL) {
    error_message = rune_dup_cstr("GetPjrtApi returned NULL");
    goto fail;
  }

  extension = api->extension_start;
  while (extension != NULL && extension->type != PJRT_Extension_Type_FFI) {
    extension = extension->next;
  }
  if (extension == NULL) {
    error_message = rune_dup_cstr("PJRT plugin does not expose the FFI extension");
    goto fail;
  }
  ffi = (const PJRT_FFI*)extension;
  if (ffi->base.struct_size < PJRT_FFI_Extension_STRUCT_SIZE ||
      ffi->register_handler == NULL) {
    error_message = rune_dup_cstr("PJRT FFI extension has no register_handler");
    goto fail;
  }

  library_handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
  if (library_handle == NULL) {
    error_message = rune_load_pjrt_error("FFI library dlopen failed", dlerror());
    goto fail;
  }

  dlerror();
  handler = dlsym(library_handle, symbol);
  {
    const char* detail = dlerror();
    if (detail != NULL || handler == NULL) {
      error_message = rune_load_pjrt_error(
          "FFI handler symbol lookup failed",
          detail != NULL ? detail : "symbol not found");
      goto fail;
    }
  }

  registration = malloc(sizeof(*registration));
  if (registration == NULL) {
    error_message = rune_dup_cstr("failed to allocate PJRT FFI registration");
    goto fail;
  }
  memset(registration, 0, sizeof(*registration));
  registration->plugin_path = rune_dup_cstr(plugin_path);
  registration->library_path = rune_dup_cstr(library_path);
  registration->library_digest = rune_dup_cstr(library_digest);
  registration->symbol = rune_dup_cstr(symbol);
  registration->target = rune_dup_cstr(target);
  if (registration->plugin_path == NULL ||
      registration->library_path == NULL ||
      registration->library_digest == NULL || registration->symbol == NULL ||
      registration->target == NULL) {
    error_message = rune_dup_cstr("failed to copy PJRT FFI registration");
    goto fail;
  }

  memset(&args, 0, sizeof(args));
  args.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
  args.target_name = target;
  args.target_name_size = strlen(target);
  args.handler = handler;
  args.platform_name = "CUDA";
  args.platform_name_size = 4;
  args.traits = (PJRT_FFI_Handler_TraitsBits)0;
  error = ffi->register_handler(&args);
  if (error != NULL) {
    error_message = rune_pjrt_error_message(api, error);
    goto fail;
  }

  registration->plugin_handle = plugin_handle;
  registration->library_handle = library_handle;
  registration->next = rune_ffi_registration_head;
  rune_ffi_registration_head = registration;
  CAMLreturn(Val_unit);

fail:
  if (registration != NULL) {
    free(registration->plugin_path);
    free(registration->library_path);
    free(registration->library_digest);
    free(registration->symbol);
    free(registration->target);
    free(registration);
  }
  if (library_handle != NULL) dlclose(library_handle);
  if (plugin_handle != NULL) dlclose(plugin_handle);
  caml_failwith(error_message != NULL ? error_message
                                     : "PJRT FFI registration failed");
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

static char* rune_await_events(const PJRT_Api* api, PJRT_Event** events,
                               size_t count) {
  char* first_error = NULL;
  size_t i;
  if (events == NULL) return NULL;
  for (i = 0; i < count; ++i) {
    char* error_message;
    if (events[i] == NULL) continue;
    error_message = rune_await_event(api, events[i]);
    events[i] = NULL;
    if (error_message == NULL) continue;
    if (first_error == NULL)
      first_error = error_message;
    else
      free(error_message);
  }
  return first_error;
}

static size_t rune_bigarray_numel(value v_data) {
  struct caml_ba_array* array = Caml_ba_array_val(v_data);
  int index;
  size_t numel = 1;
  for (index = 0; index < array->num_dims; ++index) {
    numel *= (size_t)array->dim[index];
  }
  return numel;
}

static char* rune_upload_buffer(const PJRT_Api* api, PJRT_Client* client,
                                PJRT_Device* device, const char* dtype,
                                value v_shape, const void* data,
                                size_t data_size,
                                PJRT_HostBufferSemantics semantics,
                                PJRT_Buffer** out_buffer, int64_t** out_dims,
                                PJRT_Event** out_event) {
  PJRT_Client_BufferFromHostBuffer_Args buffer_args;
  PJRT_Error* error;
  PJRT_Event* event = NULL;
  int itemsize = rune_dtype_size(dtype);
  size_t expected_bytes;
  PJRT_Buffer_Type pjrt_type = rune_pjrt_type_of_dtype(dtype);
  int64_t* dims = NULL;

  if (itemsize <= 0 || pjrt_type == PJRT_Buffer_Type_INVALID) {
    return rune_load_pjrt_error("unsupported input dtype", dtype);
  }

  expected_bytes = rune_shape_numel(v_shape) * (size_t)itemsize;
  if (data_size != expected_bytes) {
    char detail[128];
    snprintf(detail, sizeof(detail), "%s buffer has %zu bytes, expected %zu",
             dtype, data_size, expected_bytes);
    return rune_load_pjrt_error("input byte size mismatch", detail);
  }

  dims = rune_copy_shape(v_shape);
  if (dims == NULL) return rune_dup_cstr("out of memory");

  memset(&buffer_args, 0, sizeof(buffer_args));
  buffer_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  buffer_args.client = client;
  buffer_args.data = data;
  buffer_args.type = pjrt_type;
  buffer_args.dims = dims;
  buffer_args.num_dims = Wosize_val(v_shape);
  buffer_args.host_buffer_semantics = semantics;
  buffer_args.device = device;
  error = api->PJRT_Client_BufferFromHostBuffer(&buffer_args);
  if (error != NULL) {
    free(dims);
    return rune_pjrt_error_message(api, error);
  }

  event = buffer_args.done_with_host_buffer;
  if (out_event != NULL) {
    *out_event = event;
  } else if (event != NULL) {
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

static char* rune_upload_buffer_from_string(
    const PJRT_Api* api, PJRT_Client* client, PJRT_Device* device,
    value v_dtype, value v_shape, value v_data, PJRT_Buffer** out_buffer,
    int64_t** out_dims) {
  return rune_upload_buffer(api, client, device, String_val(v_dtype), v_shape,
                            String_val(v_data), caml_string_length(v_data),
                            PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
                            out_buffer, out_dims, NULL);
}

static char* rune_upload_buffer_from_bigarray(
    const PJRT_Api* api, PJRT_Client* client, PJRT_Device* device,
    value v_dtype, value v_shape, value v_data, PJRT_Buffer** out_buffer,
    int64_t** out_dims, PJRT_HostBufferSemantics semantics,
    PJRT_Event** out_event) {
  const char* dtype = String_val(v_dtype);
  int itemsize = rune_dtype_size(dtype);
  size_t numel = rune_bigarray_numel(v_data);
  if (itemsize <= 0) {
    return rune_load_pjrt_error("unsupported input dtype", dtype);
  }
  return rune_upload_buffer(api, client, device, dtype, v_shape,
                            Caml_ba_data_val(v_data),
                            numel * (size_t)itemsize, semantics, out_buffer,
                            out_dims, out_event);
}

static char* rune_get_or_compile_exec(
    value v_plugin_path, value v_cache_key, value v_device_id,
    value v_stablehlo, value v_constant_input_dtypes,
    value v_constant_input_shapes, value v_constant_input_data,
    value v_output_dtypes, rune_exec_cache** out_cache) {
  rune_exec_cache* cache = rune_find_exec_cache(String_val(v_cache_key));
  rune_exec_cache* new_cache = NULL;
  rune_client_cache* runtime = NULL;
  const PJRT_Api* api = NULL;
  PJRT_LoadedExecutable* executable = NULL;
  PJRT_Executable* executable_view = NULL;
  PJRT_Buffer** constant_buffers = NULL;
  char* error_message = NULL;
  size_t constant_input_count = Wosize_val(v_constant_input_dtypes);
  size_t output_count = Wosize_val(v_output_dtypes);
  size_t executable_output_count = 0;
  size_t i;

  if (cache != NULL) {
    if (cache->device_id != Int_val(v_device_id) ||
        strcmp(cache->plugin_path, String_val(v_plugin_path)) != 0) {
      return rune_dup_cstr(
          "cached executable belongs to a different PJRT device");
    }
    if (cache->constant_count != constant_input_count) {
      return rune_dup_cstr("cached executable constant arity mismatch");
    }
    if (cache->output_count != output_count) {
      return rune_dup_cstr("cached executable output arity mismatch");
    }
    *out_cache = cache;
    return NULL;
  }

  error_message =
      rune_get_client(String_val(v_plugin_path), Int_val(v_device_id),
                      &runtime);
  if (error_message != NULL) goto fail;
  api = runtime->api;

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
    compile_args.client = runtime->client;
    compile_args.program = &program;
    compile_args.compile_options = k_compile_options;
    compile_args.compile_options_size = sizeof(k_compile_options);
    error = api->PJRT_Client_Compile(&compile_args);
    if (error != NULL) {
      error_message = rune_pjrt_error_message(api, error);
      goto fail;
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
      goto fail;
    }
    executable_view = get_exec_args.executable;

    memset(&num_outputs_args, 0, sizeof(num_outputs_args));
    num_outputs_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    num_outputs_args.executable = executable_view;
    error = api->PJRT_Executable_NumOutputs(&num_outputs_args);
    if (error != NULL) {
      error_message = rune_pjrt_error_message(api, error);
      goto fail;
    }
    executable_output_count = num_outputs_args.num_outputs;
    if (executable_output_count != output_count) {
      char detail[128];
      snprintf(detail, sizeof(detail),
               "expected %zu outputs, executable returns %zu", output_count,
               executable_output_count);
      error_message = rune_load_pjrt_error("output arity mismatch", detail);
      goto fail;
    }
  }

  constant_buffers =
      calloc(constant_input_count == 0 ? 1 : constant_input_count,
             sizeof(PJRT_Buffer*));
  if (constant_buffers == NULL) {
    error_message = rune_dup_cstr("out of memory");
    goto fail;
  }
  for (i = 0; i < constant_input_count; ++i) {
    error_message = rune_upload_buffer_from_string(
        api, runtime->client, runtime->device,
        Field(v_constant_input_dtypes, i),
        Field(v_constant_input_shapes, i), Field(v_constant_input_data, i),
        &constant_buffers[i], NULL);
    if (error_message != NULL) goto fail;
  }

  new_cache = calloc(1, sizeof(*new_cache));
  if (new_cache == NULL) {
    error_message = rune_dup_cstr("out of memory");
    goto fail;
  }
  new_cache->cache_key = rune_dup_cstr(String_val(v_cache_key));
  new_cache->plugin_path = rune_dup_cstr(String_val(v_plugin_path));
  new_cache->device_id = Int_val(v_device_id);
  new_cache->executable = executable;
  new_cache->executable_view = executable_view;
  new_cache->output_count = executable_output_count;
  new_cache->constant_count = constant_input_count;
  new_cache->constant_buffers = constant_buffers;
  new_cache->runtime = runtime;
  if (new_cache->cache_key == NULL || new_cache->plugin_path == NULL) {
    error_message = rune_dup_cstr("out of memory");
    goto fail;
  }

  new_cache->next = rune_exec_cache_head;
  rune_exec_cache_head = new_cache;
  *out_cache = new_cache;
  return NULL;

fail:
  if (new_cache != NULL) {
    rune_free_exec_cache_entry(new_cache);
  } else {
    rune_pjrt_destroy_buffer_array(api, constant_buffers, constant_input_count);
    rune_pjrt_destroy_executable(api, executable);
    rune_pjrt_destroy_plain_executable(api, executable_view);
  }
  return error_message != NULL ? error_message
                               : rune_dup_cstr("PJRT compilation failed");
}

static char* rune_execute_buffers(rune_exec_cache* cache,
                                  PJRT_Buffer** dynamic_input_buffers,
                                  size_t dynamic_input_count,
                                  PJRT_Buffer** output_buffers) {
  size_t total_input_count = dynamic_input_count + cache->constant_count;
  PJRT_Buffer** argument_buffers =
      calloc(total_input_count == 0 ? 1 : total_input_count,
             sizeof(PJRT_Buffer*));
  PJRT_ExecuteOptions execute_options;
  PJRT_LoadedExecutable_Execute_Args execute_args;
  PJRT_Buffer* const* argument_lists[1];
  PJRT_Buffer** output_lists[1];
  PJRT_Error* error;
  size_t i;

  if (argument_buffers == NULL) return rune_dup_cstr("out of memory");
  for (i = 0; i < dynamic_input_count; ++i) {
    argument_buffers[i] = dynamic_input_buffers[i];
  }
  for (i = 0; i < cache->constant_count; ++i) {
    argument_buffers[dynamic_input_count + i] = cache->constant_buffers[i];
  }

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
  error = cache->runtime->api->PJRT_LoadedExecutable_Execute(&execute_args);
  free(argument_buffers);
  return rune_pjrt_error_message(cache->runtime->api, error);
}

static char* rune_download_buffer(const PJRT_Api* api, PJRT_Buffer* buffer,
                                  value v_dtype, value v_shape,
                                  value v_data) {
  PJRT_Buffer_ToHostBuffer_Args copy_args;
  PJRT_Error* error;
  int itemsize = rune_dtype_size(String_val(v_dtype));
  size_t numel = rune_shape_numel(v_shape);
  size_t data_numel = rune_bigarray_numel(v_data);
  size_t rank = Wosize_val(v_shape);
  int64_t* minor_to_major = NULL;
  PJRT_Buffer_MemoryLayout host_layout;

  if (itemsize <= 0) {
    return rune_load_pjrt_error("unsupported output dtype",
                                String_val(v_dtype));
  }
  if (data_numel != numel) {
    char detail[160];
    snprintf(detail, sizeof(detail),
             "%s output buffer has %zu elements, expected %zu",
             String_val(v_dtype), data_numel, numel);
    return rune_load_pjrt_error("output buffer size mismatch", detail);
  }

  minor_to_major = rune_make_minor_to_major(rank);
  if (minor_to_major == NULL) return rune_dup_cstr("out of memory");
  memset(&host_layout, 0, sizeof(host_layout));
  host_layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
  host_layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
  host_layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
  host_layout.tiled.minor_to_major = minor_to_major;
  host_layout.tiled.minor_to_major_size = rank;

  memset(&copy_args, 0, sizeof(copy_args));
  copy_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  copy_args.src = buffer;
  copy_args.host_layout = &host_layout;
  copy_args.dst = Caml_ba_data_val(v_data);
  copy_args.dst_size = numel * (size_t)itemsize;
  error = api->PJRT_Buffer_ToHostBuffer(&copy_args);
  free(minor_to_major);
  if (error != NULL) return rune_pjrt_error_message(api, error);
  return rune_await_event(api, copy_args.event);
}

static value rune_execute(value v_plugin_path, value v_cache_key, value v_device_id,
                          value v_stablehlo, value v_dynamic_input_dtypes,
                          value v_dynamic_input_shapes,
                          value v_dynamic_input_data,
                          value v_constant_input_dtypes,
                          value v_constant_input_shapes,
                          value v_constant_input_data, value v_output_dtypes,
                          value v_output_shapes, value v_output_data) {
  CAMLparam5(v_plugin_path, v_cache_key, v_device_id, v_stablehlo,
             v_dynamic_input_dtypes);
  CAMLxparam5(v_dynamic_input_shapes, v_dynamic_input_data,
              v_constant_input_dtypes, v_constant_input_shapes,
              v_constant_input_data);
  CAMLxparam3(v_output_dtypes, v_output_shapes, v_output_data);
  rune_exec_cache* cache = NULL;
  PJRT_Buffer** dynamic_input_buffers = NULL;
  PJRT_Event** dynamic_input_events = NULL;
  PJRT_Buffer** output_buffers = NULL;
  char* error_message = NULL;
  size_t dynamic_input_count = Wosize_val(v_dynamic_input_dtypes);
  size_t constant_input_count = Wosize_val(v_constant_input_dtypes);
  size_t output_count = Wosize_val(v_output_dtypes);
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
  if (Wosize_val(v_output_shapes) != output_count ||
      Wosize_val(v_output_data) != output_count) {
    caml_invalid_argument("rune_pjrt_execute: mismatched output metadata");
  }

  error_message = rune_get_or_compile_exec(
      v_plugin_path, v_cache_key, v_device_id, v_stablehlo,
      v_constant_input_dtypes, v_constant_input_shapes, v_constant_input_data,
      v_output_dtypes, &cache);
  if (error_message != NULL) goto cleanup;

  dynamic_input_buffers =
      calloc(dynamic_input_count == 0 ? 1 : dynamic_input_count,
             sizeof(PJRT_Buffer*));
  dynamic_input_events =
      calloc(dynamic_input_count == 0 ? 1 : dynamic_input_count,
             sizeof(PJRT_Event*));
  output_buffers =
      calloc(output_count == 0 ? 1 : output_count,
             sizeof(PJRT_Buffer*));
  if (dynamic_input_buffers == NULL || dynamic_input_events == NULL ||
      output_buffers == NULL) {
    error_message = rune_dup_cstr("out of memory");
    goto cleanup;
  }

  /* The OCaml array keeps every input Bigarray rooted until this primitive
     returns. Retaining the release events until cleanup lets PJRT stage all
     inputs concurrently without outliving their host storage. */
  for (i = 0; i < dynamic_input_count; ++i) {
    error_message = rune_upload_buffer_from_bigarray(
        cache->runtime->api, cache->runtime->client, cache->runtime->device,
        Field(v_dynamic_input_dtypes, i), Field(v_dynamic_input_shapes, i),
        Field(v_dynamic_input_data, i),
        &dynamic_input_buffers[i], NULL,
        PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
        &dynamic_input_events[i]);
    if (error_message != NULL) goto cleanup;
  }

  error_message = rune_execute_buffers(
      cache, dynamic_input_buffers, dynamic_input_count, output_buffers);
  if (error_message != NULL) goto cleanup;

  for (i = 0; i < output_count; ++i) {
    error_message = rune_download_buffer(
        cache->runtime->api, output_buffers[i], Field(v_output_dtypes, i),
        Field(v_output_shapes, i), Field(v_output_data, i));
    if (error_message != NULL) goto cleanup;
  }

cleanup:
  if (cache != NULL) {
    char* event_error =
        rune_await_events(cache->runtime->api, dynamic_input_events,
                          dynamic_input_count);
    if (error_message == NULL)
      error_message = event_error;
    else
      free(event_error);
    if (output_buffers != NULL) {
      for (i = 0; i < output_count; ++i) {
        rune_pjrt_destroy_buffer(cache->runtime->api, output_buffers[i]);
      }
    }
    if (dynamic_input_buffers != NULL) {
      for (i = 0; i < dynamic_input_count; ++i) {
        rune_pjrt_destroy_buffer(cache->runtime->api,
                                 dynamic_input_buffers[i]);
      }
    }
  }

  if (error_message != NULL) {
    char* message = error_message;
    free(dynamic_input_buffers);
    free(dynamic_input_events);
    free(output_buffers);
    caml_failwith(message);
  }

  free(dynamic_input_buffers);
  free(dynamic_input_events);
  free(output_buffers);
  CAMLreturn(result);
}

CAMLprim value caml_rune_pjrt_buffer_of_host(
    value v_plugin_path, value v_device_id, value v_dtype, value v_shape,
    value v_data) {
  CAMLparam5(v_plugin_path, v_device_id, v_dtype, v_shape, v_data);
  CAMLlocal1(result);
  rune_client_cache* runtime = NULL;
  rune_device_buffer* device_buffer;
  PJRT_Buffer* buffer = NULL;
  char* error_message =
      rune_get_client(String_val(v_plugin_path), Int_val(v_device_id),
                      &runtime);
  int itemsize = rune_dtype_size(String_val(v_dtype));
  size_t dependent_memory =
      itemsize > 0 ? rune_shape_numel(v_shape) * (size_t)itemsize : 0;
  if (error_message != NULL) caml_failwith(error_message);

  result = rune_alloc_device_buffer(runtime, NULL, dependent_memory);
  device_buffer = Rune_device_buffer_val(result);
  error_message = rune_upload_buffer_from_bigarray(
      runtime->api, runtime->client, runtime->device, v_dtype, v_shape, v_data,
      &buffer, NULL, PJRT_HostBufferSemantics_kImmutableOnlyDuringCall, NULL);
  if (error_message != NULL) caml_failwith(error_message);
  device_buffer->buffer = buffer;
  CAMLreturn(result);
}

CAMLprim value caml_rune_pjrt_buffer_to_host(
    value v_buffer, value v_dtype, value v_shape, value v_data) {
  CAMLparam4(v_buffer, v_dtype, v_shape, v_data);
  rune_device_buffer* device_buffer = rune_get_device_buffer(v_buffer);
  char* error_message;
  if (device_buffer->buffer == NULL) {
    caml_invalid_argument("rune-pjrt: device buffer has been released");
  }
  error_message =
      rune_download_buffer(device_buffer->runtime->api, device_buffer->buffer,
                           v_dtype, v_shape, v_data);
  if (error_message != NULL) caml_failwith(error_message);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_rune_pjrt_buffer_await(value v_buffer) {
  CAMLparam1(v_buffer);
  rune_device_buffer* device_buffer = rune_get_device_buffer(v_buffer);
  PJRT_Buffer_ReadyEvent_Args ready_args;
  PJRT_Error* error;
  char* error_message;
  if (device_buffer->buffer == NULL) {
    caml_invalid_argument("rune-pjrt: device buffer has been released");
  }
  memset(&ready_args, 0, sizeof(ready_args));
  ready_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
  ready_args.buffer = device_buffer->buffer;
  error = device_buffer->runtime->api->PJRT_Buffer_ReadyEvent(&ready_args);
  if (error != NULL) {
    error_message =
        rune_pjrt_error_message(device_buffer->runtime->api, error);
    caml_failwith(error_message);
  }
  error_message =
      rune_await_event(device_buffer->runtime->api, ready_args.event);
  if (error_message != NULL) caml_failwith(error_message);
  CAMLreturn(Val_unit);
}

static value rune_execute_device(
    value v_plugin_path, value v_cache_key, value v_device_id,
    value v_stablehlo, value v_dynamic_input_buffers,
    value v_constant_input_dtypes, value v_constant_input_shapes,
    value v_constant_input_data, value v_output_dtypes,
    value v_output_shapes) {
  CAMLparam5(v_plugin_path, v_cache_key, v_device_id, v_stablehlo,
             v_dynamic_input_buffers);
  CAMLxparam5(v_constant_input_dtypes, v_constant_input_shapes,
              v_constant_input_data, v_output_dtypes, v_output_shapes);
  CAMLlocal2(v_outputs, v_buffer);
  rune_exec_cache* cache = NULL;
  PJRT_Buffer** dynamic_input_buffers = NULL;
  PJRT_Buffer** output_buffers = NULL;
  char* error_message = NULL;
  size_t dynamic_input_count = Wosize_val(v_dynamic_input_buffers);
  size_t constant_input_count = Wosize_val(v_constant_input_dtypes);
  size_t output_count = Wosize_val(v_output_dtypes);
  size_t i;

  if (Wosize_val(v_constant_input_shapes) != constant_input_count ||
      Wosize_val(v_constant_input_data) != constant_input_count) {
    caml_invalid_argument(
        "rune_pjrt_execute_device: mismatched constant input metadata");
  }
  if (Wosize_val(v_output_shapes) != output_count) {
    caml_invalid_argument(
        "rune_pjrt_execute_device: mismatched output metadata");
  }

  error_message = rune_get_or_compile_exec(
      v_plugin_path, v_cache_key, v_device_id, v_stablehlo,
      v_constant_input_dtypes, v_constant_input_shapes, v_constant_input_data,
      v_output_dtypes, &cache);
  if (error_message != NULL) goto cleanup;

  dynamic_input_buffers =
      calloc(dynamic_input_count == 0 ? 1 : dynamic_input_count,
             sizeof(PJRT_Buffer*));
  output_buffers =
      calloc(output_count == 0 ? 1 : output_count, sizeof(PJRT_Buffer*));
  if (dynamic_input_buffers == NULL || output_buffers == NULL) {
    error_message = rune_dup_cstr("out of memory");
    goto cleanup;
  }

  for (i = 0; i < dynamic_input_count; ++i) {
    rune_device_buffer* input =
        rune_get_device_buffer(Field(v_dynamic_input_buffers, i));
    if (input->buffer == NULL) {
      error_message = rune_dup_cstr("input device buffer has been released");
      goto cleanup;
    }
    if (input->runtime != cache->runtime) {
      error_message =
          rune_dup_cstr("input device buffer belongs to a different PJRT device");
      goto cleanup;
    }
    dynamic_input_buffers[i] = input->buffer;
  }

  v_outputs = output_count == 0 ? Atom(0) : caml_alloc_tuple(output_count);
  for (i = 0; i < output_count; ++i) {
    int itemsize = rune_dtype_size(String_val(Field(v_output_dtypes, i)));
    size_t dependent_memory =
        itemsize > 0
            ? rune_shape_numel(Field(v_output_shapes, i)) * (size_t)itemsize
            : 0;
    v_buffer =
        rune_alloc_device_buffer(cache->runtime, NULL, dependent_memory);
    Store_field(v_outputs, i, v_buffer);
  }

  error_message = rune_execute_buffers(
      cache, dynamic_input_buffers, dynamic_input_count, output_buffers);
  if (error_message != NULL) goto cleanup;

  for (i = 0; i < output_count; ++i) {
    rune_device_buffer* output =
        Rune_device_buffer_val(Field(v_outputs, i));
    output->buffer = output_buffers[i];
    output_buffers[i] = NULL;
  }

cleanup:
  if (cache != NULL && output_buffers != NULL) {
    for (i = 0; i < output_count; ++i) {
      rune_pjrt_destroy_buffer(cache->runtime->api, output_buffers[i]);
    }
  }
  free(dynamic_input_buffers);
  free(output_buffers);
  if (error_message != NULL) caml_failwith(error_message);
  CAMLreturn(v_outputs);
}

CAMLprim value caml_rune_pjrt_execute_device(
    value v_plugin_path, value v_cache_key, value v_device_id,
    value v_stablehlo, value v_dynamic_input_buffers,
    value v_constant_input_dtypes, value v_constant_input_shapes,
    value v_constant_input_data, value v_output_dtypes,
    value v_output_shapes) {
  return rune_execute_device(
      v_plugin_path, v_cache_key, v_device_id, v_stablehlo,
      v_dynamic_input_buffers, v_constant_input_dtypes,
      v_constant_input_shapes, v_constant_input_data, v_output_dtypes,
      v_output_shapes);
}

CAMLprim value caml_rune_pjrt_execute_device_bc(value* argv, int argn) {
  if (argn != 10) {
    caml_invalid_argument("rune_pjrt_execute_device: arity");
  }
  return rune_execute_device(argv[0], argv[1], argv[2], argv[3], argv[4],
                             argv[5], argv[6], argv[7], argv[8], argv[9]);
}

CAMLprim value caml_rune_pjrt_execute(
    value v_plugin_path, value v_cache_key, value v_device_id,
    value v_stablehlo, value v_dynamic_input_dtypes,
    value v_dynamic_input_shapes, value v_dynamic_input_data,
    value v_constant_input_dtypes, value v_constant_input_shapes,
    value v_constant_input_data, value v_output_dtypes,
    value v_output_shapes, value v_output_data) {
  return rune_execute(
      v_plugin_path, v_cache_key, v_device_id, v_stablehlo,
      v_dynamic_input_dtypes, v_dynamic_input_shapes, v_dynamic_input_data,
      v_constant_input_dtypes, v_constant_input_shapes, v_constant_input_data,
      v_output_dtypes, v_output_shapes, v_output_data);
}

CAMLprim value caml_rune_pjrt_execute_bc(value* argv, int argn) {
  if (argn != 13) caml_invalid_argument("rune_pjrt_execute: arity");
  return rune_execute(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                      argv[6], argv[7], argv[8], argv[9], argv[10], argv[11],
                      argv[12]);
}

#else

CAMLprim value caml_rune_pjrt_register_ffi_handler(
    value v_plugin_path, value v_library_path, value v_library_digest,
    value v_symbol, value v_target) {
  (void)v_plugin_path;
  (void)v_library_path;
  (void)v_library_digest;
  (void)v_symbol;
  (void)v_target;
  caml_failwith("rune-pjrt was built without vendor/xla available");
}

CAMLprim value caml_rune_pjrt_buffer_of_host(
    value v_plugin_path, value v_device_id, value v_dtype, value v_shape,
    value v_data) {
  (void)v_plugin_path;
  (void)v_device_id;
  (void)v_dtype;
  (void)v_shape;
  (void)v_data;
  caml_failwith("rune-pjrt was built without vendor/xla available");
}

CAMLprim value caml_rune_pjrt_buffer_to_host(
    value v_buffer, value v_dtype, value v_shape, value v_data) {
  (void)v_buffer;
  (void)v_dtype;
  (void)v_shape;
  (void)v_data;
  caml_failwith("rune-pjrt was built without vendor/xla available");
}

CAMLprim value caml_rune_pjrt_buffer_await(value v_buffer) {
  (void)v_buffer;
  caml_failwith("rune-pjrt was built without vendor/xla available");
}

CAMLprim value caml_rune_pjrt_execute_device(
    value v_plugin_path, value v_cache_key, value v_device_id,
    value v_stablehlo, value v_dynamic_input_buffers,
    value v_constant_input_dtypes, value v_constant_input_shapes,
    value v_constant_input_data, value v_output_dtypes,
    value v_output_shapes) {
  (void)v_plugin_path;
  (void)v_cache_key;
  (void)v_device_id;
  (void)v_stablehlo;
  (void)v_dynamic_input_buffers;
  (void)v_constant_input_dtypes;
  (void)v_constant_input_shapes;
  (void)v_constant_input_data;
  (void)v_output_dtypes;
  (void)v_output_shapes;
  caml_failwith("rune-pjrt was built without vendor/xla available");
}

CAMLprim value caml_rune_pjrt_execute_device_bc(value* argv, int argn) {
  (void)argv;
  (void)argn;
  caml_failwith("rune-pjrt was built without vendor/xla available");
}

CAMLprim value caml_rune_pjrt_execute(
    value v_plugin_path, value v_cache_key, value v_device_id,
    value v_stablehlo, value v_dynamic_input_dtypes,
    value v_dynamic_input_shapes, value v_dynamic_input_data,
    value v_constant_input_dtypes, value v_constant_input_shapes,
    value v_constant_input_data, value v_output_dtypes,
    value v_output_shapes, value v_output_data) {
  (void)v_plugin_path;
  (void)v_cache_key;
  (void)v_device_id;
  (void)v_stablehlo;
  (void)v_dynamic_input_dtypes;
  (void)v_dynamic_input_shapes;
  (void)v_dynamic_input_data;
  (void)v_constant_input_dtypes;
  (void)v_constant_input_shapes;
  (void)v_constant_input_data;
  (void)v_output_dtypes;
  (void)v_output_shapes;
  (void)v_output_data;
  caml_failwith("rune-pjrt was built without vendor/xla available");
}

CAMLprim value caml_rune_pjrt_execute_bc(value* argv, int argn) {
  (void)argv;
  (void)argn;
  caml_failwith("rune-pjrt was built without vendor/xla available");
}

#endif
