#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include <stdbool.h>
#include <stddef.h>

extern char *xet_ocaml_version(void);
extern char *xet_ocaml_hash_files_json(const char *paths_json, char **error_out);
extern char *xet_ocaml_download_hf_file_json(const char *request_json,
                                             char **error_out);
extern void xet_ocaml_string_free(char *ptr);
extern void *xet_ocaml_session_create(char **error_out);
extern void xet_ocaml_session_free(void *session);
extern char *xet_ocaml_session_status(void *session, char **error_out);
extern bool xet_ocaml_session_abort(void *session, char **error_out);
extern void *xet_ocaml_session_new_file_download_group(void *session,
                                                       const char *config_json,
                                                       char **error_out);
extern void xet_ocaml_file_download_group_free(void *group);
extern void *xet_ocaml_file_download_group_start_download_file(
    void *group, const char *file_info_json, const char *dest_path,
    char **error_out);
extern char *xet_ocaml_file_download_group_wait_to_finish_json(void *group,
                                                               char **error_out);
extern bool xet_ocaml_file_download_group_abort(void *group, char **error_out);
extern char *xet_ocaml_file_download_group_progress_json(void *group,
                                                         char **error_out);
extern void xet_ocaml_file_download_free(void *download);
extern char *xet_ocaml_file_download_status(void *download, char **error_out);
extern void xet_ocaml_file_download_cancel(void *download);

#define Session_val(v) (*((void **)Data_custom_val(v)))
#define File_download_group_val(v) (*((void **)Data_custom_val(v)))
#define File_download_val(v) (*((void **)Data_custom_val(v)))

static void finalize_session(value v)
{
  void *session = Session_val(v);
  if (session != NULL) {
    xet_ocaml_session_free(session);
    Session_val(v) = NULL;
  }
}

static void finalize_file_download_group(value v)
{
  void *group = File_download_group_val(v);
  if (group != NULL) {
    xet_ocaml_file_download_group_free(group);
    File_download_group_val(v) = NULL;
  }
}

static void finalize_file_download(value v)
{
  void *download = File_download_val(v);
  if (download != NULL) {
    xet_ocaml_file_download_free(download);
    File_download_val(v) = NULL;
  }
}

static struct custom_operations session_ops = {
    "raven.xet.session",       finalize_session, custom_compare_default,
    custom_hash_default,       custom_serialize_default,
    custom_deserialize_default, custom_compare_ext_default,
    custom_fixed_length_default};

static struct custom_operations file_download_group_ops = {
    "raven.xet.file_download_group",
    finalize_file_download_group,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static struct custom_operations file_download_ops = {
    "raven.xet.file_download", finalize_file_download, custom_compare_default,
    custom_hash_default,       custom_serialize_default,
    custom_deserialize_default, custom_compare_ext_default,
    custom_fixed_length_default};

static value copy_rust_string(char *rust_string)
{
  value ocaml_string;

  if (rust_string == NULL) {
    caml_failwith("xet: Rust returned a null string");
  }

  ocaml_string = caml_copy_string(rust_string);
  xet_ocaml_string_free(rust_string);
  return ocaml_string;
}

static value copy_error_and_fail(char *error)
{
  CAMLparam0();
  CAMLlocal1(message);
  message = copy_rust_string(error);
  caml_failwith(String_val(message));
  CAMLreturn(Val_unit);
}

static value alloc_handle(struct custom_operations *ops, void *ptr)
{
  CAMLparam0();
  CAMLlocal1(v);
  v = caml_alloc_custom(ops, sizeof(void *), 0, 1);
  *((void **)Data_custom_val(v)) = ptr;
  CAMLreturn(v);
}

CAMLprim value caml_xet_version(value unit)
{
  CAMLparam1(unit);
  CAMLreturn(copy_rust_string(xet_ocaml_version()));
}

CAMLprim value caml_xet_hash_files_json(value paths_json)
{
  CAMLparam1(paths_json);
  CAMLlocal1(message);
  char *error = NULL;
  char *result = xet_ocaml_hash_files_json(String_val(paths_json), &error);

  if (result == NULL) {
    message = copy_rust_string(error);
    caml_failwith(String_val(message));
  }

  CAMLreturn(copy_rust_string(result));
}

CAMLprim value caml_xet_session_create(value unit)
{
  CAMLparam1(unit);
  char *error = NULL;
  void *session = xet_ocaml_session_create(&error);
  if (session == NULL) {
    copy_error_and_fail(error);
  }
  CAMLreturn(alloc_handle(&session_ops, session));
}

CAMLprim value caml_xet_session_status(value session)
{
  CAMLparam1(session);
  CAMLlocal1(message);
  char *error = NULL;
  char *status = xet_ocaml_session_status(Session_val(session), &error);
  if (status == NULL) {
    message = copy_rust_string(error);
    caml_failwith(String_val(message));
  }
  CAMLreturn(copy_rust_string(status));
}

CAMLprim value caml_xet_session_abort(value session)
{
  CAMLparam1(session);
  char *error = NULL;
  if (!xet_ocaml_session_abort(Session_val(session), &error)) {
    copy_error_and_fail(error);
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_xet_session_new_file_download_group(value session,
                                                        value config_json)
{
  CAMLparam2(session, config_json);
  char *error = NULL;
  void *group = xet_ocaml_session_new_file_download_group(
      Session_val(session), String_val(config_json), &error);
  if (group == NULL) {
    copy_error_and_fail(error);
  }
  CAMLreturn(alloc_handle(&file_download_group_ops, group));
}

CAMLprim value caml_xet_file_download_group_start_download_file(
    value group, value file_info_json, value dest_path)
{
  CAMLparam3(group, file_info_json, dest_path);
  char *error = NULL;
  void *download = xet_ocaml_file_download_group_start_download_file(
      File_download_group_val(group), String_val(file_info_json),
      String_val(dest_path), &error);
  if (download == NULL) {
    copy_error_and_fail(error);
  }
  CAMLreturn(alloc_handle(&file_download_ops, download));
}

CAMLprim value caml_xet_file_download_group_wait_to_finish_json(value group)
{
  CAMLparam1(group);
  CAMLlocal1(message);
  char *error = NULL;
  char *report = xet_ocaml_file_download_group_wait_to_finish_json(
      File_download_group_val(group), &error);
  if (report == NULL) {
    message = copy_rust_string(error);
    caml_failwith(String_val(message));
  }
  CAMLreturn(copy_rust_string(report));
}

CAMLprim value caml_xet_file_download_group_abort(value group)
{
  CAMLparam1(group);
  char *error = NULL;
  if (!xet_ocaml_file_download_group_abort(File_download_group_val(group),
                                           &error)) {
    copy_error_and_fail(error);
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_xet_file_download_group_progress_json(value group)
{
  CAMLparam1(group);
  CAMLlocal1(message);
  char *error = NULL;
  char *progress = xet_ocaml_file_download_group_progress_json(
      File_download_group_val(group), &error);
  if (progress == NULL) {
    message = copy_rust_string(error);
    caml_failwith(String_val(message));
  }
  CAMLreturn(copy_rust_string(progress));
}

CAMLprim value caml_xet_file_download_status(value download)
{
  CAMLparam1(download);
  CAMLlocal1(message);
  char *error = NULL;
  char *status = xet_ocaml_file_download_status(File_download_val(download),
                                                &error);
  if (status == NULL) {
    message = copy_rust_string(error);
    caml_failwith(String_val(message));
  }
  CAMLreturn(copy_rust_string(status));
}

CAMLprim value caml_xet_file_download_cancel(value download)
{
  CAMLparam1(download);
  xet_ocaml_file_download_cancel(File_download_val(download));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_xet_download_hf_file_json(value request_json)
{
  CAMLparam1(request_json);
  CAMLlocal1(message);
  char *error = NULL;
  char *result =
      xet_ocaml_download_hf_file_json(String_val(request_json), &error);

  if (result == NULL) {
    message = copy_rust_string(error);
    caml_failwith(String_val(message));
  }

  CAMLreturn(copy_rust_string(result));
}
