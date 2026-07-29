type compiled = {
  backend : Backend.t;
  device_id : int;
  plugin_path : string;
  cache_key : string;
  signature : Signature.t;
  program : Ir.program;
  artifact_dir : string;
  spec_path : string;
  module_path : string;
  module_text : string;
  output_descs : Ir.desc list;
  extra_inputs : (Ir.desc * string) list;
  mutable ffi_registered : bool;
}

type ('a, 'b) device_buffer

type packed_device_buffer =
  | Device_buffer : ('a, 'b) device_buffer -> packed_device_buffer

val is_available : unit -> bool
val backend_available : Backend.t -> bool
val status : unit -> string

val compile :
  backend:Backend.t ->
  device_id:int ->
  signature:Signature.t ->
  Ir.program ->
  Trace.packed list ->
  compiled

val compile_stablehlo :
  backend:Backend.t ->
  device_id:int ->
  signature:Signature.t ->
  module_text:string ->
  output_descs:Ir.desc list ->
  extra_inputs:(Ir.desc * string) list ->
  compiled

val data_string_of_literal : Ir.literal -> string
val execute : compiled -> ('a, 'b) Nx.t list -> Trace.packed list

val device_buffer_of_host :
  backend:Backend.t -> device_id:int -> ('a, 'b) Nx.t -> ('a, 'b) device_buffer

val device_buffer_to_host : ('a, 'b) device_buffer -> ('a, 'b) Nx.t
val device_buffer_await : ('a, 'b) device_buffer -> unit
val device_buffer_shape : ('a, 'b) device_buffer -> int array
val device_buffer_dtype : ('a, 'b) device_buffer -> ('a, 'b) Nx_core.Dtype.t
val device_buffer_placeholder : ('a, 'b) device_buffer -> ('a, 'b) Nx.t

val device_buffers_match_signature :
  Signature.t -> packed_device_buffer list -> bool

val execute_device :
  compiled -> packed_device_buffer list -> packed_device_buffer list
