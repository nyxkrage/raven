type compiled = {
  backend : Backend.t;
  device_id : int;
  cache_key : string;
  signature : Signature.t;
  program : Ir.program;
  artifact_dir : string;
  spec_path : string;
  module_path : string;
  module_text : string;
  output_descs : Ir.desc list;
  extra_inputs : (Ir.desc * string) list;
}

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
