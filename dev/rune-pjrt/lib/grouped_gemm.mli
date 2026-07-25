(** Grouped matrix multiplication for packed expert rows. *)

type t

val create : library:string -> ?symbol:string -> unit -> t
(** [create ~library ?symbol ()] describes a CUDA grouped GEMM handler.
    [symbol] defaults to ["raven_grouped_gemm_fwd"]. *)

val reference :
  lhs:(float, 'a) Nx.t ->
  rhs:(float, 'a) Nx.t ->
  group_sizes:Nx.int32_t ->
  (float, 'a) Nx.t
(** [reference ~lhs ~rhs ~group_sizes] computes packed grouped matrix
    multiplication using ordinary Nx operations.

    [lhs] has shape [[rows; k]], [rhs] has shape [[groups; k; n]], and
    [group_sizes] has shape [[groups]]. Rows belonging to each group are
    contiguous in [lhs], and the group sizes must be non-negative and sum to
    [rows]. The result has shape [[rows; n]]. *)

val run :
  t ->
  lhs:(float, 'a) Nx.t ->
  rhs:(float, 'a) Nx.t ->
  group_sizes:Nx.int32_t ->
  (float, 'a) Nx.t
(** [run kernel ~lhs ~rhs ~group_sizes] uses [kernel] while tracing for PJRT
    CUDA and otherwise computes {!reference}. CUDA execution supports float16,
    bfloat16, and float32 inputs. *)
