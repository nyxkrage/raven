(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type packed = Tensor : ('a, 'b) Nx.t -> packed

module Kernel : sig
  type t

  val create :
    name:string ->
    ir:string ->
    ?num_warps:int ->
    ?num_stages:int ->
    ?grid:int * int * int ->
    unit ->
    t
  (** [create ~name ~ir ?num_warps ?num_stages ?grid ()] creates a Triton kernel
      compiled by the XLA CUDA backend. [ir] is a complete TTIR module whose
      public function is [name]. Its pointer arguments are the packed inputs in
      order followed by the output. *)
end

val call :
  Kernel.t ->
  inputs:packed list ->
  fallback:(unit -> ('a, 'b) Nx.t) ->
  ('a, 'b) Nx.t
(** [call kernel ~inputs ~fallback] uses [kernel] while tracing for PJRT CUDA
    and otherwise evaluates [fallback]. *)

module Dsl : sig
  module Dtype : sig
    type f16 = Nx.float16_t
    type bf16 = Nx.bfloat16_t
    type f32 = Nx.float32_t
    type i1 = Nx.bool_t
    type i32 = Nx.int32_t
    type i64 = Nx.int64_t
    type 'a t

    val f16 : f16 t
    val bf16 : bf16 t
    val f32 : f32 t
    val i1 : i1 t
    val i32 : i32 t
    val i64 : i64 t
  end

  type axis = X | Y | Z

  module Value : sig
    type 'a t

    val shape : 'a t -> int array
    (** [shape value] returns the static blocked shape of [value]. *)

    val float : 'a Dtype.t -> float -> 'a t
    (** [float dtype value] creates a floating-point scalar. *)

    val int : 'a Dtype.t -> int -> 'a t
    (** [int dtype value] creates an i32 or i64 scalar. *)

    val bool : bool -> Dtype.i1 t
    val full : shape:int array -> 'a t -> 'a t
    val zeros : 'a Dtype.t -> shape:int array -> 'a t
    val program_id : axis -> Dtype.i32 t
    val num_programs : axis -> Dtype.i32 t

    val arange : start:int -> stop:int -> Dtype.i32 t
    (** [arange ~start ~stop] creates an i32 block containing the half-open
        interval. Its length must be a power of two. *)

    val expand_dims : axis:int -> 'a t -> 'a t
    val broadcast_to : shape:int array -> 'a t -> 'a t
    val reshape : shape:int array -> 'a t -> 'a t
    val permute : order:int array -> 'a t -> 'a t
    val cast : 'b Dtype.t -> 'a t -> 'b t
    val neg : 'a t -> 'a t
    val abs : 'a t -> 'a t
    val sqrt : 'a t -> 'a t
    val sqrt_rn : 'a t -> 'a t
    val exp : 'a t -> 'a t
    val exp2 : 'a t -> 'a t
    val log : 'a t -> 'a t
    val log2 : 'a t -> 'a t
    val sin : 'a t -> 'a t
    val cos : 'a t -> 'a t
    val erf : 'a t -> 'a t
    val floor : 'a t -> 'a t
    val ceil : 'a t -> 'a t
    val rsqrt : 'a t -> 'a t
    val add : 'a t -> 'a t -> 'a t
    val sub : 'a t -> 'a t -> 'a t
    val mul : 'a t -> 'a t -> 'a t
    val div : 'a t -> 'a t -> 'a t
    val div_rn : 'a t -> 'a t -> 'a t
    val rem : 'a t -> 'a t -> 'a t
    val maximum : 'a t -> 'a t -> 'a t
    val minimum : 'a t -> 'a t -> 'a t
    val bit_and : 'a t -> 'a t -> 'a t
    val bit_or : 'a t -> 'a t -> 'a t
    val bit_xor : 'a t -> 'a t -> 'a t
    val equal : 'a t -> 'a t -> Dtype.i1 t
    val not_equal : 'a t -> 'a t -> Dtype.i1 t
    val less : 'a t -> 'a t -> Dtype.i1 t
    val less_equal : 'a t -> 'a t -> Dtype.i1 t
    val greater : 'a t -> 'a t -> Dtype.i1 t
    val greater_equal : 'a t -> 'a t -> Dtype.i1 t
    val where : Dtype.i1 t -> 'a t -> 'a t -> 'a t
    val fma : 'a t -> 'a t -> 'a t -> 'a t
    val clamp : 'a t -> min:'a t -> max:'a t -> 'a t
    val sigmoid : 'a t -> 'a t
    val cdiv : 'a t -> 'a t -> 'a t
    val sum : ?keep_dims:bool -> axis:int -> 'a t -> 'a t
    val max : ?keep_dims:bool -> axis:int -> 'a t -> 'a t
    val min : ?keep_dims:bool -> axis:int -> 'a t -> 'a t
    val softmax : axis:int -> 'a t -> 'a t

    val dot : 'a t -> 'a t -> Dtype.f32 t -> Dtype.f32 t
    (** [dot lhs rhs accumulator] computes a rank-two blocked matrix product and
        adds the f32 accumulator. *)

    val range :
      start:Dtype.i32 t ->
      stop:Dtype.i32 t ->
      ?step:Dtype.i32 t ->
      init:'a t ->
      (Dtype.i32 t -> 'a t -> 'a t) ->
      'a t
    (** [range ~start ~stop ?step ~init body] emits a device loop with one
        loop-carried value. Bounds and step are scalar i32 values. *)
  end

  module Pointer : sig
    type 'a t

    val shape : 'a t -> int array

    val offset : 'a t -> Dtype.i32 Value.t -> 'a t
    (** [offset pointer offsets] performs element-wise pointer arithmetic. *)

    val load : ?mask:Dtype.i1 Value.t -> ?other:'a Value.t -> 'a t -> 'a Value.t
    (** [load ?mask ?other pointer] loads a scalar or block. [mask] and [other]
        are broadcast to the pointer shape. *)
  end

  module Statement : sig
    type t

    val store : ?mask:Dtype.i1 Value.t -> 'a Pointer.t -> 'a Value.t -> t
    (** [store ?mask pointer value] emits an element-wise store. *)

    val static_assert : bool -> string -> t
    (** [static_assert condition message] checks a specialization invariant
        while constructing TTIR. *)
  end

  module Signature : sig
    type ('body, 'call) t

    val returning :
      'output Dtype.t -> ('output Pointer.t -> Statement.t list, 'output) t
    (** [returning output] terminates a curried signature with its result dtype.
    *)

    val ( @-> ) :
      'input Dtype.t ->
      ('body, 'call) t ->
      ('input Pointer.t -> 'body, 'input -> 'call) t
    (** [input @-> rest] prepends one tensor input. The kernel builder receives
        a typed pointer and the bound kernel receives the corresponding [Nx.t].
    *)

    val f16 : Dtype.f16 Dtype.t
    val bf16 : Dtype.bf16 Dtype.t
    val f32 : Dtype.f32 Dtype.t
    val i1 : Dtype.i1 Dtype.t
    val i32 : Dtype.i32 Dtype.t
    val i64 : Dtype.i64 Dtype.t
  end

  module Config : sig
    type t

    val make : ?block_size:int -> ?num_warps:int -> ?num_stages:int -> unit -> t
    (** [make ?block_size ?num_warps ?num_stages ()] creates a validated static
        schedule. Block size and warp count must be powers of two. *)

    val block_size : t -> int
    val num_warps : t -> int
    val num_stages : t -> int
  end

  module Syntax : sig
    val ( ~- ) : 'a Value.t -> 'a Value.t
    val ( + ) : 'a Value.t -> 'a Value.t -> 'a Value.t
    val ( - ) : 'a Value.t -> 'a Value.t -> 'a Value.t
    val ( * ) : 'a Value.t -> 'a Value.t -> 'a Value.t
    val ( / ) : 'a Value.t -> 'a Value.t -> 'a Value.t
    val ( +: ) : Dtype.i32 Value.t -> int -> Dtype.i32 Value.t
    val ( -: ) : Dtype.i32 Value.t -> int -> Dtype.i32 Value.t
    val ( *: ) : Dtype.i32 Value.t -> int -> Dtype.i32 Value.t
    val ( /: ) : Dtype.i32 Value.t -> int -> Dtype.i32 Value.t
    val ( +@ ) : 'a Pointer.t -> Dtype.i32 Value.t -> 'a Pointer.t

    module Operand : sig
      (** Contextually typed operands emitted by [ppx_rune_kernel]. End-user
          kernel code should construct them through [fun%rune.kernel]. *)

      type 'a t

      val value : 'a Value.t -> 'a t
      val int : int -> 'a t
      val float : float -> 'a t
      val bool : bool -> Dtype.i1 t
      val neg : 'a t -> 'a Value.t
      val add : 'a t -> 'a t -> 'a Value.t
      val sub : 'a t -> 'a t -> 'a Value.t
      val mul : 'a t -> 'a t -> 'a Value.t
      val div : 'a t -> 'a t -> 'a Value.t
      val rem : 'a t -> 'a t -> 'a Value.t
      val bit_and : 'a t -> 'a t -> 'a Value.t
      val bit_or : 'a t -> 'a t -> 'a Value.t
      val bit_xor : 'a t -> 'a t -> 'a Value.t
      val equal : 'a t -> 'a t -> Dtype.i1 Value.t
      val not_equal : 'a t -> 'a t -> Dtype.i1 Value.t
      val less : 'a t -> 'a t -> Dtype.i1 Value.t
      val less_equal : 'a t -> 'a t -> Dtype.i1 Value.t
      val greater : 'a t -> 'a t -> Dtype.i1 Value.t
      val greater_equal : 'a t -> 'a t -> Dtype.i1 Value.t
      val not_ : Dtype.i1 t -> Dtype.i1 Value.t
    end
  end

  module Spec : sig
    type t

    val input_count : t -> int
    val input_shape : t -> int -> int array
    val output_shape : t -> int array
    val input_numel : t -> int -> int
    val output_numel : t -> int
  end

  module Kernel : sig
    type ('body, 'call) t

    val define :
      name:string ->
      signature:('body, 'call) Signature.t ->
      ?config:Config.t ->
      ?guard:(Spec.t -> bool) ->
      grid:(Spec.t -> int * int * int) ->
      (Spec.t -> 'body) ->
      ('body, 'call) t
    (** [define ~name ~signature ?config ?guard ~grid build] defines a
        shape-specialized blocked kernel. [signature] determines the curried
        input-pointer and output-pointer arguments received after [Spec.t]. *)

    val to_ttir_for :
      ('body, 'call) t ->
      input_shapes:int array list ->
      output_shape:int array ->
      string
    (** [to_ttir_for kernel ~input_shapes ~output_shape] renders an arbitrary
        shape-specialized kernel. *)

    val bind : fallback:'call -> ('body, 'call) t -> 'call
    (** [bind ~fallback kernel] returns an ordinary function. It converts typed
        tensor arguments and executes a compatible specialization while tracing
        PJRT CUDA. Eager execution, transformations, other backends, failed
        guards, and empty outputs use the definition-site [fallback]. *)
  end

  val static_range :
    start:int -> stop:int -> ?step:int -> init:'a -> ('a -> int -> 'a) -> 'a
  (** [static_range ~start ~stop ?step ~init body] folds an OCaml-known range
      while constructing a specialization, producing unrolled TTIR. *)
end

module Internal : sig
  type kernel = {
    name : string;
    ir : string;
    num_warps : int;
    num_stages : int;
    grid_x : int;
    grid_y : int;
    grid_z : int;
  }

  type ('a, 'b) request = {
    kernel : kernel;
    inputs : packed list;
    fallback : unit -> ('a, 'b) Nx.t;
  }

  type ('a, 'b) decision = Use_kernel of ('a, 'b) Nx.t | Use_fallback
  type _ Effect.t += Call : ('a, 'b) request -> ('a, 'b) decision Effect.t
end
