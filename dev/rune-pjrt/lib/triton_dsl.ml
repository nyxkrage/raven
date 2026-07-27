(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module type Runtime = sig
  type packed
  type raw_kernel

  val raw_kernel :
    name:string ->
    ir:string ->
    num_warps:int ->
    num_stages:int ->
    grid:int * int * int ->
    raw_kernel

  val raw_call :
    raw_kernel ->
    inputs:packed list ->
    fallback:(unit -> ('a, 'b) Nx.t) ->
    ('a, 'b) Nx.t

  val pack_tensor : ('a, 'b) Nx.t -> packed
  val packed_shape : packed -> int array
end

module Make (Runtime : Runtime) = struct
  module Dtype = struct
    type f16 = Nx.float16_t
    type bf16 = Nx.bfloat16_t
    type f32 = Nx.float32_t
    type i1 = Nx.bool_t
    type i32 = Nx.int32_t
    type i64 = Nx.int64_t

    type _ t =
      | Float16 : f16 t
      | Bfloat16 : bf16 t
      | Float32 : f32 t
      | Int1 : i1 t
      | Int32 : i32 t
      | Int64 : i64 t

    type packed = Dtype : 'a t -> packed

    let f16 = Float16
    let bf16 = Bfloat16
    let f32 = Float32
    let i1 = Int1
    let i32 = Int32
    let i64 = Int64
  end

  type dtype_kind = Boolean | Integer | Float

  let dtype_name (Dtype.Dtype dtype) =
    match dtype with
    | Dtype.Float16 -> "f16"
    | Bfloat16 -> "bf16"
    | Float32 -> "f32"
    | Int1 -> "i1"
    | Int32 -> "i32"
    | Int64 -> "i64"

  let dtype_kind (Dtype.Dtype dtype) =
    match dtype with
    | Dtype.Float16 | Bfloat16 | Float32 -> Float
    | Int1 -> Boolean
    | Int32 | Int64 -> Integer

  let dtype_width (Dtype.Dtype dtype) =
    match dtype with
    | Dtype.Int1 -> 1
    | Float16 | Bfloat16 -> 16
    | Float32 | Int32 -> 32
    | Int64 -> 64

  let same_dtype lhs rhs = String.equal (dtype_name lhs) (dtype_name rhs)

  let require_dtype function_name expected actual =
    if not (same_dtype expected actual) then
      invalid_arg
        (Printf.sprintf "Rune_pjrt.Triton.Dsl.%s: expected %s, received %s"
           function_name (dtype_name expected) (dtype_name actual))

  let require_kind function_name expected dtype =
    if dtype_kind dtype <> expected then
      let expected =
        match expected with
        | Boolean -> "a Boolean"
        | Integer -> "an integer"
        | Float -> "a floating-point"
      in
      invalid_arg
        (Printf.sprintf
           "Rune_pjrt.Triton.Dsl.%s: expected %s value, received %s"
           function_name expected (dtype_name dtype))

  let valid_identifier name =
    let valid_first = function
      | 'a' .. 'z' | 'A' .. 'Z' | '_' -> true
      | _ -> false
    in
    let valid_rest = function
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true
      | _ -> false
    in
    let length = String.length name in
    length > 0
    && valid_first name.[0]
    && String.for_all valid_rest (String.sub name 1 (length - 1))

  let positive ~function_name name value =
    if value <= 0 then
      invalid_arg
        (Printf.sprintf "Rune_pjrt.Triton.Dsl.%s: %s must be positive"
           function_name name)

  let power_of_two ~function_name name value =
    positive ~function_name name value;
    if value land (value - 1) <> 0 then
      invalid_arg
        (Printf.sprintf "Rune_pjrt.Triton.Dsl.%s: %s must be a power of two"
           function_name name)

  let shape_numel ~function_name shape =
    let total = ref 1 in
    Array.iter
      (fun dimension ->
        positive ~function_name "shape dimension" dimension;
        if !total > max_int / dimension then
          invalid_arg
            (Printf.sprintf
               "Rune_pjrt.Triton.Dsl.%s: shape element count overflows"
               function_name);
        total := !total * dimension)
      shape;
    !total

  let validate_block_shape ~function_name shape =
    let numel = shape_numel ~function_name shape in
    if Array.length shape > 0 && numel land (numel - 1) <> 0 then
      invalid_arg
        (Printf.sprintf
           "Rune_pjrt.Triton.Dsl.%s: block element count must be a power of two"
           function_name)

  let same_shape lhs rhs =
    Array.length lhs = Array.length rhs
    &&
    let equal = ref true in
    let index = ref 0 in
    while !equal && !index < Array.length lhs do
      equal := lhs.(!index) = rhs.(!index);
      incr index
    done;
    !equal

  let shape_string shape =
    if Array.length shape = 0 then "scalar"
    else Array.to_list shape |> List.map string_of_int |> String.concat "x"

  let tensor_type shape element =
    if Array.length shape = 0 then element
    else Printf.sprintf "tensor<%sx%s>" (shape_string shape) element

  let broadcast_shape ~function_name lhs rhs =
    let rank = Int.max (Array.length lhs) (Array.length rhs) in
    let result = Array.make rank 1 in
    for output_axis = 0 to rank - 1 do
      let lhs_axis = output_axis - (rank - Array.length lhs) in
      let rhs_axis = output_axis - (rank - Array.length rhs) in
      let lhs_dimension = if lhs_axis < 0 then 1 else lhs.(lhs_axis) in
      let rhs_dimension = if rhs_axis < 0 then 1 else rhs.(rhs_axis) in
      if
        lhs_dimension <> rhs_dimension
        && lhs_dimension <> 1 && rhs_dimension <> 1
      then
        invalid_arg
          (Printf.sprintf "Rune_pjrt.Triton.Dsl.%s: cannot broadcast %s with %s"
             function_name (shape_string lhs) (shape_string rhs));
      result.(output_axis) <- Int.max lhs_dimension rhs_dimension
    done;
    validate_block_shape ~function_name result;
    result

  let next_scope = ref 0

  let fresh_scope () =
    let scope = !next_scope in
    incr next_scope;
    scope

  let merge_scope function_name lhs rhs =
    match (lhs, rhs) with
    | None, scope | scope, None -> scope
    | Some lhs, Some rhs when lhs = rhs -> Some lhs
    | Some _, Some _ ->
        invalid_arg
          (Printf.sprintf
             "Rune_pjrt.Triton.Dsl.%s: values belong to different kernels"
             function_name)

  type axis = X | Y | Z

  type unary =
    | Neg
    | Abs
    | Sqrt
    | Precise_sqrt
    | Exp
    | Exp2
    | Log
    | Log2
    | Sin
    | Cos
    | Erf
    | Floor
    | Ceil
    | Rsqrt

  type binary =
    | Add
    | Sub
    | Mul
    | Div
    | Precise_div
    | Rem
    | Maximum
    | Minimum
    | Bit_and
    | Bit_or
    | Bit_xor

  type comparison = Eq | Ne | Lt | Le | Gt | Ge
  type reduction = Sum | Max | Min

  type expression = {
    dtype : Dtype.packed;
    shape : int array;
    scope : int option;
    node : expression_node;
  }

  and expression_node =
    | Float_constant of float
    | Int_constant of int
    | Bool_constant of bool
    | Program_id of axis
    | Num_programs of axis
    | Make_range of int * int
    | Splat of expression
    | Expand_dims of expression * int
    | Broadcast of expression
    | Reshape of expression
    | Permute of expression * int array
    | Cast of expression
    | Unary of unary * expression
    | Binary of binary * expression * expression
    | Compare of comparison * expression * expression
    | Select of expression * expression * expression
    | Fma of expression * expression * expression
    | Load of pointer * expression option * expression option
    | Reduce of reduction * int * expression
    | Dot of expression * expression * expression
    | Loop_index of int
    | Loop_carried of int
    | For of loop

  and loop = {
    loop_id : int;
    start : expression;
    stop : expression;
    step : expression;
    initial : expression;
    body : expression;
  }

  and pointer = {
    element : Dtype.packed;
    shape : int array;
    scope : int option;
    node : pointer_node;
  }

  and pointer_node =
    | Pointer_argument of int
    | Pointer_broadcast of pointer
    | Pointer_offset of pointer * expression

  type statement =
    | Store of pointer * expression * expression option
    | Static_assert of bool * string

  let make_expression ?scope dtype shape node =
    validate_block_shape ~function_name:"Value" shape;
    { dtype; shape = Array.copy shape; scope; node }

  let make_pointer ?scope element shape node =
    validate_block_shape ~function_name:"Pointer" shape;
    { element; shape = Array.copy shape; scope; node }

  let require_same_dtype function_name lhs rhs =
    require_dtype function_name lhs.dtype rhs.dtype

  let require_same_shape function_name expected actual =
    if not (same_shape expected actual) then
      invalid_arg
        (Printf.sprintf
           "Rune_pjrt.Triton.Dsl.%s: expected block shape %s, received %s"
           function_name (shape_string expected) (shape_string actual))

  let rec broadcast_to expression shape =
    validate_block_shape ~function_name:"Value.broadcast_to" shape;
    if same_shape expression.shape shape then expression
    else if Array.length expression.shape = 0 then
      make_expression ?scope:expression.scope expression.dtype shape
        (Splat expression)
    else (
      if Array.length expression.shape > Array.length shape then
        invalid_arg
          "Rune_pjrt.Triton.Dsl.Value.broadcast_to: cannot reduce rank";
      let expanded = ref expression in
      for _ = Array.length expression.shape to Array.length shape - 1 do
        let next_shape =
          Array.init
            (Array.length !expanded.shape + 1)
            (fun axis -> if axis = 0 then 1 else !expanded.shape.(axis - 1))
        in
        expanded :=
          make_expression ?scope:!expanded.scope !expanded.dtype next_shape
            (Expand_dims (!expanded, 0))
      done;
      Array.iteri
        (fun axis dimension ->
          let source = !expanded.shape.(axis) in
          if source <> dimension && source <> 1 then
            invalid_arg
              (Printf.sprintf
                 "Rune_pjrt.Triton.Dsl.Value.broadcast_to: dimension %d cannot \
                  expand from %d to %d"
                 axis source dimension))
        shape;
      make_expression ?scope:!expanded.scope !expanded.dtype shape
        (Broadcast !expanded))

  let broadcast_values function_name lhs rhs =
    require_same_dtype function_name lhs rhs;
    let shape = broadcast_shape ~function_name lhs.shape rhs.shape in
    (broadcast_to lhs shape, broadcast_to rhs shape, shape)

  let broadcast_condition condition shape =
    require_kind "Value.where" Boolean condition.dtype;
    broadcast_to condition shape

  let rec broadcast_pointer_to (pointer : pointer) shape =
    validate_block_shape ~function_name:"Pointer.broadcast_to" shape;
    if same_shape pointer.shape shape then pointer
    else if Array.length pointer.shape = 0 then
      make_pointer ?scope:pointer.scope pointer.element shape
        (Pointer_broadcast pointer)
    else
      invalid_arg
        (Printf.sprintf
           "Rune_pjrt.Triton.Dsl.Pointer.broadcast_to: cannot broadcast %s to \
            %s"
           (shape_string pointer.shape)
           (shape_string shape))

  module Value = struct
    type 'a t = expression

    let shape value = Array.copy value.shape

    let float dtype value =
      let packed = Dtype.Dtype dtype in
      require_kind "Value.float" Float packed;
      if not (Float.is_finite value) then
        invalid_arg "Rune_pjrt.Triton.Dsl.Value.float: value must be finite";
      make_expression packed [||] (Float_constant value)

    let int (type a) (dtype : a Dtype.t) value =
      let packed = Dtype.Dtype dtype in
      require_kind "Value.int" Integer packed;
      (match dtype with
      | Dtype.Int32 ->
          if
            value < Int32.to_int Int32.min_int
            || value > Int32.to_int Int32.max_int
          then
            invalid_arg
              "Rune_pjrt.Triton.Dsl.Value.int: value is outside the i32 range"
      | Int64 -> ()
      | _ -> assert false);
      make_expression packed [||] (Int_constant value)

    let bool value =
      make_expression (Dtype.Dtype Dtype.i1) [||] (Bool_constant value)

    let full ~shape value = broadcast_to value shape

    let zeros dtype ~shape =
      match dtype_kind (Dtype.Dtype dtype) with
      | Float -> full ~shape (float dtype 0.)
      | Integer -> full ~shape (int dtype 0)
      | Boolean -> full ~shape (bool false)

    let program_id axis =
      make_expression (Dtype.Dtype Dtype.i32) [||] (Program_id axis)

    let num_programs axis =
      make_expression (Dtype.Dtype Dtype.i32) [||] (Num_programs axis)

    let arange ~start ~stop =
      if start < 0 then
        invalid_arg
          "Rune_pjrt.Triton.Dsl.Value.arange: start must be non-negative";
      if stop <= start then
        invalid_arg
          "Rune_pjrt.Triton.Dsl.Value.arange: stop must be greater than start";
      let length = stop - start in
      power_of_two ~function_name:"Value.arange" "range length" length;
      make_expression (Dtype.Dtype Dtype.i32) [| length |]
        (Make_range (start, stop))

    let expand_dims ~axis value =
      if axis < 0 || axis > Array.length value.shape then
        invalid_arg
          "Rune_pjrt.Triton.Dsl.Value.expand_dims: axis is out of bounds";
      let shape =
        Array.init
          (Array.length value.shape + 1)
          (fun output_axis ->
            if output_axis < axis then value.shape.(output_axis)
            else if output_axis = axis then 1
            else value.shape.(output_axis - 1))
      in
      make_expression ?scope:value.scope value.dtype shape
        (if Array.length value.shape = 0 then Splat value
         else Expand_dims (value, axis))

    let broadcast_to ~shape value = broadcast_to value shape

    let reshape ~shape value =
      validate_block_shape ~function_name:"Value.reshape" shape;
      let input_numel =
        shape_numel ~function_name:"Value.reshape" value.shape
      in
      let output_numel = shape_numel ~function_name:"Value.reshape" shape in
      if input_numel <> output_numel then
        invalid_arg "Rune_pjrt.Triton.Dsl.Value.reshape: element counts differ";
      make_expression ?scope:value.scope value.dtype shape (Reshape value)

    let permute ~order value =
      let rank = Array.length value.shape in
      if Array.length order <> rank then
        invalid_arg
          "Rune_pjrt.Triton.Dsl.Value.permute: order rank does not match value";
      let seen = Array.make rank false in
      Array.iter
        (fun axis ->
          if axis < 0 || axis >= rank || seen.(axis) then
            invalid_arg
              "Rune_pjrt.Triton.Dsl.Value.permute: order is not a permutation";
          seen.(axis) <- true)
        order;
      let shape = Array.map (fun axis -> value.shape.(axis)) order in
      make_expression ?scope:value.scope value.dtype shape
        (Permute (value, Array.copy order))

    let cast dtype value =
      let dtype = Dtype.Dtype dtype in
      if dtype_kind value.dtype = Float && dtype_kind dtype = Boolean then
        invalid_arg
          "Rune_pjrt.Triton.Dsl.Value.cast: float-to-Boolean is unsupported";
      if same_dtype dtype value.dtype then
        make_expression ?scope:value.scope dtype value.shape value.node
      else make_expression ?scope:value.scope dtype value.shape (Cast value)

    let unary_float name operation value =
      require_kind name Float value.dtype;
      make_expression ?scope:value.scope value.dtype value.shape
        (Unary (operation, value))

    let neg = unary_float "Value.neg" Neg
    let abs = unary_float "Value.abs" Abs
    let sqrt = unary_float "Value.sqrt" Sqrt
    let sqrt_rn = unary_float "Value.sqrt_rn" Precise_sqrt
    let exp = unary_float "Value.exp" Exp
    let exp2 = unary_float "Value.exp2" Exp2
    let log = unary_float "Value.log" Log
    let log2 = unary_float "Value.log2" Log2
    let sin = unary_float "Value.sin" Sin
    let cos = unary_float "Value.cos" Cos
    let erf = unary_float "Value.erf" Erf
    let floor = unary_float "Value.floor" Floor
    let ceil = unary_float "Value.ceil" Ceil
    let rsqrt = unary_float "Value.rsqrt" Rsqrt

    let binary_numeric name operation lhs rhs =
      let lhs, rhs, shape = broadcast_values name lhs rhs in
      (match dtype_kind lhs.dtype with
      | Boolean ->
          invalid_arg
            (Printf.sprintf
               "Rune_pjrt.Triton.Dsl.%s: Boolean arithmetic is undefined" name)
      | Integer | Float -> ());
      let scope = merge_scope name lhs.scope rhs.scope in
      make_expression ?scope lhs.dtype shape (Binary (operation, lhs, rhs))

    let add = binary_numeric "Value.add" Add
    let sub = binary_numeric "Value.sub" Sub
    let mul = binary_numeric "Value.mul" Mul
    let div = binary_numeric "Value.div" Div
    let rem = binary_numeric "Value.rem" Rem
    let maximum = binary_numeric "Value.maximum" Maximum
    let minimum = binary_numeric "Value.minimum" Minimum

    let div_rn lhs rhs =
      require_kind "Value.div_rn" Float lhs.dtype;
      require_kind "Value.div_rn" Float rhs.dtype;
      binary_numeric "Value.div_rn" Precise_div lhs rhs

    let bitwise name operation lhs rhs =
      let lhs, rhs, shape = broadcast_values name lhs rhs in
      (match dtype_kind lhs.dtype with
      | Float ->
          invalid_arg
            (Printf.sprintf
               "Rune_pjrt.Triton.Dsl.%s: floating-point bitwise operation is \
                undefined"
               name)
      | Boolean | Integer -> ());
      let scope = merge_scope name lhs.scope rhs.scope in
      make_expression ?scope lhs.dtype shape (Binary (operation, lhs, rhs))

    let bit_and = bitwise "Value.bit_and" Bit_and
    let bit_or = bitwise "Value.bit_or" Bit_or
    let bit_xor = bitwise "Value.bit_xor" Bit_xor

    let compare name comparison lhs rhs =
      let lhs, rhs, shape = broadcast_values name lhs rhs in
      if dtype_kind lhs.dtype = Boolean && comparison <> Eq && comparison <> Ne
      then
        invalid_arg
          (Printf.sprintf
             "Rune_pjrt.Triton.Dsl.%s: Boolean values only support equality"
             name);
      let scope = merge_scope name lhs.scope rhs.scope in
      make_expression ?scope (Dtype.Dtype Dtype.i1) shape
        (Compare (comparison, lhs, rhs))

    let equal = compare "Value.equal" Eq
    let not_equal = compare "Value.not_equal" Ne
    let less = compare "Value.less" Lt
    let less_equal = compare "Value.less_equal" Le
    let greater = compare "Value.greater" Gt
    let greater_equal = compare "Value.greater_equal" Ge

    let where condition if_true if_false =
      let if_true, if_false, shape =
        broadcast_values "Value.where" if_true if_false
      in
      let condition = broadcast_condition condition shape in
      let scope =
        merge_scope "Value.where"
          (merge_scope "Value.where" if_true.scope if_false.scope)
          condition.scope
      in
      make_expression ?scope if_true.dtype shape
        (Select (condition, if_true, if_false))

    let fma x y z =
      require_kind "Value.fma" Float x.dtype;
      let x, y, shape = broadcast_values "Value.fma" x y in
      let z = broadcast_to ~shape z in
      require_same_dtype "Value.fma" x z;
      let scope =
        merge_scope "Value.fma"
          (merge_scope "Value.fma" x.scope y.scope)
          z.scope
      in
      make_expression ?scope x.dtype shape (Fma (x, y, z))

    let clamp value ~min ~max = minimum (maximum value min) max

    let sigmoid value =
      require_kind "Value.sigmoid" Float value.dtype;
      let one = make_expression value.dtype [||] (Float_constant 1.) in
      div one (add one (exp (neg value)))

    let cdiv lhs rhs =
      require_kind "Value.cdiv" Integer lhs.dtype;
      require_kind "Value.cdiv" Integer rhs.dtype;
      let one = make_expression lhs.dtype [||] (Int_constant 1) in
      div (add lhs (sub rhs one)) rhs

    let reduce name reduction ~axis ~keep_dims value =
      let rank = Array.length value.shape in
      if rank = 0 then
        invalid_arg
          (Printf.sprintf "Rune_pjrt.Triton.Dsl.%s: cannot reduce a scalar" name);
      if axis < 0 || axis >= rank then
        invalid_arg
          (Printf.sprintf "Rune_pjrt.Triton.Dsl.%s: axis is out of bounds" name);
      let shape =
        Array.init (rank - 1) (fun output_axis ->
            if output_axis < axis then value.shape.(output_axis)
            else value.shape.(output_axis + 1))
      in
      let reduced =
        make_expression ?scope:value.scope value.dtype shape
          (Reduce (reduction, axis, value))
      in
      if keep_dims then expand_dims ~axis reduced else reduced

    let sum ?(keep_dims = false) ~axis value =
      reduce "Value.sum" Sum ~axis ~keep_dims value

    let max ?(keep_dims = false) ~axis value =
      reduce "Value.max" Max ~axis ~keep_dims value

    let min ?(keep_dims = false) ~axis value =
      reduce "Value.min" Min ~axis ~keep_dims value

    let softmax ~axis value =
      require_kind "Value.softmax" Float value.dtype;
      let maximum = max ~keep_dims:true ~axis value in
      let numerator = exp (sub value maximum) in
      div numerator (sum ~keep_dims:true ~axis numerator)

    let dot lhs rhs accumulator =
      require_kind "Value.dot" Float lhs.dtype;
      require_kind "Value.dot" Float rhs.dtype;
      require_same_dtype "Value.dot" lhs rhs;
      require_dtype "Value.dot" (Dtype.Dtype Dtype.f32) accumulator.dtype;
      if
        Array.length lhs.shape <> 2
        || Array.length rhs.shape <> 2
        || Array.length accumulator.shape <> 2
      then
        invalid_arg
          "Rune_pjrt.Triton.Dsl.Value.dot: operands must be rank-two blocks";
      let m = lhs.shape.(0) in
      let k = lhs.shape.(1) in
      let rhs_k = rhs.shape.(0) in
      let n = rhs.shape.(1) in
      if k <> rhs_k || not (same_shape accumulator.shape [| m; n |]) then
        invalid_arg
          "Rune_pjrt.Triton.Dsl.Value.dot: matrix dimensions are incompatible";
      let scope =
        merge_scope "Value.dot"
          (merge_scope "Value.dot" lhs.scope rhs.scope)
          accumulator.scope
      in
      make_expression ?scope (Dtype.Dtype Dtype.f32) [| m; n |]
        (Dot (lhs, rhs, accumulator))

    let next_loop = ref 0

    let range ~start ~stop ?step ~init body =
      let step =
        match step with Some step -> step | None -> int Dtype.i32 1
      in
      List.iter
        (fun (name, value) ->
          require_dtype ("Value.range " ^ name) (Dtype.Dtype Dtype.i32)
            value.dtype;
          require_same_shape ("Value.range " ^ name) [||] value.shape)
        [ ("start", start); ("stop", stop); ("step", step) ];
      (match step.node with
      | Int_constant value when value <= 0 ->
          invalid_arg "Rune_pjrt.Triton.Dsl.Value.range: step must be positive"
      | _ -> ());
      let loop_id = !next_loop in
      incr next_loop;
      let index =
        make_expression (Dtype.Dtype Dtype.i32) [||] (Loop_index loop_id)
      in
      let carried =
        make_expression ?scope:init.scope init.dtype init.shape
          (Loop_carried loop_id)
      in
      let result = body index carried in
      require_same_dtype "Value.range" init result;
      require_same_shape "Value.range" init.shape result.shape;
      let scope =
        merge_scope "Value.range"
          (merge_scope "Value.range"
             (merge_scope "Value.range" start.scope stop.scope)
             step.scope)
          (merge_scope "Value.range" init.scope result.scope)
      in
      make_expression ?scope init.dtype init.shape
        (For { loop_id; start; stop; step; initial = init; body = result })
  end

  module Pointer = struct
    type 'a t = pointer

    let shape (pointer : pointer) = Array.copy pointer.shape

    let offset (pointer : pointer) (offsets : expression) =
      require_dtype "Pointer.offset" (Dtype.Dtype Dtype.i32) offsets.dtype;
      let shape =
        broadcast_shape ~function_name:"Pointer.offset" pointer.shape
          offsets.shape
      in
      let pointer = broadcast_pointer_to pointer shape in
      let offsets = broadcast_to offsets shape in
      let scope = merge_scope "Pointer.offset" pointer.scope offsets.scope in
      make_pointer ?scope pointer.element shape
        (Pointer_offset (pointer, offsets))

    let load ?mask ?other (pointer : pointer) =
      let mask =
        Option.map
          (fun mask ->
            require_kind "Pointer.load" Boolean mask.dtype;
            broadcast_to mask pointer.shape)
          mask
      in
      let other =
        Option.map
          (fun other ->
            require_dtype "Pointer.load" pointer.element other.dtype;
            broadcast_to other pointer.shape)
          other
      in
      let scope =
        let scope =
          match mask with
          | None -> pointer.scope
          | Some mask -> merge_scope "Pointer.load" pointer.scope mask.scope
        in
        match other with
        | None -> scope
        | Some other -> merge_scope "Pointer.load" scope other.scope
      in
      make_expression ?scope pointer.element pointer.shape
        (Load (pointer, mask, other))
  end

  module Arguments = struct
    type t = { pointers : pointer array; dtypes : Dtype.packed array }

    let get arguments index dtype =
      if index < 0 || index >= Array.length arguments.pointers then
        invalid_arg
          (Printf.sprintf
             "Rune_pjrt.Triton.Dsl.Arguments.get: input %d is out of bounds"
             index);
      let expected = arguments.dtypes.(index) in
      let actual = Dtype.Dtype dtype in
      require_dtype "Arguments.get" expected actual;
      arguments.pointers.(index)
  end

  module Signature = struct
    type (_, _) t =
      | Returning :
          'output Dtype.t
          -> ('output Pointer.t -> statement list, 'output) t
      | Input :
          'input Dtype.t * ('body, 'call) t
          -> ('input Pointer.t -> 'body, 'input -> 'call) t

    let returning output = Returning output
    let ( @-> ) input rest = Input (input, rest)
    let f16 = Dtype.f16
    let bf16 = Dtype.bf16
    let f32 = Dtype.f32
    let i1 = Dtype.i1
    let i32 = Dtype.i32
    let i64 = Dtype.i64

    let rec packed_dtypes : type body call. (body, call) t -> Dtype.packed list
        = function
      | Returning _ -> []
      | Input (dtype, rest) -> Dtype.Dtype dtype :: packed_dtypes rest

    let rec output : type body call. (body, call) t -> Dtype.packed = function
      | Returning dtype -> Dtype.Dtype dtype
      | Input (_, rest) -> output rest

    let rec apply : type body call.
        (body, call) t ->
        body ->
        Arguments.t ->
        int ->
        pointer ->
        statement list * int =
     fun signature body arguments index output ->
      match signature with
      | Returning _ -> (body output, index)
      | Input (dtype, rest) ->
          let pointer = Arguments.get arguments index dtype in
          apply rest (body pointer) arguments (index + 1) output

    let pack_tensor : type tensor. tensor Dtype.t -> tensor -> Runtime.packed =
     fun dtype tensor ->
      match dtype with
      | Dtype.Float16 -> Runtime.pack_tensor tensor
      | Bfloat16 -> Runtime.pack_tensor tensor
      | Float32 -> Runtime.pack_tensor tensor
      | Int1 -> Runtime.pack_tensor tensor
      | Int32 -> Runtime.pack_tensor tensor
      | Int64 -> Runtime.pack_tensor tensor
  end

  module Statement = struct
    type t = statement

    let store ?mask pointer value =
      require_dtype "Statement.store" pointer.element value.dtype;
      let value = broadcast_to value pointer.shape in
      let mask =
        Option.map
          (fun mask ->
            require_kind "Statement.store" Boolean mask.dtype;
            broadcast_to mask pointer.shape)
          mask
      in
      let value_scope =
        merge_scope "Statement.store" pointer.scope value.scope
      in
      (match mask with
      | None -> ()
      | Some mask ->
          ignore (merge_scope "Statement.store" value_scope mask.scope));
      Store (pointer, value, mask)

    let static_assert condition message = Static_assert (condition, message)
  end

  module Config = struct
    type t = { block_size : int; num_warps : int; num_stages : int }

    let make ?(block_size = 256) ?(num_warps = 4) ?(num_stages = 1) () =
      power_of_two ~function_name:"Config.make" "block_size" block_size;
      power_of_two ~function_name:"Config.make" "num_warps" num_warps;
      positive ~function_name:"Config.make" "num_stages" num_stages;
      { block_size; num_warps; num_stages }

    let block_size config = config.block_size
    let num_warps config = config.num_warps
    let num_stages config = config.num_stages
  end

  module Syntax = struct
    let ( ~- ) = Value.neg
    let ( + ) = Value.add
    let ( - ) = Value.sub
    let ( * ) = Value.mul
    let ( / ) = Value.div
    let ( +: ) value scalar = Value.add value (Value.int Dtype.i32 scalar)
    let ( -: ) value scalar = Value.sub value (Value.int Dtype.i32 scalar)
    let ( *: ) value scalar = Value.mul value (Value.int Dtype.i32 scalar)
    let ( /: ) value scalar = Value.div value (Value.int Dtype.i32 scalar)
    let ( +@ ) pointer offsets = Pointer.offset pointer offsets

    module Operand = struct
      type 'a t =
        | Value of expression
        | Int of int
        | Float of float
        | Bool of bool

      let value value = Value value
      let int value = Int value
      let float value = Float value
      let bool value = Bool value

      let fail_unanchored function_name =
        invalid_arg
          (Printf.sprintf
             "Rune_pjrt.Triton.Dsl.Syntax.Operand.%s: an operation on literals \
              needs a staged value to determine its dtype"
             function_name)

      let constant_like function_name reference = function
        | Value value -> value
        | Int value -> (
            match dtype_kind reference.dtype with
            | Boolean ->
                invalid_arg
                  (Printf.sprintf
                     "Rune_pjrt.Triton.Dsl.Syntax.Operand.%s: an integer \
                      literal cannot be used with a Boolean value"
                     function_name)
            | Integer ->
                (match reference.dtype with
                | Dtype.Dtype Dtype.Int32 ->
                    if
                      value < Int32.to_int Int32.min_int
                      || value > Int32.to_int Int32.max_int
                    then
                      invalid_arg
                        (Printf.sprintf
                           "Rune_pjrt.Triton.Dsl.Syntax.Operand.%s: integer \
                            literal is outside the i32 range"
                           function_name)
                | Dtype.Dtype Dtype.Int64 -> ()
                | _ -> assert false);
                make_expression reference.dtype [||] (Int_constant value)
            | Float ->
                make_expression reference.dtype [||]
                  (Float_constant (Float.of_int value)))
        | Float value ->
            require_kind
              ("Syntax.Operand." ^ function_name)
              Float reference.dtype;
            if not (Float.is_finite value) then
              invalid_arg
                (Printf.sprintf
                   "Rune_pjrt.Triton.Dsl.Syntax.Operand.%s: floating-point \
                    literal must be finite"
                   function_name);
            make_expression reference.dtype [||] (Float_constant value)
        | Bool value ->
            require_kind
              ("Syntax.Operand." ^ function_name)
              Boolean reference.dtype;
            make_expression reference.dtype [||] (Bool_constant value)

      let operands function_name lhs rhs =
        match (lhs, rhs) with
        | Value lhs, rhs -> (lhs, constant_like function_name lhs rhs)
        | lhs, Value rhs -> (constant_like function_name rhs lhs, rhs)
        | _ -> fail_unanchored function_name

      let unary function_name operation = function
        | Value value -> operation value
        | Int _ | Float _ | Bool _ -> fail_unanchored function_name

      let binary function_name operation lhs rhs =
        let lhs, rhs = operands function_name lhs rhs in
        operation lhs rhs

      let neg (operand : 'a t) : 'a Value.t = unary "neg" Value.neg operand

      let add (lhs : 'a t) (rhs : 'a t) : 'a Value.t =
        binary "add" Value.add lhs rhs

      let sub (lhs : 'a t) (rhs : 'a t) : 'a Value.t =
        binary "sub" Value.sub lhs rhs

      let mul (lhs : 'a t) (rhs : 'a t) : 'a Value.t =
        binary "mul" Value.mul lhs rhs

      let div (lhs : 'a t) (rhs : 'a t) : 'a Value.t =
        binary "div" Value.div lhs rhs

      let rem (lhs : 'a t) (rhs : 'a t) : 'a Value.t =
        binary "rem" Value.rem lhs rhs

      let bit_and (lhs : 'a t) (rhs : 'a t) : 'a Value.t =
        binary "bit_and" Value.bit_and lhs rhs

      let bit_or (lhs : 'a t) (rhs : 'a t) : 'a Value.t =
        binary "bit_or" Value.bit_or lhs rhs

      let bit_xor (lhs : 'a t) (rhs : 'a t) : 'a Value.t =
        binary "bit_xor" Value.bit_xor lhs rhs

      let equal (lhs : 'a t) (rhs : 'a t) : Dtype.i1 Value.t =
        binary "equal" Value.equal lhs rhs

      let not_equal (lhs : 'a t) (rhs : 'a t) : Dtype.i1 Value.t =
        binary "not_equal" Value.not_equal lhs rhs

      let less (lhs : 'a t) (rhs : 'a t) : Dtype.i1 Value.t =
        binary "less" Value.less lhs rhs

      let less_equal (lhs : 'a t) (rhs : 'a t) : Dtype.i1 Value.t =
        binary "less_equal" Value.less_equal lhs rhs

      let greater (lhs : 'a t) (rhs : 'a t) : Dtype.i1 Value.t =
        binary "greater" Value.greater lhs rhs

      let greater_equal (lhs : 'a t) (rhs : 'a t) : Dtype.i1 Value.t =
        binary "greater_equal" Value.greater_equal lhs rhs

      let not_ (operand : Dtype.i1 t) : Dtype.i1 Value.t =
        let value = unary "not" Fun.id operand in
        Value.equal value
          (make_expression (Dtype.Dtype Dtype.i1) [||] (Bool_constant false))
    end
  end

  module Spec = struct
    type t = { input_shapes : int array array; output_shape : int array }

    let input_count spec = Array.length spec.input_shapes

    let input_shape spec index =
      if index < 0 || index >= Array.length spec.input_shapes then
        invalid_arg
          (Printf.sprintf
             "Rune_pjrt.Triton.Dsl.Spec.input_shape: input %d is out of bounds"
             index);
      Array.copy spec.input_shapes.(index)

    let output_shape spec = Array.copy spec.output_shape

    let input_numel spec index =
      shape_numel ~function_name:"Spec.input_numel" (input_shape spec index)

    let output_numel spec =
      shape_numel ~function_name:"Spec.output_numel" spec.output_shape
  end

  module Expression_table = Hashtbl.Make (struct
    type t = expression

    let equal = ( = )
    let hash = Hashtbl.hash
  end)

  module Pointer_table = Hashtbl.Make (struct
    type t = pointer

    let equal = ( = )
    let hash = Hashtbl.hash
  end)

  type emitter = {
    body : Buffer.t;
    expressions : string Expression_table.t;
    pointers : string Pointer_table.t;
    mutable next_value : int;
    loop_bindings : (int * bool, string) Hashtbl.t;
  }

  let add_line emitter indent line =
    Buffer.add_string emitter.body (String.make indent ' ');
    Buffer.add_string emitter.body line;
    Buffer.add_char emitter.body '\n'

  let fresh_value emitter =
    let value = Printf.sprintf "%%value%d" emitter.next_value in
    emitter.next_value <- emitter.next_value + 1;
    value

  let axis_name = function X -> "x" | Y -> "y" | Z -> "z"

  let value_type (expression : expression) =
    tensor_type expression.shape (dtype_name expression.dtype)

  let pointer_type (pointer : pointer) =
    tensor_type pointer.shape
      (Printf.sprintf "!tt.ptr<%s, 1>" (dtype_name pointer.element))

  let unary_operation = function
    | Neg -> "arith.negf"
    | Abs -> "math.absf"
    | Sqrt -> "math.sqrt"
    | Precise_sqrt -> "tt.precise_sqrt"
    | Exp -> "math.exp"
    | Exp2 -> "math.exp2"
    | Log -> "math.log"
    | Log2 -> "math.log2"
    | Sin -> "math.sin"
    | Cos -> "math.cos"
    | Erf -> "math.erf"
    | Floor -> "math.floor"
    | Ceil -> "math.ceil"
    | Rsqrt -> "math.rsqrt"

  let binary_operation operation dtype =
    match (operation, dtype_kind dtype) with
    | Add, Float -> "arith.addf"
    | Add, Integer -> "arith.addi"
    | Sub, Float -> "arith.subf"
    | Sub, Integer -> "arith.subi"
    | Mul, Float -> "arith.mulf"
    | Mul, Integer -> "arith.muli"
    | Div, Float -> "arith.divf"
    | Div, Integer -> "arith.divsi"
    | Precise_div, Float -> "tt.precise_divf"
    | Rem, Float -> "arith.remf"
    | Rem, Integer -> "arith.remsi"
    | Maximum, Float -> "arith.maximumf"
    | Maximum, Integer -> "arith.maxsi"
    | Minimum, Float -> "arith.minimumf"
    | Minimum, Integer -> "arith.minsi"
    | Bit_and, (Boolean | Integer) -> "arith.andi"
    | Bit_or, (Boolean | Integer) -> "arith.ori"
    | Bit_xor, (Boolean | Integer) -> "arith.xori"
    | _ -> invalid_arg "Rune_pjrt.Triton.Dsl: invalid binary operation"

  let comparison_predicate comparison dtype =
    match dtype_kind dtype with
    | Float -> (
        match comparison with
        | Eq -> ("arith.cmpf", "oeq")
        | Ne -> ("arith.cmpf", "one")
        | Lt -> ("arith.cmpf", "olt")
        | Le -> ("arith.cmpf", "ole")
        | Gt -> ("arith.cmpf", "ogt")
        | Ge -> ("arith.cmpf", "oge"))
    | Boolean | Integer -> (
        match comparison with
        | Eq -> ("arith.cmpi", "eq")
        | Ne -> ("arith.cmpi", "ne")
        | Lt -> ("arith.cmpi", "slt")
        | Le -> ("arith.cmpi", "sle")
        | Gt -> ("arith.cmpi", "sgt")
        | Ge -> ("arith.cmpi", "sge"))

  let cast_operation source target =
    match (dtype_kind source, dtype_kind target) with
    | Float, Float ->
        if dtype_width source < dtype_width target then "arith.extf"
        else "arith.truncf"
    | Integer, Integer ->
        if dtype_width source < dtype_width target then "arith.extsi"
        else "arith.trunci"
    | Boolean, Integer -> "arith.extui"
    | Boolean, Float -> "arith.uitofp"
    | Integer, Float -> "arith.sitofp"
    | Float, Integer -> "arith.fptosi"
    | Integer, Boolean -> "arith.trunci"
    | Float, Boolean -> "float_to_bool"
    | Boolean, Boolean -> assert false

  let reduction_operation reduction dtype =
    match (reduction, dtype_kind dtype) with
    | Sum, Float -> "arith.addf"
    | Sum, Integer -> "arith.addi"
    | Max, Float -> "arith.maximumf"
    | Max, Integer -> "arith.maxsi"
    | Min, Float -> "arith.minimumf"
    | Min, Integer -> "arith.minsi"
    | _, Boolean ->
        invalid_arg "Rune_pjrt.Triton.Dsl: Boolean reduction is unsupported"

  let rec emit_pointer emitter pointer =
    match Pointer_table.find_opt emitter.pointers pointer with
    | Some value -> value
    | None ->
        let value =
          match pointer.node with
          | Pointer_argument index -> Printf.sprintf "%%arg%d" index
          | Pointer_broadcast input ->
              let input_name = emit_pointer emitter input in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = tt.splat %s : %s -> %s" result input_name
                   (pointer_type input) (pointer_type pointer));
              result
          | Pointer_offset (base, offsets) ->
              let base_name = emit_pointer emitter base in
              let offsets_name = emit_expression emitter offsets in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = tt.addptr %s, %s : %s, %s" result
                   base_name offsets_name (pointer_type base)
                   (value_type offsets));
              result
        in
        Pointer_table.add emitter.pointers pointer value;
        value

  and emit_expression emitter expression =
    match Expression_table.find_opt emitter.expressions expression with
    | Some value -> value
    | None ->
        let value =
          match expression.node with
          | Float_constant constant ->
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = arith.constant %.9e : %s" result constant
                   (dtype_name expression.dtype));
              result
          | Int_constant constant ->
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = arith.constant %d : %s" result constant
                   (dtype_name expression.dtype));
              result
          | Bool_constant constant ->
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = arith.constant %s : i1" result
                   (if constant then "true" else "false"));
              result
          | Program_id axis ->
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = tt.get_program_id %s : i32" result
                   (axis_name axis));
              result
          | Num_programs axis ->
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = tt.get_num_programs %s : i32" result
                   (axis_name axis));
              result
          | Make_range (start, stop) ->
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf
                   "%s = tt.make_range {end = %d : i32, start = %d : i32} : %s"
                   result stop start (value_type expression));
              result
          | Splat input ->
              let input_name = emit_expression emitter input in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = tt.splat %s : %s -> %s" result input_name
                   (value_type input) (value_type expression));
              result
          | Expand_dims (input, axis) ->
              let input_name = emit_expression emitter input in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf
                   "%s = tt.expand_dims %s {axis = %d : i32} : %s -> %s" result
                   input_name axis (value_type input) (value_type expression));
              result
          | Broadcast input ->
              let input_name = emit_expression emitter input in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = tt.broadcast %s : %s -> %s" result
                   input_name (value_type input) (value_type expression));
              result
          | Reshape input ->
              let input_name = emit_expression emitter input in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = tt.reshape %s : %s -> %s" result
                   input_name (value_type input) (value_type expression));
              result
          | Permute (input, order) ->
              let input_name = emit_expression emitter input in
              let result = fresh_value emitter in
              let order =
                Array.to_list order |> List.map string_of_int
                |> String.concat ", "
              in
              add_line emitter 4
                (Printf.sprintf
                   "%s = tt.trans %s {order = array<i32: %s>} : %s -> %s" result
                   input_name order (value_type input) (value_type expression));
              result
          | Cast input -> emit_cast emitter expression input
          | Unary (operation, input) ->
              let input_name = emit_expression emitter input in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = %s %s : %s" result
                   (unary_operation operation)
                   input_name (value_type expression));
              result
          | Binary (operation, lhs, rhs) ->
              let lhs_name = emit_expression emitter lhs in
              let rhs_name = emit_expression emitter rhs in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = %s %s, %s : %s" result
                   (binary_operation operation expression.dtype)
                   lhs_name rhs_name (value_type expression));
              result
          | Compare (comparison, lhs, rhs) ->
              let lhs_name = emit_expression emitter lhs in
              let rhs_name = emit_expression emitter rhs in
              let operation, predicate =
                comparison_predicate comparison lhs.dtype
              in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = %s %s, %s, %s : %s" result operation
                   predicate lhs_name rhs_name (value_type lhs));
              result
          | Select (condition, if_true, if_false) ->
              let condition_name = emit_expression emitter condition in
              let true_name = emit_expression emitter if_true in
              let false_name = emit_expression emitter if_false in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = arith.select %s, %s, %s : %s, %s" result
                   condition_name true_name false_name (value_type condition)
                   (value_type expression));
              result
          | Fma (x, y, z) ->
              let x_name = emit_expression emitter x in
              let y_name = emit_expression emitter y in
              let z_name = emit_expression emitter z in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = math.fma %s, %s, %s : %s" result x_name
                   y_name z_name (value_type expression));
              result
          | Load (pointer, mask, other) ->
              let pointer_name = emit_pointer emitter pointer in
              let mask_name = Option.map (emit_expression emitter) mask in
              let other_name = Option.map (emit_expression emitter) other in
              let operands =
                match (mask_name, other_name) with
                | None, None -> pointer_name
                | Some mask, None -> pointer_name ^ ", " ^ mask
                | Some mask, Some other ->
                    pointer_name ^ ", " ^ mask ^ ", " ^ other
                | None, Some _ -> assert false
              in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = tt.load %s : %s" result operands
                   (pointer_type pointer));
              result
          | Reduce (reduction, axis, input) ->
              emit_reduce emitter expression reduction axis input
          | Dot (lhs, rhs, accumulator) ->
              let lhs_name = emit_expression emitter lhs in
              let rhs_name = emit_expression emitter rhs in
              let accumulator_name = emit_expression emitter accumulator in
              let result = fresh_value emitter in
              add_line emitter 4
                (Printf.sprintf "%s = tt.dot %s, %s, %s : %s * %s -> %s" result
                   lhs_name rhs_name accumulator_name (value_type lhs)
                   (value_type rhs) (value_type expression));
              result
          | Loop_index loop_id -> (
              match Hashtbl.find_opt emitter.loop_bindings (loop_id, false) with
              | Some value -> value
              | None ->
                  invalid_arg
                    "Rune_pjrt.Triton.Dsl: loop index escaped its range")
          | Loop_carried loop_id -> (
              match Hashtbl.find_opt emitter.loop_bindings (loop_id, true) with
              | Some value -> value
              | None ->
                  invalid_arg
                    "Rune_pjrt.Triton.Dsl: loop value escaped its range")
          | For loop -> emit_for emitter expression loop
        in
        Expression_table.add emitter.expressions expression value;
        value

  and emit_cast emitter expression input =
    let input_name = emit_expression emitter input in
    if
      dtype_kind input.dtype = Float
      && dtype_kind expression.dtype = Float
      && dtype_width input.dtype = dtype_width expression.dtype
    then (
      let intermediate_type = tensor_type input.shape "f32" in
      let intermediate = fresh_value emitter in
      add_line emitter 4
        (Printf.sprintf "%s = arith.extf %s : %s to %s" intermediate input_name
           (value_type input) intermediate_type);
      let result = fresh_value emitter in
      add_line emitter 4
        (Printf.sprintf "%s = arith.truncf %s : %s to %s" result intermediate
           intermediate_type (value_type expression));
      result)
    else
      match cast_operation input.dtype expression.dtype with
      | "float_to_bool" -> assert false
      | operation ->
          let result = fresh_value emitter in
          add_line emitter 4
            (Printf.sprintf "%s = %s %s : %s to %s" result operation input_name
               (value_type input) (value_type expression));
          result

  and emit_reduce emitter expression reduction axis input =
    let input_name = emit_expression emitter input in
    let result = fresh_value emitter in
    let lhs = fresh_value emitter in
    let rhs = fresh_value emitter in
    let combined = fresh_value emitter in
    let element = dtype_name input.dtype in
    add_line emitter 4
      (Printf.sprintf "%s = \"tt.reduce\"(%s) <{axis = %d : i32}> ({" result
         input_name axis);
    add_line emitter 6
      (Printf.sprintf "^bb0(%s: %s, %s: %s):" lhs element rhs element);
    add_line emitter 8
      (Printf.sprintf "%s = %s %s, %s : %s" combined
         (reduction_operation reduction input.dtype)
         lhs rhs element);
    add_line emitter 8
      (Printf.sprintf "tt.reduce.return %s : %s" combined element);
    add_line emitter 4
      (Printf.sprintf "}) : (%s) -> %s" (value_type input)
         (value_type expression));
    result

  and emit_for emitter expression loop =
    let start = emit_expression emitter loop.start in
    let stop = emit_expression emitter loop.stop in
    let step = emit_expression emitter loop.step in
    let initial = emit_expression emitter loop.initial in
    let result = fresh_value emitter in
    let index = fresh_value emitter in
    let carried = fresh_value emitter in
    add_line emitter 4
      (Printf.sprintf
         "%s = scf.for %s = %s to %s step %s iter_args(%s = %s) -> (%s) : i32 {"
         result index start stop step carried initial (value_type expression));
    let nested_body = Buffer.create 1024 in
    let nested_bindings = Hashtbl.copy emitter.loop_bindings in
    Hashtbl.replace nested_bindings (loop.loop_id, false) index;
    Hashtbl.replace nested_bindings (loop.loop_id, true) carried;
    let nested =
      {
        body = nested_body;
        expressions = Expression_table.copy emitter.expressions;
        pointers = Pointer_table.copy emitter.pointers;
        next_value = emitter.next_value;
        loop_bindings = nested_bindings;
      }
    in
    let yielded = emit_expression nested loop.body in
    emitter.next_value <- nested.next_value;
    Buffer.contents nested_body
    |> String.split_on_char '\n'
    |> List.iter (fun line ->
        if not (String.equal line "") then (
          Buffer.add_string emitter.body "    ";
          Buffer.add_string emitter.body line;
          Buffer.add_char emitter.body '\n'));
    add_line emitter 8
      (Printf.sprintf "scf.yield %s : %s" yielded (value_type expression));
    add_line emitter 4 "}";
    result

  let emit_statement emitter = function
    | Store (pointer, value, mask) ->
        let pointer_name = emit_pointer emitter pointer in
        let value_name = emit_expression emitter value in
        let mask =
          match mask with
          | None -> ""
          | Some mask -> ", " ^ emit_expression emitter mask
        in
        add_line emitter 4
          (Printf.sprintf "tt.store %s, %s%s : %s" pointer_name value_name mask
             (pointer_type pointer))
    | Static_assert (condition, message) ->
        if not condition then
          invalid_arg ("Rune_pjrt.Triton.Dsl static assertion: " ^ message)

  let render_module ~name ~input_dtypes ~output_dtype ~statements =
    let body = Buffer.create 4096 in
    let emitter =
      {
        body;
        expressions = Expression_table.create 64;
        pointers = Pointer_table.create 32;
        next_value = 0;
        loop_bindings = Hashtbl.create 4;
      }
    in
    List.iter (emit_statement emitter) statements;
    add_line emitter 4 "tt.return";
    let arguments =
      Array.to_list input_dtypes @ [ output_dtype ]
      |> List.mapi (fun index dtype ->
          Printf.sprintf "%%arg%d: !tt.ptr<%s, 1>" index (dtype_name dtype))
      |> String.concat ", "
    in
    Printf.sprintf "module {\n  tt.func public @%s(%s) {\n%s  }\n}\n" name
      arguments (Buffer.contents body)

  type ('body, 'call) kernel = {
    name : string;
    signature : ('body, 'call) Signature.t;
    input_dtypes : Dtype.packed array;
    output_dtype : Dtype.packed;
    config : Config.t;
    scope : int;
    guard : Spec.t -> bool;
    grid : Spec.t -> int * int * int;
    build : Spec.t -> Arguments.t -> pointer -> Statement.t list;
    cache : (string, Runtime.raw_kernel) Hashtbl.t;
  }

  let validate_name function_name name =
    if not (valid_identifier name) then
      invalid_arg
        (Printf.sprintf
           "Rune_pjrt.Triton.Dsl.%s: name must be an OCaml-style identifier"
           function_name)

  let make_arguments scope input_dtypes =
    let pointers =
      Array.mapi
        (fun index dtype ->
          make_pointer ~scope dtype [||] (Pointer_argument index))
        input_dtypes
    in
    Arguments.{ pointers; dtypes = input_dtypes }

  let make_output_pointer scope input_count output_dtype =
    make_pointer ~scope output_dtype [||] (Pointer_argument input_count)

  let make_spec input_shapes result_shape =
    let validate shape =
      Array.iter
        (fun dimension ->
          if dimension < 0 then
            invalid_arg
              "Rune_pjrt.Triton.Dsl.Kernel: tensor dimensions must be \
               non-negative")
        shape
    in
    List.iter validate input_shapes;
    validate result_shape;
    Spec.
      {
        input_shapes = Array.of_list (List.map Array.copy input_shapes);
        output_shape = Array.copy result_shape;
      }

  let specialization_key spec =
    let shapes =
      Array.to_list spec.Spec.input_shapes @ [ spec.output_shape ]
      |> List.map shape_string
    in
    String.concat ";" shapes

  module Kernel = struct
    type ('body, 'call) t = ('body, 'call) kernel

    let define ~name ~signature ?(config = Config.make ())
        ?(guard = fun _ -> true) ~grid build =
      validate_name "Kernel.define" name;
      let scope = fresh_scope () in
      let input_dtypes = Signature.packed_dtypes signature |> Array.of_list in
      let output_dtype = Signature.output signature in
      let build spec arguments output_pointer =
        let statements, count =
          Signature.apply signature (build spec) arguments 0 output_pointer
        in
        if count <> Array.length input_dtypes then
          invalid_arg
            "Rune_pjrt.Triton.Dsl.Kernel.define: invalid typed signature";
        statements
      in
      {
        name;
        signature;
        input_dtypes;
        output_dtype;
        config;
        scope;
        guard;
        grid;
        build;
        cache = Hashtbl.create 4;
      }

    let render kernel spec =
      let arguments = make_arguments kernel.scope kernel.input_dtypes in
      let output =
        make_output_pointer kernel.scope
          (Array.length kernel.input_dtypes)
          kernel.output_dtype
      in
      let statements = kernel.build spec arguments output in
      render_module ~name:kernel.name ~input_dtypes:kernel.input_dtypes
        ~output_dtype:kernel.output_dtype ~statements

    let to_ttir_for kernel ~input_shapes ~output_shape =
      if List.length input_shapes <> Array.length kernel.input_dtypes then
        invalid_arg
          (Printf.sprintf
             "Rune_pjrt.Triton.Dsl.Kernel.to_ttir_for: expected %d input \
              shapes, received %d"
             (Array.length kernel.input_dtypes)
             (List.length input_shapes));
      let spec = make_spec input_shapes output_shape in
      if Spec.output_numel spec = 0 then
        invalid_arg
          "Rune_pjrt.Triton.Dsl.Kernel.to_ttir_for: output must not be empty";
      render kernel spec

    let grid kernel spec =
      let grid = kernel.grid spec in
      let x, y, z = grid in
      positive ~function_name:"Kernel.grid" "grid x" x;
      positive ~function_name:"Kernel.grid" "grid y" y;
      positive ~function_name:"Kernel.grid" "grid z" z;
      grid

    let eligible kernel spec = kernel.guard spec

    let raw kernel spec =
      let key = specialization_key spec in
      match Hashtbl.find_opt kernel.cache key with
      | Some raw -> raw
      | None ->
          let ir = render kernel spec in
          let raw =
            Runtime.raw_kernel ~name:kernel.name ~ir
              ~num_warps:(Config.num_warps kernel.config)
              ~num_stages:(Config.num_stages kernel.config)
              ~grid:(grid kernel spec)
          in
          Hashtbl.add kernel.cache key raw;
          raw

    let execute : type output.
        output Dtype.t ->
        Runtime.raw_kernel ->
        Runtime.packed list ->
        output ->
        output =
     fun dtype raw inputs output ->
      match dtype with
      | Dtype.Float16 ->
          Runtime.raw_call raw ~inputs ~fallback:(fun () -> output)
      | Bfloat16 -> Runtime.raw_call raw ~inputs ~fallback:(fun () -> output)
      | Float32 -> Runtime.raw_call raw ~inputs ~fallback:(fun () -> output)
      | Int1 -> Runtime.raw_call raw ~inputs ~fallback:(fun () -> output)
      | Int32 -> Runtime.raw_call raw ~inputs ~fallback:(fun () -> output)
      | Int64 -> Runtime.raw_call raw ~inputs ~fallback:(fun () -> output)

    let dispatch kernel dtype inputs output =
      let output_packed = Signature.pack_tensor dtype output in
      let input_shapes = List.map Runtime.packed_shape inputs in
      let output_shape = Runtime.packed_shape output_packed in
      let spec = make_spec input_shapes output_shape in
      if Array.exists (( = ) 0) output_shape || not (eligible kernel spec) then
        output
      else
        let raw = raw kernel spec in
        execute dtype raw inputs output

    let rec bind_signature : type body call kernel_body kernel_call.
        (body, call) Signature.t ->
        (kernel_body, kernel_call) kernel ->
        call ->
        Runtime.packed list ->
        call =
     fun signature kernel fallback inputs ->
      match signature with
      | Signature.Returning dtype ->
          dispatch kernel dtype (List.rev inputs) fallback
      | Input (dtype, rest) ->
          fun tensor ->
            bind_signature rest kernel (fallback tensor)
              (Signature.pack_tensor dtype tensor :: inputs)

    let bind ~fallback kernel =
      bind_signature kernel.signature kernel fallback []
  end

  let static_range ~start ~stop ?(step = 1) ~init body =
    positive ~function_name:"static_range" "step" step;
    let rec loop state index =
      if index >= stop then state else loop (body state index) (index + step)
    in
    loop init start
end
