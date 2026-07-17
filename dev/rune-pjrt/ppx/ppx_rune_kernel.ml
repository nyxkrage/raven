(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ppxlib
open Ast_builder.Default

let kernel_attribute =
  Attribute.declare "rune.kernel.cuda" Attribute.Context.value_binding
    Ast_pattern.(pstr (pstr_eval __ nil ^:: nil))
    Fun.id

let error ~loc fmt = Location.raise_errorf ~loc ("rune.kernel.cuda: " ^^ fmt)

let longident = function
  | [] -> invalid_arg "ppx_rune_kernel: empty long identifier"
  | first :: rest ->
      List.fold_left
        (fun path name -> Longident.Ldot (path, name))
        (Longident.Lident first) rest

let ident ~loc path = pexp_ident ~loc (Located.mk ~loc (longident path))

let construct ~loc path argument =
  pexp_construct ~loc (Located.mk ~loc (longident path)) argument

let apply ~loc fn arguments = pexp_apply ~loc fn arguments
let call ~loc path arguments = apply ~loc (ident ~loc path) arguments
let variable ~loc name = evar ~loc name
let string ~loc value = estring ~loc value

let list ~loc expressions =
  List.fold_right
    (fun expression rest ->
      construct ~loc [ "::" ] (Some (pexp_tuple ~loc [ expression; rest ])))
    expressions
    (construct ~loc [ "[]" ] None)

let tensor ~loc expression =
  construct ~loc [ "Rune_pjrt"; "Ffi"; "Tensor" ] (Some expression)

type specification = {
  library : string;
  fwd : string option;
  bwd : string option;
}

let string_literal ~field expression =
  match expression.pexp_desc with
  | Pexp_constant (Pconst_string (value, _, _)) -> value
  | _ -> error ~loc:expression.pexp_loc "%s must be a literal string" field

let valid_symbol symbol =
  let valid_first = function
    | 'a' .. 'z' | 'A' .. 'Z' | '_' -> true
    | _ -> false
  in
  let valid_rest = function
    | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true
    | _ -> false
  in
  let length = String.length symbol in
  length > 0
  && valid_first symbol.[0]
  && String.for_all valid_rest (String.sub symbol 1 (length - 1))

let parse_specification expression =
  let fields =
    match expression.pexp_desc with
    | Pexp_record (fields, None) -> fields
    | _ -> error ~loc:expression.pexp_loc "payload must be a record"
  in
  let library = ref None in
  let source = ref None in
  let fwd = ref None in
  let bwd = ref None in
  let set field slot value loc =
    if Option.is_some !slot then error ~loc "duplicate %s field" field;
    slot := Some value
  in
  List.iter
    (fun ({ txt; loc }, value) ->
      let field =
        match txt with
        | Longident.Lident field -> field
        | _ -> error ~loc "field names must be unqualified"
      in
      let value = string_literal ~field value in
      match field with
      | "library" -> set field library value loc
      | "source" -> set field source value loc
      | "fwd" -> set field fwd value loc
      | "bwd" -> set field bwd value loc
      | _ -> error ~loc "unknown field %s" field)
    fields;
  if Option.is_some !source then
    error ~loc:expression.pexp_loc
      "source compilation is not implemented; build a shared library and use \
       library";
  let library =
    match !library with
    | Some library when String.trim library <> "" -> library
    | Some _ -> error ~loc:expression.pexp_loc "library must not be empty"
    | None -> error ~loc:expression.pexp_loc "missing library field"
  in
  if Option.is_none !fwd && Option.is_none !bwd then
    error ~loc:expression.pexp_loc "at least one of fwd or bwd is required";
  let validate direction = function
    | Some symbol when valid_symbol symbol -> Some symbol
    | Some symbol ->
        error ~loc:expression.pexp_loc "%s symbol %S is not a C identifier"
          direction symbol
    | None -> None
  in
  { library; fwd = validate "fwd" !fwd; bwd = validate "bwd" !bwd }

let fallback_call ~loc fallback argument =
  apply ~loc (variable ~loc fallback) [ (Nolabel, argument) ]

let pack_inputs ~loc expressions =
  list ~loc (List.map (tensor ~loc) expressions)

let fallback_thunk ~loc expression =
  pexp_fun ~loc Nolabel None (punit ~loc) expression

let make_kernel ~loc specification =
  let optional label = function
    | Some value -> [ (Labelled label, string ~loc value) ]
    | None -> []
  in
  call ~loc
    [ "Rune_pjrt"; "Ffi"; "Kernel"; "create" ]
    ([ (Labelled "library", string ~loc specification.library) ]
    @ optional "fwd" specification.fwd
    @ optional "bwd" specification.bwd
    @ [ (Nolabel, eunit ~loc) ])

let make_forward ~loc ~kernel ~fallback ~argument =
  let x = variable ~loc argument in
  let y_name = gen_symbol ~prefix:"y" () in
  let y = variable ~loc y_name in
  let dispatch =
    call ~loc
      [ "Rune_pjrt"; "Ffi"; "call_fwd" ]
      [
        (Nolabel, variable ~loc kernel);
        (Labelled "inputs", pack_inputs ~loc [ x ]);
        ( Labelled "fallback",
          fallback_thunk ~loc (fallback_call ~loc fallback x) );
      ]
  in
  let result = pexp_tuple ~loc [ y; pexp_tuple ~loc [ x; y ] ] in
  pexp_fun ~loc Nolabel None (pvar ~loc argument)
    (pexp_let ~loc Nonrecursive
       [ value_binding ~loc ~pat:(pvar ~loc y_name) ~expr:dispatch ]
       result)

let make_backward ~loc ~kernel ~fallback =
  let x_name = gen_symbol ~prefix:"x" () in
  let y_name = gen_symbol ~prefix:"y" () in
  let dy_name = gen_symbol ~prefix:"dy" () in
  let x = variable ~loc x_name in
  let y = variable ~loc y_name in
  let dy = variable ~loc dy_name in
  let residuals = ppat_tuple ~loc [ pvar ~loc x_name; pvar ~loc y_name ] in
  let fallback_vjp =
    call ~loc [ "Rune"; "vjp" ]
      [ (Nolabel, variable ~loc fallback); (Nolabel, x); (Nolabel, dy) ]
    |> fun vjp -> call ~loc [ "Stdlib"; "snd" ] [ (Nolabel, vjp) ]
  in
  let dispatch =
    call ~loc
      [ "Rune_pjrt"; "Ffi"; "call_bwd" ]
      [
        (Nolabel, variable ~loc kernel);
        (Labelled "inputs", pack_inputs ~loc [ x; y; dy ]);
        (Labelled "fallback", fallback_thunk ~loc fallback_vjp);
      ]
  in
  pexp_fun ~loc Nolabel None residuals
    (pexp_fun ~loc Nolabel None (pvar ~loc dy_name) dispatch)

let expand_binding specification binding =
  let loc = binding.pvb_loc in
  if Option.is_some binding.pvb_constraint then
    error ~loc "type-constrained bindings are not supported";
  let function_name =
    match binding.pvb_pat.ppat_desc with
    | Ppat_var { txt; _ } -> txt
    | _ ->
        error ~loc:binding.pvb_pat.ppat_loc "function name must be a variable"
  in
  let argument =
    match binding.pvb_expr.pexp_desc with
    | Pexp_function
        ( [
            {
              pparam_desc =
                Pparam_val
                  (Nolabel, None, { ppat_desc = Ppat_var { txt; _ }; _ });
              _;
            };
          ],
          None,
          Pfunction_body _ ) ->
        txt
    | _ ->
        error ~loc:binding.pvb_expr.pexp_loc
          "expected one positional argument bound to a variable"
  in
  let fallback = gen_symbol ~prefix:(function_name ^ "_fallback") () in
  let kernel = gen_symbol ~prefix:(function_name ^ "_kernel") () in
  let forward = make_forward ~loc ~kernel ~fallback ~argument in
  let backward = make_backward ~loc ~kernel ~fallback in
  let dispatch =
    call ~loc [ "Rune"; "custom_vjp" ]
      [
        (Labelled "fwd", forward);
        (Labelled "bwd", backward);
        (Nolabel, variable ~loc argument);
      ]
  in
  let expression =
    pexp_fun ~loc Nolabel None (pvar ~loc argument)
      (pexp_let ~loc Nonrecursive
         [ value_binding ~loc ~pat:(pvar ~loc fallback) ~expr:binding.pvb_expr ]
         (pexp_let ~loc Nonrecursive
            [
              value_binding ~loc ~pat:(pvar ~loc kernel)
                ~expr:(make_kernel ~loc specification);
            ]
            dispatch))
  in
  { binding with pvb_expr = expression }

let expand_item item =
  match item.pstr_desc with
  | Pstr_value (recursive, bindings) ->
      let bindings, specifications =
        List.map
          (fun binding ->
            match Attribute.consume kernel_attribute binding with
            | Some (binding, payload) ->
                (binding, Some (parse_specification payload))
            | None -> (binding, None))
          bindings
        |> List.split
      in
      if List.for_all Option.is_none specifications then item
      else (
        if recursive = Recursive then
          error ~loc:item.pstr_loc "recursive functions are not supported";
        if List.length bindings <> 1 then
          error ~loc:item.pstr_loc "let-and groups are not supported";
        match (bindings, specifications) with
        | [ binding ], [ Some specification ] ->
            {
              item with
              pstr_desc =
                Pstr_value
                  (Nonrecursive, [ expand_binding specification binding ]);
            }
        | _ -> assert false)
  | _ -> item

let rewrite_structure structure = List.map expand_item structure

let () =
  Driver.register_transformation "ppx_rune_kernel" ~impl:rewrite_structure
