(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_ir
module P = Program
module K = Kernel

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)

let int32_to_bytes values =
  let bytes = Bytes.create (List.length values * 4) in
  let set off value =
    let open Int32 in
    Bytes.set bytes off (Char.chr (to_int (logand value 0xFFl)));
    Bytes.set bytes (off + 1)
      (Char.chr (to_int (logand (shift_right_logical value 8) 0xFFl)));
    Bytes.set bytes (off + 2)
      (Char.chr (to_int (logand (shift_right_logical value 16) 0xFFl)));
    Bytes.set bytes (off + 3)
      (Char.chr (to_int (logand (shift_right_logical value 24) 0xFFl)))
  in
  List.iteri (fun i value -> set (i * 4) (Int32.of_int value)) values;
  bytes

let int32_list_of_bytes bytes =
  let len = Bytes.length bytes / 4 in
  let get off =
    let open Int32 in
    logor
      (of_int (Char.code (Bytes.get bytes off)))
      (logor
         (shift_left (of_int (Char.code (Bytes.get bytes (off + 1)))) 8)
         (logor
            (shift_left (of_int (Char.code (Bytes.get bytes (off + 2)))) 16)
            (shift_left (of_int (Char.code (Bytes.get bytes (off + 3)))) 24)))
  in
  List.init len (fun i -> Int32.to_int (get (i * 4)))

let cpu name = Tolk_cpu.create ("CPU:" ^ name)

let create_i32_buffer device values =
  let buf =
    Device.create_buffer ~size:(List.length values) ~dtype:Dtype.int32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (int32_to_bytes values);
  buf

let read_i32_buffer buf = Device.Buffer.as_bytes buf |> int32_list_of_bytes

let increment_program () =
  let dt = Dtype.Val.int32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let idx_src = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx_dst = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let l0 = P.emit b (Load { src = idx_src; alt = None; dtype = dt }) in
  let c1 = P.emit b (Const { value = Const.int dt 1; dtype = dt }) in
  let sum = P.emit b (Binary { op = `Add; lhs = l0; rhs = c1; dtype = dt }) in
  let _ = P.emit b (Store { dst = idx_dst; value = sum }) in
  P.finish b

let core_id_program ~threads =
  let dt = Dtype.Val.int32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let dv = P.emit b (Define_var { name = "core_id"; lo = 0; hi = threads - 1; dtype = dt }) in
  let idx = P.emit b (Index { ptr = p0; idxs = [ dv ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx; value = dv }) in
  P.finish b

let run_spec device spec bufs =
  let car = Realize.Compiled_runner.create ~device spec in
  ignore (Realize.Compiled_runner.call car bufs [] ~wait:true ~timeout:None);
  Device.synchronize device

let () =
  run "Cpu_runtime"
    [
      group "Execution"
        [
          test "compile and run one kernel" (fun () ->
            let device = cpu "run-one" in
            let spec =
              Device.compile_program device ~name:"add_one"
                (increment_program ())
            in
            let dst = create_i32_buffer device [ 0 ] in
            let src = create_i32_buffer device [ 41 ] in
            run_spec device spec [ dst; src ];
            equal (list int) [ 42 ] (read_i32_buffer dst));
          test "exec is ordered" (fun () ->
            let device = cpu "ordered" in
            let spec =
              Device.compile_program device ~name:"ordered_add_one"
                (increment_program ())
            in
            let a = create_i32_buffer device [ 0 ] in
            let b = create_i32_buffer device [ 0 ] in
            run_spec device spec [ b; a ];
            run_spec device spec [ a; b ];
            equal (list int) [ 2 ] (read_i32_buffer a);
            equal (list int) [ 1 ] (read_i32_buffer b));
          test "core_id drives parallel execution" (fun () ->
            let device = cpu "core-id" in
            let threads = 4 in
            let spec =
              Device.compile_program device ~name:"write_core_id"
                (core_id_program ~threads)
            in
            let dst = create_i32_buffer device [ 0; 0; 0; 0 ] in
            run_spec device spec [ dst ];
            equal (list int) [ 0; 1; 2; 3 ] (read_i32_buffer dst));
          test "runner cache keeps device instance" (fun () ->
            let device1 = cpu "same-name-runner" in
            let device2 = cpu "same-name-runner" in
            let calls = ref 0 in
            let get_program _ =
              incr calls;
              Device.compile_program device1 ~name:"cached_add_one"
                (increment_program ())
            in
            let ast = K.sink [] in
            let runner1 = Realize.get_runner ~device:device1 ~get_program ast in
            let runner2 = Realize.get_runner ~device:device2 ~get_program ast in
            let runner2_again =
              Realize.get_runner ~device:device2 ~get_program ast
            in
            is_true
              (Realize.Runner.dev (Realize.Compiled_runner.runner runner1)
              == device1);
            is_true
              (Realize.Runner.dev (Realize.Compiled_runner.runner runner2)
              == device2);
            is_true (runner1 != runner2);
            is_true (runner2 == runner2_again);
            equal int 1 !calls;
            let dst = create_i32_buffer device2 [ 0 ] in
            let src = create_i32_buffer device2 [ 41 ] in
            ignore
              (Realize.Compiled_runner.call runner2 [ dst; src ] [] ~wait:false
                 ~timeout:None);
            Device.synchronize device2;
            equal (list int) [ 42 ] (read_i32_buffer dst));
          test "dispatch rejects unresolved buffer slots" (fun () ->
            let device = cpu "unresolved-buffer" in
            let called = ref false in
            let runner =
              Realize.Runner.make ~display_name:"must-not-run" ~device
                (fun _ _ ~wait:_ ~timeout:_ ->
                  called := true;
                  None)
            in
            let buffer = create_i32_buffer device [ 0 ] in
            let exec =
              Realize.Exec_item.make ~ast:(Tensor.sink [])
                ~bufs:[ Some buffer; None ] ~prg:runner ()
            in
            raises_match
              (function
                | Invalid_argument message ->
                    String.equal message
                      "exec item: unresolved buffer at argument 1 during \
                       execution"
                | _ -> false)
              (fun () -> ignore (Realize.Exec_item.run exec ()));
            raises_match
              (function
                | Jit.Jit_error message ->
                    String.equal message
                      "unresolved captured buffer at argument 1"
                | _ -> false)
              (fun () ->
                ignore
                  (Jit.lower_realize_ei ~device
                     ~get_program:(fun _ -> failwith "unexpected compilation")
                     exec));
            let jit_exec : Jit.exec_item =
              {
                uid = -1;
                bufs = [| Some buffer; None |];
                prg = View_op runner;
                fixedvars = [];
              }
            in
            raises_match
              (function
                | Jit.Jit_error message ->
                    String.equal message
                      "unresolved runtime buffer at argument 1"
                | _ -> false)
              (fun () -> Jit.run_ei jit_exec [] ~jit:false);
            is_false !called);
          test "CPU buffer views compose offsets and check bounds" (fun () ->
            let device = cpu "nested-views" in
            let base = create_i32_buffer device [ 10; 11; 12; 13; 14; 15 ] in
            let parent =
              Device.Buffer.view base ~size:4 ~dtype:Dtype.int32 ~offset:4
            in
            let nested =
              Device.Buffer.view parent ~size:2 ~dtype:Dtype.int32 ~offset:4
            in
            Device.Buffer.ensure_allocated nested;
            equal int 8 (Device.Buffer.offset nested);
            equal (list int) [ 12; 13 ] (read_i32_buffer nested);
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () ->
                ignore
                  (Device.Buffer.view base ~size:2 ~dtype:Dtype.int32
                     ~offset:20));
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () ->
                ignore
                  (Device.Buffer.view base ~size:(-1) ~dtype:Dtype.int32
                     ~offset:0)));
          test "JIT replay preserves byte offset for view input" (fun () ->
            let device = cpu "view-input-offset" in
            let captured_base =
              create_i32_buffer device [ 0; 0; 0; 0; 0; 0 ]
            in
            let captured_view =
              Device.Buffer.view captured_base ~size:2 ~dtype:Dtype.int32
                ~offset:4
            in
            let observed = ref [] in
            let runner =
              Realize.Runner.make ~display_name:"observe-view" ~device
                (fun bufs _ ~wait:_ ~timeout:_ ->
                  observed := read_i32_buffer (List.hd bufs);
                  None)
            in
            let exec : Jit.exec_item =
              {
                uid = -1;
                bufs = [| Some captured_view |];
                prg = View_op runner;
                fixedvars = [];
              }
            in
            let input_replace = Hashtbl.create 1 in
            Hashtbl.add input_replace (0, 0) 1;
            let views : Jit.view_input list =
              [
                {
                  vi_base_idx = 0;
                  vi_offset = Device.Buffer.offset captured_view;
                  vi_device = Device.name device;
                  vi_size = 2;
                  vi_dtype = Dtype.int32;
                };
              ]
            in
            let input_info : Jit.input_info array =
              [|
                {
                  ii_size = 6;
                  ii_dtype = Dtype.int32;
                  ii_device = Device.name device;
                };
              |]
            in
            let captured =
              Jit.create_captured () [ exec ] input_replace views input_info
            in
            let replay =
              create_i32_buffer device [ 10; 11; 12; 13; 14; 15 ]
            in
            Jit.exec_captured captured ~device [| replay |] [];
            equal (list int) [ 11; 12 ] !observed);
        ];
    ]
