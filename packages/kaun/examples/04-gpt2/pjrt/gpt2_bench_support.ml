open Kaun

let fields ~ctx t = Ptree.Dict.fields_exn ~ctx t
let get fs ~name dtype = Ptree.Dict.get_tensor_exn fs ~name dtype
let find ~ctx key fs = Ptree.Dict.find_exn ~ctx key fs

let wrapped_position_ids ~n_positions ~batch ~seq =
  Array.init (batch * seq) (fun i ->
      Int32.of_int ((i mod seq) mod n_positions))
  |> Nx.create Nx.int32 [| batch; seq |]

let causal_self_attention (type l) ~(cfg : Gpt2.config)
    ~(dtype : (float, l) Nx.dtype) ~training ~params (x : (float, l) Nx.t) :
    (float, l) Nx.t =
  let shape = Nx.shape x in
  let batch = shape.(0) in
  let seq = shape.(1) in
  let h = cfg.n_embd in
  let heads = cfg.n_head in
  let head_dim = h / heads in
  let fs = fields ~ctx:"Gpt2_bench_support.attention" params in
  let qkv_w = get fs ~name:"qkv_weight" dtype in
  let qkv_b = get fs ~name:"qkv_bias" dtype in
  let qkv = Nx.add (Nx.matmul x qkv_w) qkv_b in
  let qkv_parts = Nx.split ~axis:(-1) 3 qkv in
  let q = List.nth qkv_parts 0 in
  let k = List.nth qkv_parts 1 in
  let v = List.nth qkv_parts 2 in
  let split_heads t =
    Nx.reshape [| batch; seq; heads; head_dim |] t
    |> Nx.transpose ~axes:[ 0; 2; 1; 3 ]
  in
  let q = split_heads q in
  let k = split_heads k in
  let v = split_heads v in
  let dropout_rate =
    if training && cfg.attn_pdrop > 0.0 then Some cfg.attn_pdrop else None
  in
  let attn =
    Kaun.Fn.dot_product_attention ~is_causal:true ?dropout_rate q k v
  in
  let merged =
    Nx.transpose attn ~axes:[ 0; 2; 1; 3 ]
    |> Nx.contiguous
    |> Nx.reshape [| batch; seq; h |]
  in
  let o_w = get fs ~name:"o_weight" dtype in
  let o_b = get fs ~name:"o_bias" dtype in
  Nx.add (Nx.matmul merged o_w) o_b

let transformer_block (type l) ~(cfg : Gpt2.config)
    ~(dtype : (float, l) Nx.dtype) ~training ~params (x : (float, l) Nx.t) :
    (float, l) Nx.t =
  let fs = fields ~ctx:"Gpt2_bench_support.block" params in
  let ln1_g = get fs ~name:"ln1_gamma" dtype in
  let ln1_b = get fs ~name:"ln1_beta" dtype in
  let x' =
    Kaun.Fn.layer_norm ~gamma:ln1_g ~beta:ln1_b ~epsilon:cfg.layer_norm_eps x
  in
  let attn_params = find ~ctx:"Gpt2_bench_support.block" "attention" fs in
  let attn =
    causal_self_attention ~cfg ~dtype ~training ~params:attn_params x'
  in
  let attn =
    if training && cfg.resid_pdrop > 0.0 then
      Kaun.Fn.dropout ~rate:cfg.resid_pdrop attn
    else attn
  in
  let x = Nx.add x attn in
  let ln2_g = get fs ~name:"ln2_gamma" dtype in
  let ln2_b = get fs ~name:"ln2_beta" dtype in
  let x' =
    Kaun.Fn.layer_norm ~gamma:ln2_g ~beta:ln2_b ~epsilon:cfg.layer_norm_eps x
  in
  let ffn_up_w = get fs ~name:"ffn_up_weight" dtype in
  let ffn_up_b = get fs ~name:"ffn_up_bias" dtype in
  let ffn_down_w = get fs ~name:"ffn_down_weight" dtype in
  let ffn_down_b = get fs ~name:"ffn_down_bias" dtype in
  let y =
    Nx.add (Nx.matmul x' ffn_up_w) ffn_up_b |> Kaun.Activation.gelu_approx
  in
  let y = Nx.add (Nx.matmul y ffn_down_w) ffn_down_b in
  let y =
    if training && cfg.resid_pdrop > 0.0 then
      Kaun.Fn.dropout ~rate:cfg.resid_pdrop y
    else y
  in
  Nx.add x y

let forward_with_position_ids (type l) ~(cfg : Gpt2.config) ~params
    ~(dtype : (float, l) Nx.dtype) ~training
    ~(position_ids : (int32, Bigarray.int32_elt) Nx.t)
    (input_ids : (int32, Bigarray.int32_elt) Nx.t) :
    (float, l) Nx.t =
  let input_ids = Nx.cast Nx.int32 input_ids in
  let shape = Nx.shape input_ids in
  let batch = shape.(0) in
  let seq = shape.(1) in
  let root = fields ~ctx:"Gpt2_bench_support.forward" params in
  let wte = get root ~name:"wte" dtype in
  let wpe = get root ~name:"wpe" dtype in
  let layers_t = find ~ctx:"Gpt2_bench_support.forward" "layers" root in
  let tok = Kaun.Fn.embedding ~scale:false ~embedding:wte input_ids in
  let pos = Kaun.Fn.embedding ~scale:false ~embedding:wpe position_ids in
  let x = Nx.add tok pos in
  let x =
    if training && cfg.embd_pdrop > 0.0 then
      Kaun.Fn.dropout ~rate:cfg.embd_pdrop x
    else x
  in
  let blocks = Ptree.List.items_exn ~ctx:"Gpt2_bench_support.layers" layers_t in
  let x =
    List.fold_left
      (fun h block_params ->
        transformer_block ~cfg ~dtype ~training ~params:block_params h)
      x blocks
  in
  let ln_f_g = get root ~name:"ln_f_gamma" dtype in
  let ln_f_b = get root ~name:"ln_f_beta" dtype in
  let hidden =
    Kaun.Fn.layer_norm ~gamma:ln_f_g ~beta:ln_f_b ~epsilon:cfg.layer_norm_eps x
  in
  let _ = (batch, seq) in
  Nx.matmul hidden (Nx.transpose wte ~axes:[ 1; 0 ])

let make_prompt_ids ~vocab_size ~prompt_tokens =
  Array.init prompt_tokens (fun i ->
      Int32.of_int ((((i * 17) + 11) mod (vocab_size - 1)) + 1))
