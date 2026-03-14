val greedy_decode :
  ?backend:Backend.t ->
  ?device_id:int ->
  max_tokens:int ->
  ((int32, 'a) Nx.t -> ('b, 'c) Nx.t) ->
  (int32, 'a) Nx.t ->
  (int32, 'a) Nx.t
