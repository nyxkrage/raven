type t = { backend : Backend.t; device_id : int }

let create ?(device_id = 0) backend =
  if device_id < 0 then invalid_arg "Rune_pjrt.Device.create: device_id < 0";
  { backend; device_id }

let cpu ?device_id () = create ?device_id `Cpu
let cuda ?device_id () = create ?device_id `Cuda
let backend t = t.backend
let device_id t = t.device_id
