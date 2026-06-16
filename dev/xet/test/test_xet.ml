let () =
  let file = Filename.temp_file "xet-" ".bin" in
  let oc = open_out_bin file in
  output_string oc "hello xet\n";
  close_out oc;
  let json = Xet.hash_files_raw [ file ] in
  Sys.remove file;
  assert (String.length (Xet.version ()) > 0);
  assert (String.contains json '[');
  assert (String.contains json ']');
  let session = Xet.Session.create () in
  assert (String.length (Xet.Session.status session) > 0);
  let group =
    Xet.Session.new_file_download_group
      ~endpoint:("local://" ^ Filename.get_temp_dir_name () ^ "/xet-cas")
      session
  in
  let progress = Xet.File_download_group.progress group in
  assert (String.contains progress '{');
  Xet.File_download_group.abort group
