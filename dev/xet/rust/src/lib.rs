use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;

use serde::{Deserialize, Serialize};
use xet_pkg::legacy::hash_files_async;
use xet_pkg::xet_session::{
    HeaderMap, HeaderValue, XetFileDownload, XetFileDownloadGroup, XetFileInfo, XetSession,
    XetSessionBuilder, XetTaskState,
};
use xet_runtime::core::XetContext;

const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const X_REPO_COMMIT: &str = "x-repo-commit";
const X_LINKED_SIZE: &str = "x-linked-size";
const X_XET_HASH: &str = "x-xet-hash";
const X_XET_REFRESH_ROUTE: &str = "x-xet-refresh-route";

#[derive(Deserialize)]
struct DownloadHfFileRequest {
    repo_id: String,
    filename: String,
    destination: String,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    repo_type: Option<String>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    token: Option<String>,
}

#[derive(Serialize)]
struct DownloadHfFileReport {
    repo_id: String,
    filename: String,
    destination: String,
    revision: String,
    repo_type: String,
    file_hash: String,
    refresh_route: String,
    size: Option<u64>,
    commit_hash: Option<String>,
    location: String,
}

#[derive(Deserialize)]
struct FileDownloadGroupConfig {
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    token: Option<String>,
    #[serde(default)]
    token_expiry_unix_secs: Option<u64>,
    #[serde(default)]
    token_refresh_url: Option<String>,
    #[serde(default)]
    token_refresh_headers: HashMap<String, String>,
    #[serde(default)]
    custom_headers: HashMap<String, String>,
}

#[derive(Serialize)]
struct GroupProgressJson {
    total_bytes: u64,
    total_bytes_completed: u64,
    total_transfer_bytes: u64,
    total_transfer_bytes_completed: u64,
}

#[derive(Serialize)]
struct ItemProgressJson {
    item_name: String,
    total_bytes: u64,
    bytes_completed: u64,
}

#[derive(Serialize)]
struct DownloadReportJson {
    task_id: String,
    path: String,
    file_info: XetFileInfo,
    progress: Option<ItemProgressJson>,
}

#[derive(Serialize)]
struct DownloadGroupReportJson {
    progress: GroupProgressJson,
    downloads: Vec<DownloadReportJson>,
}

pub struct OcamlSession {
    inner: XetSession,
}

pub struct OcamlFileDownloadGroup {
    inner: XetFileDownloadGroup,
}

pub struct OcamlFileDownload {
    inner: XetFileDownload,
}

fn into_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(s) => s.into_raw(),
        Err(e) => CString::new(e.to_string())
            .expect("CString error without nul")
            .into_raw(),
    }
}

unsafe fn set_error(error_out: *mut *mut c_char, message: String) {
    if !error_out.is_null() {
        unsafe {
            *error_out = into_c_string(message);
        }
    }
}

unsafe fn c_str_arg<'a>(
    ptr: *const c_char,
    name: &str,
    context: &str,
    error_out: *mut *mut c_char,
) -> Option<&'a str> {
    if ptr.is_null() {
        unsafe {
            set_error(error_out, format!("{context}: null {name}"));
        }
        return None;
    }

    match unsafe { CStr::from_ptr(ptr) }.to_str() {
        Ok(value) => Some(value),
        Err(e) => {
            unsafe {
                set_error(error_out, format!("{context}: invalid UTF-8 {name}: {e}"));
            }
            None
        }
    }
}

fn json_string_result<T: Serialize>(value: &T, context: &str) -> Result<*mut c_char, String> {
    serde_json::to_string(value)
        .map(into_c_string)
        .map_err(|e| format!("{context}: failed to encode JSON: {e}"))
}

fn endpoint_with_no_trailing_slash(endpoint: Option<String>) -> String {
    endpoint
        .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string())
        .trim_end_matches('/')
        .to_string()
}

fn encode_path(path: &str) -> String {
    path.split('/')
        .map(urlencoding::encode)
        .collect::<Vec<_>>()
        .join("/")
}

fn repo_type_path_prefix(repo_type: &str) -> Result<&'static str, String> {
    match repo_type {
        "model" => Ok(""),
        "dataset" => Ok("datasets/"),
        "space" => Ok("spaces/"),
        other => Err(format!(
            "xet.download_hf_file: unsupported repo_type {other:?}; expected model, dataset, or space"
        )),
    }
}

fn hf_file_url(req: &DownloadHfFileRequest) -> Result<(String, String, String, String), String> {
    let endpoint = endpoint_with_no_trailing_slash(req.endpoint.clone());
    let revision = req.revision.clone().unwrap_or_else(|| "main".to_string());
    let repo_type = req.repo_type.clone().unwrap_or_else(|| "model".to_string());
    let prefix = repo_type_path_prefix(&repo_type)?;
    let url = format!(
        "{endpoint}/{prefix}{}/resolve/{}/{}",
        encode_path(&req.repo_id),
        encode_path(&revision),
        encode_path(&req.filename)
    );
    Ok((url, endpoint, revision, repo_type))
}

fn build_hub_headers(token: Option<&str>) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();
    headers.insert(
        "user-agent",
        HeaderValue::from_static(concat!("raven-xet/", env!("CARGO_PKG_VERSION"))),
    );
    headers.insert("accept-encoding", HeaderValue::from_static("identity"));

    if let Some(token) = token {
        let value = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|e| format!("xet.download_hf_file: invalid token header: {e}"))?;
        headers.insert("authorization", value);
    }

    Ok(headers)
}

fn build_header_map(headers: HashMap<String, String>, context: &str) -> Result<HeaderMap, String> {
    let mut header_map = HeaderMap::new();
    for (name, value) in headers {
        let name = http::header::HeaderName::from_bytes(name.as_bytes())
            .map_err(|e| format!("{context}: invalid header name {name:?}: {e}"))?;
        let value = HeaderValue::from_str(&value)
            .map_err(|e| format!("{context}: invalid header value for {name}: {e}"))?;
        header_map.insert(name, value);
    }
    Ok(header_map)
}

fn task_state_to_string(
    state: Result<XetTaskState, impl std::fmt::Display>,
) -> Result<String, String> {
    match state {
        Ok(XetTaskState::Running) => Ok("running".to_string()),
        Ok(XetTaskState::Finalizing) => Ok("finalizing".to_string()),
        Ok(XetTaskState::Completed) => Ok("completed".to_string()),
        Ok(XetTaskState::UserCancelled) => Ok("user_cancelled".to_string()),
        Ok(XetTaskState::Error(message)) => Err(message),
        Err(e) => Err(e.to_string()),
    }
}

fn group_progress_to_json(
    progress: xet_pkg::xet_session::GroupProgressReport,
) -> GroupProgressJson {
    GroupProgressJson {
        total_bytes: progress.total_bytes,
        total_bytes_completed: progress.total_bytes_completed,
        total_transfer_bytes: progress.total_transfer_bytes,
        total_transfer_bytes_completed: progress.total_transfer_bytes_completed,
    }
}

fn item_progress_to_json(progress: xet_pkg::xet_session::ItemProgressReport) -> ItemProgressJson {
    ItemProgressJson {
        item_name: progress.item_name,
        total_bytes: progress.total_bytes,
        bytes_completed: progress.bytes_completed,
    }
}

fn download_report_to_json(report: xet_pkg::xet_session::XetDownloadReport) -> DownloadReportJson {
    DownloadReportJson {
        task_id: report.task_id.to_string(),
        path: report.path.display().to_string(),
        file_info: report.file_info,
        progress: report.progress.map(item_progress_to_json),
    }
}

fn download_group_report_to_json(
    report: xet_pkg::xet_session::XetDownloadGroupReport,
) -> DownloadGroupReportJson {
    DownloadGroupReportJson {
        progress: group_progress_to_json(report.progress),
        downloads: report
            .downloads
            .into_values()
            .map(download_report_to_json)
            .collect(),
    }
}

fn xet_headers_without_auth(headers: &HeaderMap) -> HeaderMap {
    headers
        .iter()
        .filter(|(name, _)| name.as_str() != "authorization")
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn new_file_download_group(
    session: &XetSession,
    config: FileDownloadGroupConfig,
) -> Result<XetFileDownloadGroup, String> {
    let mut builder = session
        .new_file_download_group()
        .map_err(|e| format!("xet.Session.new_file_download_group: {e}"))?;
    if let Some(endpoint) = config.endpoint {
        builder = builder.with_endpoint(endpoint);
    }
    if let (Some(token), Some(expiry)) = (config.token, config.token_expiry_unix_secs) {
        builder = builder.with_token_info(token, expiry);
    }
    if let Some(url) = config.token_refresh_url {
        let headers = build_header_map(
            config.token_refresh_headers,
            "xet.Session.new_file_download_group",
        )?;
        builder = builder.with_token_refresh_url(url, headers);
    }
    let custom_headers =
        build_header_map(config.custom_headers, "xet.Session.new_file_download_group")?;
    builder
        .with_custom_headers(custom_headers)
        .build_blocking()
        .map_err(|e| format!("xet.Session.new_file_download_group: {e}"))
}

fn header_to_string(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

fn parse_link_header_for_xet_auth(link: &str) -> Option<String> {
    for item in link.split(',') {
        let mut parts = item.split(';').map(str::trim);
        let url = parts.next()?;
        let is_xet_auth = parts.any(|part| {
            part.eq_ignore_ascii_case("rel=\"xet-auth\"")
                || part.eq_ignore_ascii_case("rel=xet-auth")
        });
        if is_xet_auth && url.starts_with('<') && url.ends_with('>') {
            return Some(url[1..url.len() - 1].to_string());
        }
    }
    None
}

fn normalize_refresh_route(refresh_route: String, endpoint: &str) -> String {
    let hf_home = DEFAULT_HF_ENDPOINT;
    if refresh_route.starts_with(hf_home) {
        refresh_route.replacen(hf_home, endpoint, 1)
    } else {
        refresh_route
    }
}

async fn fetch_hf_file_metadata(
    req: &DownloadHfFileRequest,
) -> Result<(DownloadHfFileReport, HeaderMap), String> {
    let (url, endpoint, revision, repo_type) = hf_file_url(req)?;
    let hub_headers = build_hub_headers(req.token.as_deref())?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| format!("xet.download_hf_file: failed to initialize HTTP client: {e}"))?;
    let response = client
        .head(&url)
        .headers(hub_headers.clone())
        .send()
        .await
        .map_err(|e| format!("xet.download_hf_file: failed to fetch Hub metadata: {e}"))?;

    let headers = response.headers();
    if !response.status().is_success() && header_to_string(headers, X_XET_HASH).is_none() {
        return Err(format!(
            "xet.download_hf_file: Hub metadata request failed with HTTP {} for {url}",
            response.status()
        ));
    }

    let location =
        header_to_string(headers, "location").unwrap_or_else(|| response.url().to_string());
    let file_hash = header_to_string(headers, X_XET_HASH).ok_or_else(|| {
        "xet.download_hf_file: Hub metadata did not include Xet file data; this file may not be stored with Xet"
            .to_string()
    })?;
    let refresh_route = header_to_string(headers, "link")
        .and_then(|link| parse_link_header_for_xet_auth(&link))
        .or_else(|| header_to_string(headers, X_XET_REFRESH_ROUTE))
        .ok_or_else(|| {
            "xet.download_hf_file: Hub metadata did not include a Xet refresh route".to_string()
        })?;
    let size = header_to_string(headers, X_LINKED_SIZE)
        .or_else(|| header_to_string(headers, "content-length"))
        .and_then(|size| size.parse::<u64>().ok());

    let report = DownloadHfFileReport {
        repo_id: req.repo_id.clone(),
        filename: req.filename.clone(),
        destination: req.destination.clone(),
        revision,
        repo_type,
        file_hash,
        refresh_route: normalize_refresh_route(refresh_route, &endpoint),
        size,
        commit_hash: header_to_string(headers, X_REPO_COMMIT),
        location,
    };

    Ok((report, hub_headers))
}

fn run_download_hf_file(req: DownloadHfFileRequest) -> Result<DownloadHfFileReport, String> {
    let ctx = XetContext::default()
        .map_err(|e| format!("xet.download_hf_file: failed to initialize xet runtime: {e}"))?;
    let runtime = ctx.runtime.clone();
    runtime
        .bridge_sync(async move {
            let (report, hub_headers) = fetch_hf_file_metadata(&req).await?;
            let session = XetSessionBuilder::new()
                .build()
                .map_err(|e| format!("xet.download_hf_file: failed to create xet session: {e}"))?;
            let group = session
                .new_file_download_group()
                .map_err(|e| {
                    format!("xet.download_hf_file: failed to create xet download group: {e}")
                })?
                .with_token_refresh_url(&report.refresh_route, hub_headers.clone())
                .with_custom_headers(xet_headers_without_auth(&hub_headers))
                .build()
                .await
                .map_err(|e| {
                    format!("xet.download_hf_file: failed to initialize xet download group: {e}")
                })?;
            let file_info = XetFileInfo {
                hash: report.file_hash.clone(),
                file_size: report.size,
                sha256: None,
            };
            group
                .download_file_to_path(file_info, PathBuf::from(&report.destination))
                .await
                .map_err(|e| format!("xet.download_hf_file: failed to start xet download: {e}"))?;
            group
                .finish()
                .await
                .map_err(|e| format!("xet.download_hf_file: failed to finish xet download: {e}"))?;
            Ok(report)
        })
        .map_err(|e| format!("xet.download_hf_file: {e}"))?
}

#[unsafe(no_mangle)]
pub extern "C" fn xet_ocaml_version() -> *mut c_char {
    into_c_string(env!("CARGO_PKG_VERSION").to_string())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_session_create(
    error_out: *mut *mut c_char,
) -> *mut OcamlSession {
    match XetSessionBuilder::new().build() {
        Ok(session) => Box::into_raw(Box::new(OcamlSession { inner: session })),
        Err(e) => {
            unsafe {
                set_error(error_out, format!("xet.Session.create: {e}"));
            }
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_session_free(session: *mut OcamlSession) {
    if !session.is_null() {
        unsafe {
            drop(Box::from_raw(session));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_session_status(
    session: *mut OcamlSession,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    let Some(session) = (unsafe { session.as_ref() }) else {
        unsafe {
            set_error(error_out, "xet.Session.status: null session".to_string());
        }
        return ptr::null_mut();
    };

    match task_state_to_string(session.inner.status()) {
        Ok(status) => into_c_string(status),
        Err(e) => {
            unsafe {
                set_error(error_out, format!("xet.Session.status: {e}"));
            }
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_session_abort(
    session: *mut OcamlSession,
    error_out: *mut *mut c_char,
) -> bool {
    let Some(session) = (unsafe { session.as_ref() }) else {
        unsafe {
            set_error(error_out, "xet.Session.abort: null session".to_string());
        }
        return false;
    };

    match session.inner.abort() {
        Ok(()) => true,
        Err(e) => {
            unsafe {
                set_error(error_out, format!("xet.Session.abort: {e}"));
            }
            false
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_session_new_file_download_group(
    session: *mut OcamlSession,
    config_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut OcamlFileDownloadGroup {
    let Some(session) = (unsafe { session.as_ref() }) else {
        unsafe {
            set_error(
                error_out,
                "xet.Session.new_file_download_group: null session".to_string(),
            );
        }
        return ptr::null_mut();
    };
    let Some(config_json) = (unsafe {
        c_str_arg(
            config_json,
            "config JSON",
            "xet.Session.new_file_download_group",
            error_out,
        )
    }) else {
        return ptr::null_mut();
    };
    let config: FileDownloadGroupConfig = match serde_json::from_str(config_json) {
        Ok(config) => config,
        Err(e) => {
            unsafe {
                set_error(
                    error_out,
                    format!("xet.Session.new_file_download_group: invalid config JSON: {e}"),
                );
            }
            return ptr::null_mut();
        }
    };

    match new_file_download_group(&session.inner, config) {
        Ok(group) => Box::into_raw(Box::new(OcamlFileDownloadGroup { inner: group })),
        Err(e) => {
            unsafe {
                set_error(error_out, e);
            }
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_file_download_group_free(group: *mut OcamlFileDownloadGroup) {
    if !group.is_null() {
        unsafe {
            drop(Box::from_raw(group));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_file_download_group_start_download_file(
    group: *mut OcamlFileDownloadGroup,
    file_info_json: *const c_char,
    dest_path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut OcamlFileDownload {
    let Some(group) = (unsafe { group.as_ref() }) else {
        unsafe {
            set_error(
                error_out,
                "xet.File_download_group.start_download_file: null group".to_string(),
            );
        }
        return ptr::null_mut();
    };
    let Some(file_info_json) = (unsafe {
        c_str_arg(
            file_info_json,
            "file_info JSON",
            "xet.File_download_group.start_download_file",
            error_out,
        )
    }) else {
        return ptr::null_mut();
    };
    let Some(dest_path) = (unsafe {
        c_str_arg(
            dest_path,
            "destination path",
            "xet.File_download_group.start_download_file",
            error_out,
        )
    }) else {
        return ptr::null_mut();
    };
    let file_info: XetFileInfo = match serde_json::from_str(file_info_json) {
        Ok(file_info) => file_info,
        Err(e) => {
            unsafe {
                set_error(
                    error_out,
                    format!(
                        "xet.File_download_group.start_download_file: invalid file_info JSON: {e}"
                    ),
                );
            }
            return ptr::null_mut();
        }
    };

    match group
        .inner
        .download_file_to_path_blocking(file_info, PathBuf::from(dest_path))
    {
        Ok(download) => Box::into_raw(Box::new(OcamlFileDownload { inner: download })),
        Err(e) => {
            unsafe {
                set_error(
                    error_out,
                    format!("xet.File_download_group.start_download_file: {e}"),
                );
            }
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_file_download_group_wait_to_finish_json(
    group: *mut OcamlFileDownloadGroup,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    let Some(group) = (unsafe { group.as_ref() }) else {
        unsafe {
            set_error(
                error_out,
                "xet.File_download_group.wait_to_finish: null group".to_string(),
            );
        }
        return ptr::null_mut();
    };

    let report = match group.inner.clone().finish_blocking() {
        Ok(report) => download_group_report_to_json(report),
        Err(e) => {
            unsafe {
                set_error(
                    error_out,
                    format!("xet.File_download_group.wait_to_finish: {e}"),
                );
            }
            return ptr::null_mut();
        }
    };

    match json_string_result(&report, "xet.File_download_group.wait_to_finish") {
        Ok(json) => json,
        Err(e) => {
            unsafe {
                set_error(error_out, e);
            }
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_file_download_group_abort(
    group: *mut OcamlFileDownloadGroup,
    error_out: *mut *mut c_char,
) -> bool {
    let Some(group) = (unsafe { group.as_ref() }) else {
        unsafe {
            set_error(
                error_out,
                "xet.File_download_group.abort: null group".to_string(),
            );
        }
        return false;
    };

    match group.inner.abort() {
        Ok(()) => true,
        Err(e) => {
            unsafe {
                set_error(error_out, format!("xet.File_download_group.abort: {e}"));
            }
            false
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_file_download_group_progress_json(
    group: *mut OcamlFileDownloadGroup,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    let Some(group) = (unsafe { group.as_ref() }) else {
        unsafe {
            set_error(
                error_out,
                "xet.File_download_group.progress: null group".to_string(),
            );
        }
        return ptr::null_mut();
    };

    match json_string_result(
        &group_progress_to_json(group.inner.progress()),
        "xet.File_download_group.progress",
    ) {
        Ok(json) => json,
        Err(e) => {
            unsafe {
                set_error(error_out, e);
            }
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_file_download_free(download: *mut OcamlFileDownload) {
    if !download.is_null() {
        unsafe {
            drop(Box::from_raw(download));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_file_download_status(
    download: *mut OcamlFileDownload,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    let Some(download) = (unsafe { download.as_ref() }) else {
        unsafe {
            set_error(
                error_out,
                "xet.File_download.status: null download".to_string(),
            );
        }
        return ptr::null_mut();
    };

    match task_state_to_string(download.inner.status()) {
        Ok(status) => into_c_string(status),
        Err(e) => {
            unsafe {
                set_error(error_out, format!("xet.File_download.status: {e}"));
            }
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_file_download_cancel(download: *mut OcamlFileDownload) {
    if let Some(download) = unsafe { download.as_ref() } {
        download.inner.cancel();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_download_hf_file_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    if request_json.is_null() {
        unsafe {
            set_error(
                error_out,
                "xet.download_hf_file: null request JSON".to_string(),
            );
        }
        return ptr::null_mut();
    }

    let request_json = match unsafe { CStr::from_ptr(request_json) }.to_str() {
        Ok(request_json) => request_json,
        Err(e) => {
            unsafe {
                set_error(
                    error_out,
                    format!("xet.download_hf_file: invalid UTF-8 request JSON: {e}"),
                );
            }
            return ptr::null_mut();
        }
    };

    let request: DownloadHfFileRequest = match serde_json::from_str(request_json) {
        Ok(request) => request,
        Err(e) => {
            unsafe {
                set_error(
                    error_out,
                    format!("xet.download_hf_file: invalid request JSON: {e}"),
                );
            }
            return ptr::null_mut();
        }
    };

    match run_download_hf_file(request).and_then(|report| {
        serde_json::to_string(&report)
            .map_err(|e| format!("xet.download_hf_file: failed to encode result JSON: {e}"))
    }) {
        Ok(json) => into_c_string(json),
        Err(e) => {
            unsafe {
                set_error(error_out, e);
            }
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xet_ocaml_hash_files_json(
    paths_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    if paths_json.is_null() {
        unsafe {
            set_error(error_out, "xet.hash_files: null paths JSON".to_string());
        }
        return ptr::null_mut();
    }

    let paths_json = match unsafe { CStr::from_ptr(paths_json) }.to_str() {
        Ok(paths_json) => paths_json,
        Err(e) => {
            unsafe {
                set_error(
                    error_out,
                    format!("xet.hash_files: invalid UTF-8 paths JSON: {e}"),
                );
            }
            return ptr::null_mut();
        }
    };

    let paths: Vec<String> = match serde_json::from_str(paths_json) {
        Ok(paths) => paths,
        Err(e) => {
            unsafe {
                set_error(
                    error_out,
                    format!("xet.hash_files: invalid paths JSON: {e}"),
                );
            }
            return ptr::null_mut();
        }
    };

    let ctx = match XetContext::default() {
        Ok(ctx) => ctx,
        Err(e) => {
            unsafe {
                set_error(
                    error_out,
                    format!("xet.hash_files: failed to initialize xet runtime: {e}"),
                );
            }
            return ptr::null_mut();
        }
    };

    let runtime = ctx.runtime.clone();
    let files = match runtime.bridge_sync(async move { hash_files_async(&ctx, paths).await }) {
        Ok(Ok(files)) => files,
        Ok(Err(e)) => {
            unsafe {
                set_error(error_out, format!("xet.hash_files: {e}"));
            }
            return ptr::null_mut();
        }
        Err(e) => {
            unsafe {
                set_error(error_out, format!("xet.hash_files: {e}"));
            }
            return ptr::null_mut();
        }
    };

    match serde_json::to_string(&files) {
        Ok(json) => into_c_string(json),
        Err(e) => {
            unsafe {
                set_error(
                    error_out,
                    format!("xet.hash_files: failed to encode result JSON: {e}"),
                );
            }
            ptr::null_mut()
        }
    }
}
