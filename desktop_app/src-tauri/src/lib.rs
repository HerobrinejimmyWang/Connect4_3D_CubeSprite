#[cfg(not(any(target_os = "android", target_os = "ios")))]
use std::{ffi::OsString, fs, path::PathBuf, sync::Mutex};

#[cfg(not(any(target_os = "android", target_os = "ios")))]
use tauri::{Emitter, Manager, RunEvent, State, WindowEvent};
#[cfg(not(any(target_os = "android", target_os = "ios")))]
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

#[cfg(not(any(target_os = "android", target_os = "ios")))]
const SIDECAR_NAME: &str = "cubesprite-backend";

#[cfg(not(any(target_os = "android", target_os = "ios")))]
struct ManagedChild {
    pid: u32,
    child: CommandChild,
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
#[derive(Default)]
struct SidecarState {
    child: Mutex<Option<ManagedChild>>,
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn state_lock_error() -> String {
    "The AI sidecar state lock is unavailable".to_owned()
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn is_current_sidecar(state: &SidecarState, pid: u32) -> bool {
    match state.child.lock() {
        Ok(current) => current.as_ref().map(|managed| managed.pid) == Some(pid),
        // A poisoned lifecycle lock already makes the managed sidecar
        // unusable, so surface its events instead of hiding diagnostics.
        Err(_) => true,
    }
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn clear_current_sidecar(state: &SidecarState, pid: u32) -> bool {
    match state.child.lock() {
        Ok(mut current) if current.as_ref().map(|managed| managed.pid) == Some(pid) => {
            current.take();
            true
        }
        Ok(_) => false,
        // Surface the termination so frontend requests do not remain pending
        // when the lifecycle lock itself is unusable.
        Err(_) => true,
    }
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn resource_directory(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let bundled = app
        .path()
        .resource_dir()
        .map_err(|error| format!("Unable to locate bundled resources: {error}"))?;

    // During `tauri dev`, use the source resource folder when the platform
    // resource directory has not been populated yet. Release builds always
    // resolve to the installer's read-only resource directory.
    let development = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources");
    if cfg!(debug_assertions)
        && development.join("model_registry.json").is_file()
        && !bundled.join("model_registry.json").is_file()
    {
        Ok(development)
    } else {
        Ok(bundled)
    }
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn application_data_directory(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let directory = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Unable to locate the application data directory: {error}"))?;
    fs::create_dir_all(&directory)
        .map_err(|error| format!("Unable to prepare the application data directory: {error}"))?;
    Ok(directory)
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
#[tauri::command]
fn start_sidecar(app: tauri::AppHandle, state: State<'_, SidecarState>) -> Result<u32, String> {
    let mut slot = state.child.lock().map_err(|_| state_lock_error())?;
    if let Some(managed) = slot.as_ref() {
        return Ok(managed.pid);
    }

    let resource_dir = resource_directory(&app)?;
    let data_dir = application_data_directory(&app)?;
    let args = [
        OsString::from("--resource-dir"),
        resource_dir.into_os_string(),
        OsString::from("--data-dir"),
        data_dir.into_os_string(),
    ];
    let command = app
        .shell()
        .sidecar(SIDECAR_NAME)
        .map_err(|error| format!("Unable to prepare the AI sidecar: {error}"))?
        .args(args);
    let (mut events, child) = command
        .spawn()
        .map_err(|error| format!("Unable to start the AI sidecar: {error}"))?;
    let pid = child.pid();
    *slot = Some(ManagedChild { pid, child });
    drop(slot);

    let event_app = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = events.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let state = event_app.state::<SidecarState>();
                    if is_current_sidecar(&state, pid) {
                        let line = String::from_utf8_lossy(&bytes).into_owned();
                        let _ = event_app.emit("sidecar-stdout", line);
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    let state = event_app.state::<SidecarState>();
                    if is_current_sidecar(&state, pid) {
                        let line = String::from_utf8_lossy(&bytes).into_owned();
                        let _ = event_app.emit("sidecar-stderr", line);
                    }
                }
                CommandEvent::Error(message) => {
                    let state = event_app.state::<SidecarState>();
                    if is_current_sidecar(&state, pid) {
                        let _ = event_app.emit("sidecar-stderr", message);
                    }
                }
                CommandEvent::Terminated(payload) => {
                    let state = event_app.state::<SidecarState>();
                    if clear_current_sidecar(&state, pid) {
                        let _ = event_app.emit("sidecar-exit", payload);
                    }
                    break;
                }
                _ => {}
            }
        }
    });

    Ok(pid)
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
#[tauri::command]
fn write_sidecar(line: String, state: State<'_, SidecarState>) -> Result<(), String> {
    if line.contains('\r') || line.contains('\n') {
        return Err("A sidecar request must contain exactly one JSON line".to_owned());
    }

    let mut slot = state.child.lock().map_err(|_| state_lock_error())?;
    let managed = slot
        .as_mut()
        .ok_or_else(|| "The AI sidecar is not running".to_owned())?;
    let mut request = line.into_bytes();
    request.push(b'\n');
    managed
        .child
        .write(&request)
        .map_err(|error| format!("Unable to send a request to the AI sidecar: {error}"))
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn stop_sidecar_inner(state: &SidecarState) -> Result<bool, String> {
    let managed = state.child.lock().map_err(|_| state_lock_error())?.take();
    if let Some(managed) = managed {
        managed
            .child
            .kill()
            .map_err(|error| format!("Unable to stop the AI sidecar: {error}"))?;
        Ok(true)
    } else {
        Ok(false)
    }
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
#[tauri::command]
fn stop_sidecar(state: State<'_, SidecarState>) -> Result<bool, String> {
    stop_sidecar_inner(&state)
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn run_desktop() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState::default())
        .invoke_handler(tauri::generate_handler![
            start_sidecar,
            write_sidecar,
            stop_sidecar
        ])
        .on_window_event(|window, event| {
            if matches!(event, WindowEvent::CloseRequested { .. }) {
                let state = window.state::<SidecarState>();
                let _ = stop_sidecar_inner(&state);
            }
        })
        .build(tauri::generate_context!())
        .expect("failed to build CubeSprite");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::Exit) {
            let state = app_handle.state::<SidecarState>();
            let _ = stop_sidecar_inner(&state);
        }
    });
}

#[cfg(target_os = "android")]
fn run_android() {
    tauri::Builder::default()
        .plugin(tauri_plugin_cubesprite_mobile::init())
        .run(tauri::generate_context!())
        .expect("failed to run CubeSprite on Android");
}

#[cfg(target_os = "ios")]
fn run_ios() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("failed to run CubeSprite on iOS");
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    run_desktop();
    #[cfg(target_os = "android")]
    run_android();
    #[cfg(target_os = "ios")]
    run_ios();
}
