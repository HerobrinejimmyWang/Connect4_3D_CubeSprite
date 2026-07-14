use std::{ffi::OsString, path::PathBuf, sync::Mutex};

use tauri::{Emitter, Manager, RunEvent, State, WindowEvent};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

const SIDECAR_NAME: &str = "cubesprite-backend";

struct ManagedChild {
    pid: u32,
    child: CommandChild,
}

#[derive(Default)]
struct SidecarState {
    child: Mutex<Option<ManagedChild>>,
}

fn state_lock_error() -> String {
    "The AI sidecar state lock is unavailable".to_owned()
}

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

#[tauri::command]
fn start_sidecar(app: tauri::AppHandle, state: State<'_, SidecarState>) -> Result<u32, String> {
    let mut slot = state.child.lock().map_err(|_| state_lock_error())?;
    if let Some(managed) = slot.as_ref() {
        return Ok(managed.pid);
    }

    let resource_dir = resource_directory(&app)?;
    let args = [
        OsString::from("--resource-dir"),
        resource_dir.into_os_string(),
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
                    let line = String::from_utf8_lossy(&bytes).into_owned();
                    let _ = event_app.emit("sidecar-stdout", line);
                }
                CommandEvent::Stderr(bytes) => {
                    let line = String::from_utf8_lossy(&bytes).into_owned();
                    let _ = event_app.emit("sidecar-stderr", line);
                }
                CommandEvent::Error(message) => {
                    let _ = event_app.emit("sidecar-stderr", message);
                }
                CommandEvent::Terminated(payload) => {
                    let state = event_app.state::<SidecarState>();
                    if let Ok(mut current) = state.child.lock() {
                        if current.as_ref().map(|managed| managed.pid) == Some(pid) {
                            current.take();
                        }
                    }
                    let _ = event_app.emit("sidecar-exit", payload);
                    break;
                }
                _ => {}
            }
        }
    });

    Ok(pid)
}

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

#[tauri::command]
fn stop_sidecar(state: State<'_, SidecarState>) -> Result<bool, String> {
    stop_sidecar_inner(&state)
}

pub fn run() {
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
