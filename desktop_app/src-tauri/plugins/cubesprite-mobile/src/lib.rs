use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{
    plugin::{Builder, TauriPlugin},
    Runtime,
};
#[cfg(mobile)]
use tauri::{AppHandle, Manager};

#[cfg(mobile)]
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct MobileRequest {
    command: String,
    params: Value,
}

#[cfg(mobile)]
#[derive(Clone)]
struct MobileBackend<R: Runtime>(tauri::plugin::PluginHandle<R>);

#[cfg(mobile)]
#[tauri::command]
async fn request<R: Runtime>(
    app: AppHandle<R>,
    command: String,
    params: Value,
) -> Result<Value, String> {
    let backend = app.state::<MobileBackend<R>>();
    backend
        .0
        .run_mobile_plugin_async("request", MobileRequest { command, params })
        .await
        .map_err(|error| error.to_string())
}

#[cfg(not(mobile))]
#[tauri::command]
async fn request(command: String, params: Value) -> Result<Value, String> {
    let _ = (command, params);
    Err("The CubeSprite mobile backend is only available on mobile targets.".to_owned())
}

#[cfg(mobile)]
#[tauri::command]
async fn close<R: Runtime>(app: AppHandle<R>) -> Result<Value, String> {
    let backend = app.state::<MobileBackend<R>>();
    backend
        .0
        .run_mobile_plugin_async(
            "close",
            MobileRequest {
                command: "system.close".to_owned(),
                params: Value::Object(Default::default()),
            },
        )
        .await
        .map_err(|error| error.to_string())
}

#[cfg(not(mobile))]
#[tauri::command]
async fn close() -> Result<Value, String> {
    Err("The CubeSprite mobile backend is only available on mobile targets.".to_owned())
}

/// Register the Kotlin-backed CubeSprite engine on Android.
///
/// A complete game or MCTS request crosses the WebView/native boundary once;
/// simulations and ONNX calls stay on the Android worker thread.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::<R, ()>::new("cubesprite-mobile")
        .invoke_handler(tauri::generate_handler![request, close])
        .setup(|_app, _api| {
            #[cfg(target_os = "android")]
            {
                let handle =
                    _api.register_android_plugin("com.cubesprite.mobile", "CubeSpritePlugin")?;
                _app.manage(MobileBackend(handle));
            }
            Ok(())
        })
        .build()
}
