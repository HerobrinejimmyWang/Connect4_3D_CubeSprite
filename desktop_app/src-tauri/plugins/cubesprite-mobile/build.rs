const COMMANDS: &[&str] = &["request", "close"];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .build();
}
