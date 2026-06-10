mod app;
mod camera;
mod gaussian_resources;
mod passes;
mod ply_loader;
mod scene;

// Headless offscreen renderer that bakes a DASH model to PNG frames (native only).
#[cfg(not(target_arch = "wasm32"))]
pub mod dash_bake;

use gaussian_resources as gaussian;
use winit::event_loop::EventLoop;

/// Resolve the DASH model directory to auto-load on startup:
/// `--dash-model <dir>` (or `--dash-model=<dir>`) overrides `DASH_MODEL_DIR`.
#[cfg(not(target_arch = "wasm32"))]
fn parse_dash_model_dir() -> Option<std::path::PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--dash-model" {
            return args.next().map(std::path::PathBuf::from);
        }
        if let Some(v) = a.strip_prefix("--dash-model=") {
            return Some(std::path::PathBuf::from(v));
        }
    }
    std::env::var_os("DASH_MODEL_DIR").map(std::path::PathBuf::from)
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::<app::UserEvent>::with_user_event().build()?;

    #[cfg(not(target_arch = "wasm32"))]
    {
        let dash_model_dir = parse_dash_model_dir();
        let mut app = app::App::new(dash_model_dir);
        event_loop.run_app(&mut app)?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let app = app::App::new(&event_loop);
        event_loop.spawn_app(app);
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
