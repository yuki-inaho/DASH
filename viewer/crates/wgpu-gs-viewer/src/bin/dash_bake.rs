//! Headless bake CLI: render a DASH model directory to a PNG sequence.
//!
//! Usage:
//!   dash_bake [--model <dir>] [--out <dir>] [--frames N]
//! Falls back to DASH_MODEL_DIR for the model when --model is omitted.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    use std::path::PathBuf;
    env_logger::init();

    let mut model: Option<PathBuf> = None;
    let mut out = PathBuf::from("web/baked");
    let mut frames: u32 = 24;
    let mut orbit: f32 = 1.0;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next().map(PathBuf::from),
            "--out" => {
                if let Some(v) = args.next() {
                    out = PathBuf::from(v);
                }
            }
            "--frames" => {
                if let Some(v) = args.next() {
                    frames = v.parse().unwrap_or(frames);
                }
            }
            "--orbit" => {
                if let Some(v) = args.next() {
                    orbit = v.parse().unwrap_or(orbit);
                }
            }
            other => {
                if let Some(v) = other.strip_prefix("--model=") {
                    model = Some(PathBuf::from(v));
                } else if let Some(v) = other.strip_prefix("--out=") {
                    out = PathBuf::from(v);
                } else if let Some(v) = other.strip_prefix("--frames=") {
                    frames = v.parse().unwrap_or(frames);
                } else if let Some(v) = other.strip_prefix("--orbit=") {
                    orbit = v.parse().unwrap_or(orbit);
                }
            }
        }
    }

    let model_dir = model
        .or_else(|| std::env::var_os("DASH_MODEL_DIR").map(PathBuf::from))
        .ok_or_else(|| anyhow::anyhow!("--model <dir> or DASH_MODEL_DIR is required"))?;

    wgpu_gs_viewer::dash_bake::run(wgpu_gs_viewer::dash_bake::Options {
        model_dir,
        out_dir: out,
        frames,
        orbit_turns: orbit,
    })
}

#[cfg(target_arch = "wasm32")]
fn main() {}
