# Provenance & vendoring

## Vendored renderer

`crates/wgpu-gs-viewer/` is a vendored copy of
[`abist-co-ltd/wgpu-gs-viewer`](https://github.com/abist-co-ltd/wgpu-gs-viewer):

- tag/branch: `v-0.2.0`
- commit: `9348fa63ea1fc5d615c04ad6d81381866147bf80` (2026-06-04)
- license: **MIT** (株式会社アビスト イノベーションセンター) — retained in
  `crates/wgpu-gs-viewer/LICENSE`.

`.git`, `target/`, and `Cargo.lock` were dropped on import; `[profile.release]`
moved to the workspace root `Cargo.toml`.

## DASH integration changes applied to the vendored crate

Functionally equivalent to the upstream `dash_wgpu_integration_patch`, re-expressed
to depend on the reusable `dash-runtime` crate instead of an in-tree module:

- `Cargo.toml`: add native-only dep `dash-runtime = { path = "../dash-runtime" }`.
- `gaussian_resources.rs`: `gaussian_buffer` usage gains `COPY_DST` (per-frame streaming).
- `scene.rs`: add `SceneUniform::time()` and `set_time()` accessors.
- `app.rs`: optional `dash_session`; per-frame `update_dash_frame`; render gate
  uses `is_time_dependent()`; `replace_gaussians` clears the session; directory
  drop → `load_dash_model_dir`; `App::new(dash_model_dir)` + `resumed` auto-load.
- `lib.rs`: `parse_dash_model_dir` (`--dash-model` / `DASH_MODEL_DIR`); expose
  `pub mod dash_bake`.
- **New, beyond the patch:** `dash_bake.rs` + `bin/dash_bake.rs` (headless
  offscreen renderer with robust auto-fit camera + orbit) for verification and
  the web demo.
- **Usability, beyond the patch:** mouse navigation in `app.rs`/`camera.rs`
  (left-drag orbit, right/middle-drag pan, wheel zoom); the orbit camera now
  tracks a movable `target` and **auto-fits** to the loaded model (median centre
  + p95 radius, robust to far outliers); the window title shows the controls.

Differences vs. the original patch (intentional):
- DASH bridge is a standalone **`dash-runtime`** crate returning raw bytes (no
  `crate::gaussian` coupling) → reusable by other wgpu viewers.
- Sidecar spawned with **CWD = `DASH_ROOT`** to reuse DASH's prebuilt hashencoder
  JIT (`./tmp_build`) and with canonicalized absolute paths.
- The 442-line `dash_sidecar.py` monolith is split into SOLID modules under
  `sidecar/dash_viewer_sidecar/` (behaviour/ABI/protocol unchanged); a thin
  `sidecar/dash_sidecar.py` launcher preserves the spawn contract.

## Toolchain warnings

`pixi run check` is strict (`-D warnings`) only for our own `dash-runtime` crate.
The vendored `wgpu-gs-viewer` keeps its upstream warnings (a few unused
imports / dead fields); run `pixi run clippy-viewer` to see them. This keeps the
vendor diff minimal (DRY/KISS) rather than editing upstream cosmetically.
