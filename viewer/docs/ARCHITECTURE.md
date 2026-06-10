# Architecture

## Goal

Play a **trained** DASH dynamic-scene Gaussian Splatting model in a lightweight
GPU viewer. DASH inference (4D hash encoding + DeformNetwork) needs PyTorch/CUDA,
so it is **not** ported to Rust/WGSL. Instead, a Python sidecar evaluates the
deformation for a requested time `t` and returns the deformed Gaussians in the
viewer's native 3DGS format; the viewer reuses its existing
`preprocess → prefix-scan → duplicate → radix-sort → tile-range → tile-render`
pipeline every frame. ("4D" = per-frame streamed 3DGS, matching the upstream
integration MVP.)

## Components

| Component | Path | Role |
|---|---|---|
| `dash-runtime` | `crates/dash-runtime` | **Reusable transport.** Spawns the sidecar (CWD = `DASH_ROOT` so it reuses DASH's prebuilt hashencoder JIT), speaks the TCP protocol, validates stride/size, returns raw frame bytes. No viewer/GPU dependency. |
| viewer (vendored fork) | `crates/wgpu-gs-viewer` | wgpu tile renderer + DASH glue: per-frame `queue.write_buffer` of streamed gaussians (`gaussian_buffer` gains `COPY_DST`), directory drag-and-drop / `--dash-model` auto-load, and a headless offscreen **`dash_bake`** binary. |
| sidecar | `sidecar/dash_viewer_sidecar` | Loads the model once; per `t` evaluates `DeformModel.step` and packs deformed Gaussians into the 240-byte ABI. SOLID modules: `abi`, `protocol`, `model_repository`, `runtime`, `server`, `bake`, `cli`. |
| splaTV exporter | `sidecar/dash_viewer_sidecar/splatv.py` | Converts 4DGS/STG-Lite Gaussian PLY files to the `antimatter15/splaTV` `.splatv` texture container, with tqdm progress. DASH/3DGS PLY input is supported only as an explicit static compatibility export. |
| web player | `web/` | WebGPU-free 2D-canvas flipbook of baked PNG frames; the `playwright-cli` target. |

## Data flow

1. Viewer (or `dash_bake`) builds `DashSessionConfig::from_env(model_dir)` and
   `DashSession::start`, which spawns
   `python dash_sidecar.py --dash-root … --model-dir … --iteration … --mask-mode … --host 127.0.0.1 --port 0`.
2. Sidecar loads ply + `deform.pth` + `cfg_args`, prints `DASH_SIDECAR_READY {"host","port"}`.
3. Per frame: viewer sends `{"cmd":"frame","time":t}`; sidecar returns metadata +
   `count × 240` bytes; viewer writes them into `gaussian_buffer` and renders.

## Wire protocol (length-prefixed TCP, little-endian)

```
request  = u32 json_len + json
           {"cmd":"info"} | {"cmd":"frame","time":<f>} | {"cmd":"shutdown"}
response = u32 meta_len + meta_json + u64 payload_len + payload
meta(info)  = {ok, kind, gaussian_count, stride, mask_mode, device}
meta(frame) = {ok, kind, time, gaussian_count, stride, byte_len}
```

Readiness: exactly one stdout line `DASH_SIDECAR_READY {json}`; all other logs go
to stderr.

## Gaussian3d ABI — 240 bytes (single source of truth)

`#[repr(C)]` in `crates/wgpu-gs-viewer/src/gaussian_resources.rs`, mirrored by
`numpy` dtype in `sidecar/dash_viewer_sidecar/abi.py`; `dash-runtime` only
asserts the stride (`GAUSSIAN3D_STRIDE = 240`).

| field | type | bytes | offset |
|---|---|---|---|
| position | f32×3 | 12 | 0 |
| opacity | f32 | 4 | 12 |
| scale | f32×3 | 12 | 16 |
| _pad0 | u32 | 4 | 28 |
| rotation | f32×4 | 16 | 32 |
| sh | f32×48 | 192 | 48 |

Activation convention (sidecar → viewer shader): the shader applies
`exp(scale)` and `sigmoid(opacity)`, so the sidecar emits `scale = log(scale_act)`
and `opacity = inverse_sigmoid(opacity_act)`; rotations are normalized; SH is
laid out DC (3) + rest (45).

## Headless bake (verification backbone)

`dash_bake` creates a **surfaceless** wgpu device, runs the full compute+render
pipeline into an offscreen `Rgba8Unorm` texture (`COPY_SRC`), reads it back, and
writes `frame_%04d.png` + `manifest.json`. It uses a robust auto-fit camera
(per-axis median center + p95 radius — resilient to far COLMAP outliers) and an
optional camera **orbit** so frames vary visibly even when a model is temporally
near-static. Avoiding a swapchain makes rendering reliable on headless / VNC GPUs.

## splaTV export

`python -m dash_viewer_sidecar splatv` reads a Gaussian PLY and writes the splaTV
4D texture layout: position/rotation/scale/color plus cubic `motion_0..8`,
`omega_0..3`, and temporal `trbf_center/trbf_scale`. Use `--require-4d` for real
4DGS/STG-Lite conversion. Without it, plain 3DGS/DASH PLY files are exported as
static splats by writing zero motion and a wide temporal RBF; DASH's learned
deformation remains available through the native sidecar path above.
