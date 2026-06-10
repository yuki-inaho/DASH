# DASH 4DGS viewer (`viewer/`)

A self-contained, pixi-managed viewer that plays a **trained DASH** model
(dynamic-scene Gaussian Splatting) in real time. Training is **not** part of this
component — it consumes an existing DASH output directory.

It reuses [`abist-co-ltd/wgpu-gs-viewer`](https://github.com/abist-co-ltd/wgpu-gs-viewer)
(`v-0.2.0`, MIT — vendored, see [`docs/PROVENANCE.md`](docs/PROVENANCE.md)) for the
GPU tile-rendering pipeline, and runs DASH inference in a Python/CUDA **sidecar**
that streams per-time deformed 3D Gaussians into the renderer.

```
┌──────────────┐  TCP (240-byte Gaussian3d frames)  ┌───────────────────────────┐
│ Python sidecar│ ─────────────────────────────────▶ │ Rust viewer (wgpu)        │
│ DASH .venv    │  ◀── {"cmd":"frame","time":t} ───── │ dash-runtime → GPU render │
│ (PyTorch/CUDA)│                                     │ (preprocess→sort→tile)    │
└──────────────┘                                     └───────────────────────────┘
```

## Layout

```
viewer/
  pixi.toml                       pixi project (rust + nodejs); all tasks
  crates/
    dash-runtime/                 reusable, viewer-agnostic transport crate
    wgpu-gs-viewer/               vendored fork + DASH glue + headless bake
  sidecar/
    dash_sidecar.py               thin launcher the Rust side spawns
    dash_viewer_sidecar/          SOLID modules: abi/protocol/model_repository/
                                  runtime/server/bake/cli
    tests/                        pytest (ABI + real-model contract)
  web/                            2D-canvas flipbook player (playwright target)
  scripts/                        e2e_native.sh / e2e_web.sh / build_web.sh
  docs/                           ARCHITECTURE / PROVENANCE / DOD / SPLATV
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design and the
240-byte ABI, and [`docs/DOD.md`](docs/DOD.md) for the verification record.

## Prerequisites

- pixi (`~/.pixi/bin/pixi`) — provides the Rust toolchain + node.
- An NVIDIA GPU + the DASH uv `.venv` at `../.venv` (torch cu124 + built CUDA
  extensions). The sidecar reuses it via `DASH_PYTHON`.
- A trained DASH model directory (`cfg_args`, `point_cloud/iteration_*/`,
  `deform/iteration_*/`).

## Quick start

```bash
cd viewer
pixi install

# Headless: render the model to a PNG sequence (no window needed).
pixi run build-web          # bakes web/baked/ from $DASH_MODEL_DIR
pixi run e2e-web            # playwright-cli verifies it plays in a browser

# Native window (needs a GPU-backed display):
pixi run run                # auto-loads $DASH_MODEL_DIR; drag a dir/.ply to switch
```

`pixi.toml` defaults point at the repo's `output/tva_nyx650_400_smoke` model and
the sibling `../.venv`. Override with the env vars below.

## Environment variables

| var | default | meaning |
|---|---|---|
| `DASH_ROOT` | `..` | DASH repo root (sidecar CWD; reuses its hashencoder JIT) |
| `DASH_PYTHON` | `../.venv/bin/python` | Python that can import DASH + torch/CUDA |
| `DASH_SIDECAR_SCRIPT` | `sidecar/dash_sidecar.py` | launcher the viewer spawns |
| `DASH_MODEL_DIR` | `../output/tva_nyx650_400_smoke` | model to auto-load |
| `DASH_ITERATION` | `-2` | `-1` latest, `-2` best, or an explicit N |
| `DASH_MASK_MODE` | `ply_dynamic` | `ply_dynamic` \| `all` \| `dash_spatial` |

## Tasks

```bash
pixi run build         # cargo build --release (workspace)
pixi run run           # native viewer (auto-load + drag-and-drop)
pixi run bake          # dash_bake: headless model -> PNG sequence
pixi run splatv -- --input /path/to/point_cloud.ply --output web/baked/model.splatv --require-4d
pixi run check         # fmt-check + clippy(-D) + tests for dash-runtime
pixi run sidecar-test  # pytest (ABI + real-model contract) in the DASH .venv
pixi run e2e-native    # headless bake + assert frames non-blank & differ
pixi run build-web     # bake web/baked/ for the browser player
pixi run e2e-web       # playwright-cli: load + verify the browser playback
```

## Notes

- **Native vs web.** DASH playback (sidecar/CUDA) is native-only. The web player
  is a WebGPU-free 2D flipbook of GPU-rendered frames baked by `dash_bake`, so it
  runs in any (headless) browser — which is what `playwright-cli` drives.
- **splaTV export.** `pixi run splatv -- ...` writes `.splatv` with a tqdm
  progress bar. It is a true 4D export when the input PLY contains
  `motion_*`/`omega_*`/`trbf_*`; DASH's own `point_cloud.ply` can also be
  exported as a static compatibility preview. See [`docs/SPLATV.md`](docs/SPLATV.md).
- **Reusability.** `crates/dash-runtime` has no viewer dependency: it owns the
  process lifecycle + TCP protocol and returns raw 240-byte frames. Any wgpu
  Gaussian viewer can reuse it by casting the bytes to its own `Gaussian3d`.
