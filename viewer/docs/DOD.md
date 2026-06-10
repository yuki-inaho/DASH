# Definition of Done — verification record

Goal: build a DASH 4DGS viewer and confirm it actually works, using
`playwright-cli`. All levels below were run on this machine (RTX 4090, CUDA 12.4).

## L1 — static / unit (`pixi run check`)

`cargo fmt --check`, `clippy -D warnings`, and tests for the reusable
`dash-runtime` crate.

- Result: **PASS** — 4 unit tests (protocol framing, response parse,
  `validate_frame`, loopback client round-trip) + 3 integration tests (ABI
  stride, config defaults, `from_env`).

## L2 — sidecar contract (`pixi run sidecar-test`)

`pytest` in the DASH `.venv` against the real trained model.

- Result: **PASS — 5 passed in ~91s.** `test_abi` (stride 240, pack round-trip,
  inverse-sigmoid, quat normalize) + `test_contract` (loads
  `output/tva_nyx650_400_smoke`: `info.stride == 240`, `gaussian_count = 41333`,
  `frame(0)/frame(0.5)` length `= 41333×240`, finite, unit quaternions).

## L3 — native E2E (`pixi run e2e-native`)

Headless offscreen bake (sidecar → dash-runtime → GPU pipeline → PNG), asserting
frames are non-blank and differ.

- Result: **PASS — `E2E_NATIVE_PASS frames=8 distinct=8`.** Each PNG > 20 KB
  (non-blank); 8 distinct content hashes (camera orbit).

## L4 — web E2E with playwright-cli (`pixi run e2e-web`)  ← goal requirement

`playwright-cli` (bundled chromium, headless) loads the web player serving the
baked frames and verifies playback.

- Result: **PASS — `E2E_WEB_PASS`.** Report:
  `ready frames=8 drawn=25 distinct=8 nonblank=0.520 errors=0`; `console error`
  level = 0. Screenshot saved to `e2e-out/web_playwright.png` (shows the rendered
  DASH gaussian splat + HUD on the page).

## Notes / honest caveats

- The bundled `tva_nyx650_400_smoke` is a short "smoke" training: its ply marks
  **0 dynamic points** and the deform network's temporal motion is negligible
  (max |Δposition| ≈ 1.9e-5 between t=0 and t=0.5, `mask_mode=all`). The viewer
  *does* stream a freshly DASH-evaluated frame every time step (the 4D pipeline
  runs each frame); with this model the visible inter-frame change comes from the
  camera **orbit**. A fully-trained dynamic model would show temporal motion
  through the same path.
- DASH playback is native-only (PyTorch/CUDA sidecar). The web/playwright path
  verifies the *rendered output* in a browser via a WebGPU-free 2D flipbook of
  genuine GPU-rendered frames.

## Reproduce

```bash
cd viewer && pixi install
pixi run check
pixi run sidecar-test
pixi run e2e-native
pixi run build-web && pixi run e2e-web
```
