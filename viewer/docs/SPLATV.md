# splaTV export

`dash_viewer_sidecar splatv` converts Gaussian PLY files into the
[`antimatter15/splaTV`](https://github.com/antimatter15/splaTV) `.splatv`
container.

The exporter targets the splaTV 4DGS/STG-Lite texture layout:

- static fields: `x/y/z`, `rot_0..3`, `scale_0..2`, `opacity`, `f_dc_0..2`
- 4D fields: `motion_0..8`, `omega_0..3`, `trbf_center`, `trbf_scale`
- output texture: `RGBA32UI`, 16 `uint32` words per Gaussian, `texwidth=4096`
- file header: magic `0x674b`, JSON manifest length, JSON manifest, texture bytes

## Command

```bash
cd viewer

# True 4DGS/STG-Lite PLY. The progress bar is enabled by default.
pixi run splatv -- \
  --input /path/to/point_cloud.ply \
  --output web/baked/model.splatv \
  --require-4d
```

For a plain 3DGS/DASH PLY, omit `--require-4d`:

```bash
pixi run splatv -- \
  --input ../output/tva_nyx650_400_smoke/point_cloud/iteration_best/point_cloud.ply \
  --output web/baked/model_static.splatv
```

That path is intentionally a **static 3DGS compatibility export**. It writes zero
motion/angular velocity and a wide temporal RBF so the splat remains visible in
splaTV, but it does not encode DASH's learned time deformation.

## Options

| option | meaning |
|---|---|
| `--require-4d` | fail unless every `motion_*`, `omega_*`, and `trbf_*` field is present |
| `--camera-json` | use an existing splaTV camera JSON/list instead of the auto-fit camera |
| `--color-mode auto` | 4DGS PLY uses raw `f_dc_*` color; 3DGS/DASH PLY converts SH DC with `0.5 + SH_C0 * f_dc` |
| `--chunk-size` | number of Gaussians packed per progress-bar update |
| `--no-progress` | disable the tqdm progress bar for scripts/tests |

## DASH boundary

DASH stores time-varying motion in `deform/iteration_*/deform.pth`, not in
`point_cloud.ply`. A direct `.splatv` export of DASH dynamic motion would require
sampling the neural deformation and fitting splaTV's cubic `motion_*` plus
`omega_*` fields, or exporting a frame sequence. The current correct paths are:

- real DASH dynamic viewing: `pixi run run` or `pixi run bake`
- portable splaTV file from a true 4DGS/STG-Lite PLY: `pixi run splatv -- --require-4d ...`
- static portable preview from a DASH/3DGS PLY: `pixi run splatv -- ...`
