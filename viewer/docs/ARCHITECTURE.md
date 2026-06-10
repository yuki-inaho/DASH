# アーキテクチャ

## 目的

**学習済み** の DASH（動的シーン Gaussian Splatting）モデルを、軽量な GPU ビューワで
再生する。DASH の推論（4D ハッシュエンコーディング + DeformNetwork）は PyTorch/CUDA を
必要とするため、Rust/WGSL へは移植**しない**。代わりに Python サイドカーが時刻 `t` の
変形を評価し、ビューワ native の 3DGS 形式で変形済みガウシアンを返す。ビューワは既存の
`preprocess → prefix-scan → duplicate → radix-sort → tile-range → tile-render`
パイプラインを毎フレーム再利用する（「4D」=毎フレーム stream される 3DGS。上流統合 MVP と同方式）。

## コンポーネント

| コンポーネント | パス | 役割 |
|---|---|---|
| `dash-runtime` | `crates/dash-runtime` | **再利用可能な transport。** サイドカーを起動（CWD = `DASH_ROOT` とし DASH の事前ビルド済 hashencoder JIT を再利用）、TCP プロトコルを話し、stride/サイズを検証し、生のフレームバイト列を返す。viewer/GPU 非依存。 |
| viewer（vendored fork） | `crates/wgpu-gs-viewer` | wgpu タイルレンダラ + DASH グルー: stream されたガウシアンの毎フレーム `queue.write_buffer`（`gaussian_buffer` に `COPY_DST` を付与）、ディレクトリ D&D / `--dash-model` 自動ロード、ヘッドレス offscreen の **`dash_bake`** バイナリ。 |
| sidecar | `sidecar/dash_viewer_sidecar` | モデルを一度ロードし、`t` ごとに `DeformModel.step` を評価して 240 バイト ABI に詰める。SOLID モジュール: `abi`, `protocol`, `model_repository`, `runtime`, `server`, `bake`, `cli`。 |
| splaTV exporter | `sidecar/dash_viewer_sidecar/splatv.py` | 4DGS/STG-Lite の Gaussian PLY を `antimatter15/splaTV` の `.splatv` テクスチャコンテナへ変換（tqdm 進捗付き）。DASH/3DGS PLY 入力は「明示的な静的互換エクスポート」としてのみ対応。 |
| web player | `web/` | WebGPU 不要の 2D-canvas フリップブック（baked PNG フレーム）。`playwright-cli` の検証対象。 |

## データフロー

1. ビューワ（または `dash_bake`）が `DashSessionConfig::from_env(model_dir)` と
   `DashSession::start` を構築し、次を spawn する:
   `python dash_sidecar.py --dash-root … --model-dir … --iteration … --mask-mode … --host 127.0.0.1 --port 0`
2. サイドカーが ply + `deform.pth` + `cfg_args` をロードし、`DASH_SIDECAR_READY {"host","port"}` を出力。
3. 毎フレーム: ビューワが `{"cmd":"frame","time":t}` を送る → サイドカーがメタデータ +
   `count × 240` バイトを返す → ビューワが `gaussian_buffer` に書き込み描画。

## ワイヤプロトコル（長さ前置 TCP、little-endian）

```
request  = u32 json_len + json
           {"cmd":"info"} | {"cmd":"frame","time":<f>} | {"cmd":"shutdown"}
response = u32 meta_len + meta_json + u64 payload_len + payload
meta(info)  = {ok, kind, gaussian_count, stride, mask_mode, device}
meta(frame) = {ok, kind, time, gaussian_count, stride, byte_len}
```

readiness: stdout に `DASH_SIDECAR_READY {json}` をちょうど 1 行のみ。その他のログはすべて stderr へ。

## Gaussian3d ABI — 240 バイト（単一の真実源）

`#[repr(C)]`（`crates/wgpu-gs-viewer/src/gaussian_resources.rs`）と、`numpy` dtype
（`sidecar/dash_viewer_sidecar/abi.py`）で同一レイアウトを定義。`dash-runtime` は stride
（`GAUSSIAN3D_STRIDE = 240`）のみを検証する。

| フィールド | 型 | バイト | オフセット |
|---|---|---|---|
| position | f32×3 | 12 | 0 |
| opacity | f32 | 4 | 12 |
| scale | f32×3 | 12 | 16 |
| _pad0 | u32 | 4 | 28 |
| rotation | f32×4 | 16 | 32 |
| sh | f32×48 | 192 | 48 |

activation 規約（サイドカー → ビューワ shader）: shader 側が `exp(scale)` と
`sigmoid(opacity)` を適用するため、サイドカーは `scale = log(scale_act)`、
`opacity = inverse_sigmoid(opacity_act)` を出力する。rotation は正規化済み。SH は
DC(3) + rest(45) の並び。

## ヘッドレス bake（検証の土台）

`dash_bake` は **surface 無し**の wgpu デバイスを生成し、compute+render パイプライン一式を
offscreen の `Rgba8Unorm` テクスチャ（`COPY_SRC`）へ描いて読み戻し、`frame_%04d.png` +
`manifest.json` を書き出す。カメラはロバストな auto-fit（各軸 median 中心 + p95 半径 ——
遠方の COLMAP 外れ点に強い）と、任意のカメラ **orbit** を使い、時間的にほぼ静的なモデルでも
フレームが視覚的に変化するようにする。swapchain を避けることで、ヘッドレス / VNC GPU 上でも
描画が安定する。

## splaTV エクスポート

`python -m dash_viewer_sidecar splatv` は Gaussian PLY を読み、splaTV の 4D テクスチャ
レイアウト（position/rotation/scale/color + 3 次の `motion_0..8`、`omega_0..3`、
時間方向の `trbf_center/trbf_scale`）で書き出す。真の 4DGS/STG-Lite 変換には `--require-4d`
を使う。これを付けない場合、素の 3DGS/DASH PLY は motion をゼロ・時間 RBF を広めに書いて
**静的スプラット**としてエクスポートされる（DASH が学習した変形は上記の native サイドカー経路で利用する）。
