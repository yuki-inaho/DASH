# DASH 4DGS ビューワ（`viewer/`）

**学習済み DASH** モデル（動的シーン Gaussian Splatting）をリアルタイム再生する、
pixi 管理の自己完結ビューワ。学習はこのコンポーネントの対象外で、既存の DASH 出力
ディレクトリを消費するだけ。

GPU タイルレンダリングパイプラインは
[`abist-co-ltd/wgpu-gs-viewer`](https://github.com/abist-co-ltd/wgpu-gs-viewer)
（`v-0.2.0`, MIT —— vendored、[`docs/PROVENANCE.md`](docs/PROVENANCE.md) 参照）を再利用し、
DASH の推論は Python/CUDA **サイドカー**で実行して、時刻ごとの変形済み 3D ガウシアンを
レンダラへ stream する。

```
┌──────────────┐  TCP (240-byte Gaussian3d frames)  ┌───────────────────────────┐
│ Python sidecar│ ─────────────────────────────────▶ │ Rust viewer (wgpu)        │
│ DASH .venv    │  ◀── {"cmd":"frame","time":t} ───── │ dash-runtime → GPU render │
│ (PyTorch/CUDA)│                                     │ (preprocess→sort→tile)    │
└──────────────┘                                     └───────────────────────────┘
```

## 構成

```
viewer/
  pixi.toml                       pixi プロジェクト（rust + nodejs）; 全タスク
  crates/
    dash-runtime/                 再利用可能・viewer 非依存の transport crate
    wgpu-gs-viewer/               vendored fork + DASH グルー + headless bake
  sidecar/
    dash_sidecar.py               Rust 側が spawn する薄いランチャ
    dash_viewer_sidecar/          SOLID モジュール: abi/protocol/model_repository/
                                  runtime/server/bake/splatv/cli
    tests/                        pytest（ABI + 実モデル契約）
  web/                            2D-canvas フリップブック player（playwright 対象）
  scripts/                        e2e_native.sh / e2e_web.sh / build_web.sh
  docs/                           ARCHITECTURE / PROVENANCE / DOD / SPLATV
```

設計と 240 バイト ABI は [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)、検証記録は
[`docs/DOD.md`](docs/DOD.md) を参照。

## 前提条件

- pixi（`~/.pixi/bin/pixi`）—— Rust ツールチェーン + node を提供。
- NVIDIA GPU と、`../.venv` にある DASH の uv `.venv`（torch cu124 + ビルド済 CUDA 拡張）。
  サイドカーは `DASH_PYTHON` 経由でこれを再利用する。
- 学習済み DASH モデルディレクトリ（`cfg_args`, `point_cloud/iteration_*/`,
  `deform/iteration_*/`）。

## クイックスタート

```bash
cd viewer
pixi install

# ヘッドレス: モデルを PNG 連番へ描画（ウィンドウ不要）。
pixi run build-web          # $DASH_MODEL_DIR から web/baked/ を bake
pixi run e2e-web            # playwright-cli でブラウザ再生を検証

# native ウィンドウ（GPU バック付きディスプレイが必要）:
pixi run run                # $DASH_MODEL_DIR を自動ロード; ディレクトリ/.ply をドロップで切替
```

`pixi.toml` の既定はリポジトリの `output/tva_nyx650_400_smoke` モデルと隣接の `../.venv` を
指す。下記の環境変数で上書き可能。

## 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `DASH_ROOT` | `..` | DASH リポジトリルート（サイドカー CWD; その hashencoder JIT を再利用） |
| `DASH_PYTHON` | `../.venv/bin/python` | DASH + torch/CUDA を import できる Python |
| `DASH_SIDECAR_SCRIPT` | `sidecar/dash_sidecar.py` | ビューワが spawn するランチャ |
| `DASH_MODEL_DIR` | `../output/tva_nyx650_400_smoke` | 自動ロードするモデル |
| `DASH_ITERATION` | `-2` | `-1` 最新, `-2` best, または明示的な N |
| `DASH_MASK_MODE` | `ply_dynamic` | `ply_dynamic` \| `all` \| `dash_spatial` |

## タスク

```bash
pixi run build         # cargo build --release（ワークスペース）
pixi run run           # native ビューワ（自動ロード + D&D）
pixi run bake          # dash_bake: ヘッドレスでモデル → PNG 連番
pixi run splatv -- --input /path/to/point_cloud.ply --output web/baked/model.splatv --require-4d
pixi run check         # dash-runtime の fmt-check + clippy(-D) + テスト
pixi run sidecar-test  # DASH .venv 内で pytest（ABI + 実モデル契約）
pixi run e2e-native    # ヘッドレス bake + フレーム非黒&相違を assert
pixi run build-web     # ブラウザ player 用に web/baked/ を bake
pixi run e2e-web       # playwright-cli: ブラウザ再生をロード&検証
```

## 補足

- **native と web。** DASH 再生（サイドカー/CUDA）は native 限定。web player は `dash_bake` が
  焼いた GPU 描画フレームの WebGPU 不要な 2D フリップブックで、任意の（ヘッドレス）ブラウザで動く ——
  これが `playwright-cli` の対象。
- **splaTV エクスポート。** `pixi run splatv -- ...` は tqdm 進捗バー付きで `.splatv` を書き出す。
  入力 PLY が `motion_*`/`omega_*`/`trbf_*` を含めば真の 4D エクスポート。DASH 自身の
  `point_cloud.ply` は静的互換プレビューとしてもエクスポート可能。[`docs/SPLATV.md`](docs/SPLATV.md) 参照。
- **再利用性。** `crates/dash-runtime` は viewer 非依存: プロセスライフサイクル + TCP プロトコルを
  所有し、生の 240 バイトフレームを返す。任意の wgpu Gaussian ビューワが、そのバイト列を自前の
  `Gaussian3d` へキャストするだけで再利用できる。
