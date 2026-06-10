# 由来（Provenance）と vendoring

## Vendored レンダラ

`crates/wgpu-gs-viewer/` は
[`abist-co-ltd/wgpu-gs-viewer`](https://github.com/abist-co-ltd/wgpu-gs-viewer)
を vendoring したもの:

- tag/branch: `v-0.2.0`
- commit: `9348fa63ea1fc5d615c04ad6d81381866147bf80`（2026-06-04）
- ライセンス: **MIT**（株式会社アビスト イノベーションセンター）—— `crates/wgpu-gs-viewer/LICENSE` に保持。

取り込み時に `.git`・`target/`・`Cargo.lock` を除外し、`[profile.release]` は
ワークスペースルートの `Cargo.toml` へ移動。容量削減のため上流 `docs/*.gif` のデモ素材も除外。

## vendored crate へ適用した DASH 統合変更

上流の `dash_wgpu_integration_patch` と機能的に等価。ただし、in-tree モジュールではなく
再利用可能な `dash-runtime` crate に依存する形へ書き換えている:

- `Cargo.toml`: native 限定の依存 `dash-runtime = { path = "../dash-runtime" }` を追加。
- `gaussian_resources.rs`: `gaussian_buffer` の usage に `COPY_DST` を付与（毎フレーム stream のため）。
- `scene.rs`: `SceneUniform::time()` と `set_time()` のアクセサを追加。
- `app.rs`: 任意の `dash_session`、毎フレーム `update_dash_frame`、描画ゲートを
  `is_time_dependent()` 化、`replace_gaussians` でセッション破棄、ディレクトリ drop →
  `load_dash_model_dir`、`App::new(dash_model_dir)` + `resumed` での自動ロード。
- `lib.rs`: `parse_dash_model_dir`（`--dash-model` / `DASH_MODEL_DIR`）、`pub mod dash_bake` の公開。
- **パッチを超える新規:** `dash_bake.rs` + `bin/dash_bake.rs`（ロバスト auto-fit カメラ + orbit を
  備えたヘッドレス offscreen レンダラ）。検証と web デモ用。
- **パッチを超える usability:** `app.rs`/`camera.rs` のマウス操作（左ドラッグ=オービット、
  右/中ドラッグ=パン、ホイール=ズーム）。オービットカメラは可動 `target` を持ち、ロード時に
  モデルへ **auto-fit**（median 中心 + p95 半径、遠方外れ点に強い）。操作はウィンドウタイトルに表示。

元パッチとの相違点（意図的）:
- DASH ブリッジを生バイト列を返す独立 crate **`dash-runtime`** に分離（`crate::gaussian` へ
  結合しない）→ 他の wgpu ビューワからも再利用可能。
- サイドカーは **CWD = `DASH_ROOT`** で spawn し、DASH の事前ビルド済 hashencoder JIT
  （`./tmp_build`）を再利用。パスは canonicalize して絶対パス化。
- 442 行モノリスの `dash_sidecar.py` を `sidecar/dash_viewer_sidecar/` 配下の SOLID モジュールへ
  分割（挙動 / ABI / プロトコルは不変）。薄い `sidecar/dash_sidecar.py` ランチャが spawn 契約を維持。

## ツールチェーン警告

`pixi run check` は自作の `dash-runtime` crate にのみ厳格（`-D warnings`）。vendored の
`wgpu-gs-viewer` は上流の警告（未使用 import / dead field がいくつか）をそのまま保持する。確認は
`pixi run clippy-viewer`。上流を見栄えのために改変せず、vendor 差分を最小化するため（DRY/KISS）。

## splaTV フォーマット参照

`sidecar/dash_viewer_sidecar/splatv.py` の `.splatv` エクスポータは
[`antimatter15/splaTV`](https://github.com/antimatter15/splaTV)（MIT, Kevin Kwok）が用いる
フォーマットを対象とする: 8 バイトの magic/manifest ヘッダ、JSON チャンクリスト、Gaussian あたり
16 個の `uint32` ワードを持つ `RGBA32UI` ペイロード。実装は上流 JavaScript の vendoring ではなく、
DASH/ビューワ向けのクリーンな Python コンバータ。
