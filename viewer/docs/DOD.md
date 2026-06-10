# 完了の定義（DOD）—— 検証記録

ゴール: DASH 4DGS ビューワを構築し、**`playwright-cli` を使って実際に動作することを確認**する。
以下のレベルはすべて本マシン（RTX 4090, CUDA 12.4）で実行・確認済み。

## L1 —— 静的 / ユニット（`pixi run check`）

再利用可能な `dash-runtime` crate に対する `cargo fmt --check`・`clippy -D warnings`・テスト。

- 結果: **PASS** —— ユニット 4 件（プロトコルのフレーミング、レスポンス解析、`validate_frame`、
  ループバック client ラウンドトリップ）+ 統合 3 件（ABI stride、config デフォルト、`from_env`）。

## L2 —— サイドカー契約（`pixi run sidecar-test`）

DASH `.venv` 内の `pytest` を、実学習済みモデルに対して実行。

- 結果: **PASS —— 5 passed（約 91 秒）。** `test_abi`（stride 240、pack ラウンドトリップ、
  inverse-sigmoid、quat 正規化）+ `test_contract`（`output/tva_nyx650_400_smoke` をロード:
  `info.stride == 240`、`gaussian_count = 41333`、`frame(0)/frame(0.5)` の長さ `= 41333×240`、
  有限値、単位クォータニオン）。

## L3 —— native E2E（`pixi run e2e-native`）

ヘッドレス offscreen bake（サイドカー → dash-runtime → GPU パイプライン → PNG）。
フレームが非黒かつ相違することを assert。

- 結果: **PASS —— `E2E_NATIVE_PASS frames=8 distinct=8`。** 各 PNG > 20 KB（非黒）、
  8 個の相異なるコンテンツハッシュ（カメラ orbit による）。

## L4 —— playwright-cli による web E2E（`pixi run e2e-web`）  ← ゴール要件

`playwright-cli`（同梱 chromium、ヘッドレス）が、baked フレームを配信する web player をロードし、
再生を検証。

- 結果: **PASS —— `E2E_WEB_PASS`。** レポート:
  `ready frames=8 drawn=25 distinct=8 nonblank=0.520 errors=0`。`console error` レベル = 0。
  スクリーンショットを `e2e-out/web_playwright.png` に保存（ページ上に描画された DASH の
  ガウシアンスプラット + HUD を確認）。

## 注意 / 正直な但し書き

- 同梱の `tva_nyx650_400_smoke` は短い「smoke（動作確認）」学習: ply の動的点は **0**、
  変形ネットワークの時間変化は無視できる大きさ（`mask_mode=all` で t=0 と t=0.5 の間の
  max |Δposition| ≈ 1.9e-5）。ビューワは時間ステップごとに DASH を評価したフレームを実際に stream
  している（4D パイプラインは毎フレーム走る）が、このモデルでは可視的なフレーム間変化はカメラ
  **orbit** に由来する。十分に学習された動的モデルなら、同じ経路で時間変形が見える。
- DASH の再生は native 限定（PyTorch/CUDA サイドカー）。web/playwright 経路は、本物の GPU 描画
  フレームを WebGPU 不要の 2D フリップブックでブラウザ表示し、その描画結果を検証する。

## 再現手順

```bash
cd viewer && pixi install
pixi run check
pixi run sidecar-test
pixi run e2e-native
pixi run build-web && pixi run e2e-web
```
