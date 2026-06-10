# splaTV エクスポート

`dash_viewer_sidecar splatv` は Gaussian PLY ファイルを
[`antimatter15/splaTV`](https://github.com/antimatter15/splaTV) の `.splatv`
コンテナへ変換する。

エクスポータは splaTV の 4DGS/STG-Lite テクスチャレイアウトを対象とする:

- 静的フィールド: `x/y/z`, `rot_0..3`, `scale_0..2`, `opacity`, `f_dc_0..2`
- 4D フィールド: `motion_0..8`, `omega_0..3`, `trbf_center`, `trbf_scale`
- 出力テクスチャ: `RGBA32UI`、Gaussian あたり 16 個の `uint32` ワード、`texwidth=4096`
- ファイルヘッダ: magic `0x674b`、JSON manifest の長さ、JSON manifest、テクスチャバイト列

## コマンド

```bash
cd viewer

# 真の 4DGS/STG-Lite PLY。進捗バーは既定で有効。
pixi run splatv -- \
  --input /path/to/point_cloud.ply \
  --output web/baked/model.splatv \
  --require-4d
```

素の 3DGS/DASH PLY では `--require-4d` を省略する:

```bash
pixi run splatv -- \
  --input ../output/tva_nyx650_400_smoke/point_cloud/iteration_best/point_cloud.ply \
  --output web/baked/model_static.splatv
```

このパスは意図的に **静的 3DGS 互換エクスポート** である。motion / 角速度をゼロにし、時間 RBF を
広く取ることで splaTV 上で見えたままにするが、DASH が学習した時間変形はエンコードしない。

## オプション

| オプション | 意味 |
|---|---|
| `--require-4d` | すべての `motion_*`・`omega_*`・`trbf_*` フィールドが存在しなければ失敗させる |
| `--camera-json` | auto-fit カメラの代わりに既存の splaTV カメラ JSON/リストを使う |
| `--color-mode auto` | 4DGS PLY は生の `f_dc_*` 色を使用、3DGS/DASH PLY は `0.5 + SH_C0 * f_dc` で SH DC を変換 |
| `--chunk-size` | 進捗バー更新あたりに pack する Gaussian 数 |
| `--no-progress` | スクリプト/テスト向けに tqdm 進捗バーを無効化 |

## DASH との境界

DASH は時間変化する motion を `point_cloud.ply` ではなく `deform/iteration_*/deform.pth` に
保持する。DASH の動的 motion を直接 `.splatv` へエクスポートするには、ニューラル変形を
サンプリングして splaTV の 3 次 `motion_*` および `omega_*` フィールドに fit するか、フレーム列を
書き出す必要がある。現状の正しい経路は次のとおり:

- 本物の DASH 動的視聴: `pixi run run` または `pixi run bake`
- 真の 4DGS/STG-Lite PLY から可搬な splaTV ファイル: `pixi run splatv -- --require-4d ...`
- DASH/3DGS PLY からの静的な可搬プレビュー: `pixi run splatv -- ...`
