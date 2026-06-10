# データ / COLMAP-GLOMAP / DASH ワークフローメモ

**作成日**: `2026-06-10`  
**対象**: 新しいセッション・エージェント、DASH / gluemap / COLMAP 周辺作業の引き継ぎ  
**目的**: TVA_NYX650 の元画像から 400-frame の COLMAP 互換 pose / sparse reconstruction を作り、DASH の学習・viewer・splaTV へ接続した流れと、使った外部リポジトリを明示する。

---

## 1. 要点

- 元データは `/home/kasm-user/Desktop/Downloads/TVA_NYX650_2026_06_04_colmap/images` の 2000 枚。
- そこから 5 フレーム間隔で 400 枚を選び、`00000.jpg` から `00399.jpg` にリネームした subset を作った。
- 対応関係は `mapping.csv` に保存した。例: `00000.jpg -> frame_00001.jpg`、`00399.jpg -> frame_01996.jpg`。
- pose / sparse reconstruction は gluemap 側の pixi 環境で ALIKED + LightGlue + GLOMAP/COLMAP を使って作成した。
- 検証済み reconstruction は registered `400/400`、points `41333`、mean reprojection error `1.147551px`。
- DASH 側では `data/tva_nyx650_400_aliked_lg_glomap` を processed COLMAP dataset として読み、`mapping.csv` 由来の元 frame index を `fid` に反映する。
- dummy/random/synthetic pose fallback は使っていない。COLMAP DB/log、`model_analyzer`、pycolmap で実データ由来を確認した。

---

## 2. データ配置

| 種別 | パス | 内容 | Git 方針 |
|---|---|---|---|
| 元画像 | `/home/kasm-user/Desktop/Downloads/TVA_NYX650_2026_06_04_colmap/images` | 2000 枚の入力フレーム | リポジトリ外 |
| gluemap subset | `/home/kasm-user/Desktop/gluemap/data/tva_nyx650_400_uniform` | 400 枚 subset。画像は元画像への symlink | commit しない |
| gluemap SfM 結果 | `/home/kasm-user/Desktop/gluemap/results/tva_nyx650_400_aliked_lightglue_glomap` | ALIKED + LightGlue + GLOMAP/COLMAP の DB/log/sparse | commit しない |
| DASH processed dataset | `data/tva_nyx650_400_aliked_lg_glomap` | `mapping.csv`, `images/`, `sparse/0/*.bin` | `data/` は ignore。実体はローカル成果物 |
| DASH 学習出力 | `output/tva_nyx650_400_smoke`, `output/tva_nyx650_400_5k`, `output/tva_nyx650_400_15k` | short train / checkpoint / TensorBoard | `output/` は ignore |

`data/tva_nyx650_400_aliked_lg_glomap/images` は 400 本の symlink で、DASH から見ると gluemap subset を経由して元画像へ解決される。

確認コマンド:

```bash
cd /home/kasm-user/Desktop/DASH

find /home/kasm-user/Desktop/Downloads/TVA_NYX650_2026_06_04_colmap/images \
  -maxdepth 1 -type f | wc -l

find data/tva_nyx650_400_aliked_lg_glomap/images \
  -maxdepth 1 -type l | wc -l

find -L data/tva_nyx650_400_aliked_lg_glomap/images \
  -maxdepth 1 -type f | wc -l

wc -l data/tva_nyx650_400_aliked_lg_glomap/mapping.csv
head -5 data/tva_nyx650_400_aliked_lg_glomap/mapping.csv
tail -5 data/tva_nyx650_400_aliked_lg_glomap/mapping.csv
```

期待値:

```text
source images: 2000
DASH image symlinks: 400
DASH resolved images: 400
mapping.csv: 401 lines = header + 400 rows
```

---

## 3. 400-frame subset と mapping

subset は元画像 2000 枚から 5 フレーム間隔で 400 枚を等間隔サンプリングしたもの。

`mapping.csv` のカラム:

| カラム | 意味 |
|---|---|
| `subset_index` | 0 から 399 の subset 内 index |
| `subset_filename` | DASH/gluemap subset 側の `00000.jpg` 形式ファイル名 |
| `source_index_0based` | 元動画 frame の 0-based index |
| `source_index_1based` | 元動画 frame の 1-based index |
| `source_filename` | 元画像の `frame_00001.jpg` 形式ファイル名 |
| `source_path` | 元画像ファイルの絶対パス |

先頭と末尾の例:

```text
0,00000.jpg,0,1,frame_00001.jpg,/home/kasm-user/Desktop/Downloads/TVA_NYX650_2026_06_04_colmap/images/frame_00001.jpg
1,00001.jpg,5,6,frame_00006.jpg,/home/kasm-user/Desktop/Downloads/TVA_NYX650_2026_06_04_colmap/images/frame_00006.jpg
...
398,00398.jpg,1990,1991,frame_01991.jpg,/home/kasm-user/Desktop/Downloads/TVA_NYX650_2026_06_04_colmap/images/frame_01991.jpg
399,00399.jpg,1995,1996,frame_01996.jpg,/home/kasm-user/Desktop/Downloads/TVA_NYX650_2026_06_04_colmap/images/frame_01996.jpg
```

DASH の `scene/colmap_processed.py` はこの mapping を検証し、`source_index_0based / max_source_index` から `fid` を作る。これにより、単なる `00000.jpg..00399.jpg` の連番ではなく、元 2000-frame 時系列上の位置が DASH に伝わる。

---

## 4. SfM / pose 推論ワークフロー

### 4.1 gluemap 側

作業ディレクトリ:

```bash
cd /home/kasm-user/Desktop/gluemap
```

環境:

- pixi 管理。
- ブランチ: `cu129-l4`
- remote: `git@github.com:yuki-inaho/gluemap.git`
- 記録時点 commit: `baa8598 Document local third-party patches`

使った流れ:

1. 元画像から `data/tva_nyx650_400_uniform` を作成。
2. ALIKED で 400 画像の特徴量を抽出。
3. LightGlue で対応点を作成/import。
4. GLOMAP/COLMAP global mapper で sparse reconstruction を作成。
5. COLMAP の `model_analyzer` と pycolmap で reconstruction を検証。
6. `mapping.csv` と `sparse/0/*.bin` を DASH の processed dataset へ渡す。

主要成果物:

```text
/home/kasm-user/Desktop/gluemap/results/tva_nyx650_400_aliked_lightglue_glomap/
├── database_aliked_features.db
├── logs/
│   ├── matches_importer_aliked_lightglue.log
│   ├── sequential_matcher_aliked_lightglue.log
│   ├── global_mapper.log
│   ├── model_analyzer_result.log
│   ├── pycolmap_model_check.log
│   └── db_final_stats.log
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    ├── frames.bin
    └── rigs.bin
```

### 4.2 検証結果

`model_analyzer_result.log`:

```text
Frames: 400
Registered frames: 400
Images: 400
Registered images: 400
Points: 41333
Observations: 319972
Mean track length: 7.741320
Mean observations per image: 799.930000
Mean reprojection error: 1.147551px
```

`pycolmap_model_check.log`:

```text
num_images 400
num_reg_images 400
num_points3D 41333
camera 1 PINHOLE 800 600 [...]
```

`db_final_stats.log`:

```text
images [(400,)]
keypoints [(400,)]
descriptors [(400,)]
matches_rows_gt0 [(3816,)]
two_view_geometries_rows_gt0 [(3800,)]
kp_rows_min_max_avg [(1839, 2046, 2022.145)]
```

このため、COLMAP と同等の camera pose 推論は `sparse/0/images.bin` / `cameras.bin` / `points3D.bin` として成立している。DASH はこの COLMAP 互換 sparse model を読む。

---

## 5. DASH 側への接続

対象 dataset:

```text
data/tva_nyx650_400_aliked_lg_glomap/
├── images/                  # 400 symlinks
├── mapping.csv              # source frame 対応表
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    ├── frames.bin
    ├── rigs.bin
    └── points3D.ply
```

検証 CLI:

```bash
cd /home/kasm-user/Desktop/DASH

uv run python scripts/validate_colmap_dataset.py \
  --source-path data/tva_nyx650_400_aliked_lg_glomap \
  --mapping data/tva_nyx650_400_aliked_lg_glomap/mapping.csv \
  --expect-images 400 \
  --expect-registered 400
```

期待値:

```text
images: 400
mapping_rows: 400
registered: 400
points: 41333
camera_models: PINHOLE
source_paths_validated: False
```

元画像パスまで検証する場合:

```bash
uv run python scripts/validate_colmap_dataset.py \
  --source-path data/tva_nyx650_400_aliked_lg_glomap \
  --mapping data/tva_nyx650_400_aliked_lg_glomap/mapping.csv \
  --expect-images 400 \
  --expect-registered 400 \
  --validate-source-paths
```

`--validate-source-paths` はローカル絶対パスに依存するため、別マシンでは失敗しても dataset 自体の sparse validation とは切り分ける。

---

## 6. DASH 学習・viewer・splaTV

### 6.1 short train

実データ smoke train の代表コマンド:

```bash
cd /home/kasm-user/Desktop/DASH

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

uv run python train.py \
  -s data/tva_nyx650_400_aliked_lg_glomap \
  -m output/tva_nyx650_400_smoke \
  --images images \
  --iterations 20 \
  --test_iterations 10 20 \
  --save_iterations 20 \
  --conf arguments/n3dv.py \
  --hydra_config configs/train/tva400_muon_schedulefree.yaml \
  --quiet
```

記録済み evidence:

- `pts=41333`
- iteration `20/20` 到達
- `output/tva_nyx650_400_smoke/point_cloud/iteration_20/point_cloud.ply`
- TensorBoard event
- `deform/iteration_20/deform.pth`

### 6.2 viewer

DASH viewer は `viewer/` の pixi 環境で動かす。

```bash
cd /home/kasm-user/Desktop/DASH/viewer
pixi run run
```

デフォルトの `DASH_MODEL_DIR` は `../output/tva_nyx650_400_smoke`。必要に応じて `DASH_MODEL_DIR` と `DASH_ITERATION` を上書きする。

### 6.3 splaTV

`.splatv` 変換器は `viewer/sidecar/dash_viewer_sidecar/splatv.py`。真の 4DGS/STG-Lite PLY なら:

```bash
cd /home/kasm-user/Desktop/DASH/viewer
pixi run splatv -- \
  --input /path/to/point_cloud_4d.ply \
  --output web/baked/model.splatv \
  --require-4d
```

DASH の通常 `point_cloud.ply` は motion を持たないため、直接変換すると静的 preview になる。DASH の learned deformation を `.splatv` に寄せる場合は、DASH sidecar で複数時刻をサンプリングし、splaTV の `motion_*` / `omega_*` / `trbf_*` に fit する必要がある。詳細は `viewer/docs/SPLATV.md` を参照。

---

## 7. 使ったリポジトリ / 参照元

| リポジトリ / パス | 用途 | 記録時点 |
|---|---|---|
| `/home/kasm-user/Desktop/DASH` / `git@github.com:yuki-inaho/DASH.git` | DASH 本体、uv 移行、processed COLMAP loader、viewer、splaTV exporter | `main`, `b3a28d4` |
| `/home/kasm-user/Desktop/gluemap` / `git@github.com:yuki-inaho/gluemap.git` | ALIKED + LightGlue + GLOMAP/COLMAP pipeline、400-frame sparse reconstruction | `cu129-l4`, `baa8598` |
| `/home/kasm-user/Desktop/colmap` / `git@github.com:yuki-inaho/colmap.git` | COLMAP fork / Python image processing dependency 調整の確認 | `main`, `a8f717b` |
| `temp/colmap_refs/COLMAP_SLAM` / `https://github.com/3DOM-FBK/COLMAP_SLAM.git` | COLMAP SLAM / odometry 周辺の参照資料。DASH 本 pipeline の実行依存ではない | `df4d581` |
| `temp/colmap_refs/colmap-odometry` / `https://github.com/3DOM-FBK/colmap-odometry.git` | COLMAP odometry 周辺の参照資料。DASH 本 pipeline の実行依存ではない | `4973d0e` |
| `abist-co-ltd/wgpu-gs-viewer` | DASH viewer の vendored renderer ベース | viewer/docs/PROVENANCE.md 参照 |
| `antimatter15/splaTV` | `.splatv` 形式の参照 | viewer/docs/SPLATV.md 参照 |
| `KellerJordan/Muon` | Muon optimizer 依存 | `pyproject.toml` の `tool.uv.sources` |

### gluemap の local patch

gluemap は Ubuntu 20.04 / GLIBC 2.31 / CUDA 12.9 workstation で動かすため、vendored third-party に意図的な local patch を当てている。詳細は `/home/kasm-user/Desktop/gluemap/docs/local_patches.md`。

要点:

- Ceres: `GLUEMAP_CUDA_ARCH` / `CMAKE_CUDA_ARCHITECTURES` を尊重する。
- COLMAP: Ubuntu 20.04 の system OpenImageIO へ fallback する。
- Doppelgangers++ / MAST3R: trusted checkpoint 読込で `weights_only=False` を使う。

検証:

```bash
cd /home/kasm-user/Desktop/gluemap
just check-local-patches
just check-tva400-reconstruction
```

---

## 8. 再現・確認コマンド集

### gluemap 側

```bash
cd /home/kasm-user/Desktop/gluemap

pixi run colmap model_analyzer \
  --path results/tva_nyx650_400_aliked_lightglue_glomap/sparse/0

pixi run python -c "import pycolmap; r=pycolmap.Reconstruction('results/tva_nyx650_400_aliked_lightglue_glomap/sparse/0'); print(r.num_images(), r.num_reg_images(), r.num_points3D())"

just check-local-patches
just check-tva400-reconstruction
```

### DASH 側

```bash
cd /home/kasm-user/Desktop/DASH

uv run pytest tests/test_colmap_processed.py -q
uv run python scripts/validate_colmap_dataset.py \
  --source-path data/tva_nyx650_400_aliked_lg_glomap \
  --mapping data/tva_nyx650_400_aliked_lg_glomap/mapping.csv \
  --expect-images 400 \
  --expect-registered 400
```

### viewer 側

```bash
cd /home/kasm-user/Desktop/DASH/viewer

pixi run run
pixi run build-web
pixi run e2e-web
pixi run splatv -- --help
```

---

## 9. 注意事項

- `data/`, `output/`, `viewer/web/baked/`, `temp/` の生成物は commit しない。
- `mapping.csv` はローカル絶対パスを含むため、別マシンでは `--validate-source-paths` が失敗する可能性がある。
- pose / sparse が実データ由来かどうかは、`registered 400/400` だけでなく DB/log の ALIKED/LightGlue 証跡も見る。
- 現在の smoke モデルでは `dynamic=0` と記録されており、DASH viewer の時間変化は小さい。viewer の可視的な frame 差分はカメラ orbit によるところが大きい。
- `.splatv` は 4DGS/STG-Lite PLY なら `--require-4d` で真の 4D 変換になる。DASH の通常 PLY は learned deformation を内包していないため、そのままでは静的 preview になる。

---

## 10. 更新履歴

- `2026-06-10`: TVA_NYX650 2000→400 frame、ALIKED + LightGlue + GLOMAP/COLMAP、DASH processed dataset、viewer/splaTV までの引き継ぎメモを初版作成。
