# DASH オンボーディングガイド

**作成日**: `2026-06-10`
**対象**: `新しいセッション・エージェント、DASH の開発/検証を引き継ぐ開発メンバー`
**プロジェクト**: `DASH (yuki-inaho/DASH fork, upstream: chenj02/DASH)`
**目的**: `uv + CUDA 12.4 ベースの実行環境、COLMAP/GLOMAP 処理済みデータ読込、Hydra/MUON 設定、短時間学習検証を安全に引き継げるようにする`

---

## 0. LLMオンボーディングサマリー

### プロジェクト概要と目的

- **プロジェクト名称・領域:** DASH。動的シーン向け 4D Gaussian Splatting / dynamic scene rendering。
- **最終成果物:** CUDA 12.4 + uv 環境で、COLMAP/GLOMAP 処理済みデータを DASH に読み込み、学習・レンダリング・検証できる再現可能な fork。
- **価値:** 古い conda 前提や未検証の fallback を避け、実データの pose / sparse reconstruction から DASH 学習までを追跡可能にする。
- **現時点の進捗サマリ:** Issue 2/3 修正、uv 移行、Hydra config、Muon + Schedule-Free optimizer、TensorBoard metadata、processed COLMAP dataset validation、TVA 400-frame short train まで完了。基準実装 commit `923101f` は `origin/main` に push 済み。

### クリティカルな要求・制約

- dummy/random/synthetic pose や random point cloud への fallback は禁止。実データ由来であることを DB/log/model analyzer/pycolmap で確認する。
- `data/`, `output/`, SfM 結果、TensorBoard event、checkpoint は Git に入れない。
- DASH 側は `uv`、gluemap/COLMAP 側は `pixi` を使う。環境を暗黙に混ぜない。
- CUDA 拡張は CUDA Toolkit 12.4 / torch 2.5.1+cu124 を前提にする。
- 既存の通常 COLMAP loader を壊さず、processed dataset のときだけ `mapping.csv` を検証して source frame 由来 `fid` を使う。
- `ty` は legacy code 全体では既存診断が残る。今回追加ファイル単位での `ty` 成功と、全体 `pytest` 成功を品質ゲートにする。

### 参照すべき合意済み資料

| 種別 | ファイル/リンク | 概要・用途 |
| --- | --- | --- |
| 環境/引継ぎ | `docs/ONBOARDING.md` | 本ドキュメント。環境、制約、検証コマンド、次タスクを集約 |
| データ lineage | `docs/DATA_COLMAP_WORKFLOW.md` | TVA_NYX650 元画像、400-frame subset、ALIKED + LightGlue + GLOMAP/COLMAP、DASH 接続、使用リポジトリのメモ |
| 作業記録 | `temp/workdoc_Jun10-2026_dash_hydra_types_optimizer_refactor.md` | Hydra、型ヒント、optimizer、TensorBoard 追加の作業証跡 |
| 作業記録 | `temp/workdoc_Jun10-2026_full_colmap_to_dash_training.md` | 400-frame COLMAP/GLOMAP 検証、DASH loader/CLI、short train 証跡 |
| テスト資産 | `tests/` | 回帰テスト、Hydra/optimizer/TensorBoard/processed COLMAP validation |
| データ検証 CLI | `scripts/validate_colmap_dataset.py` | processed COLMAP dataset の mapping/sparse/images 検証 |
| 実装入口 | `train.py`, `render.py`, `scene/dataset_readers.py` | 学習、レンダリング、dataset 読込 |

### タスク境界

任せるタスク:

- 回帰テスト付きの小さな修正、loader/CLI/config/test の追加。
- 既存 workdoc に沿った検証、証跡収集、commit 前の git hygiene 確認。
- 短時間 smoke train / render / TensorBoard 確認。

任せないタスク:

- データセットや学習出力の commit。
- 長時間学習の結果を論文値として断定すること。
- 未検証の pose 補完・dummy fallback・best checkpoint ロジック変更。
- ユーザー指示なしの force push や履歴破壊。

### インタラクション方針

- **回答スタイル:** 日本語、結論先行、コマンドとパスを明示。
- **回答手順:** 前提確認、実行内容、検証結果、残リスクの順で報告する。
- **禁止事項・注意:** 未確定事項を断定しない。データ/出力を Git に混ぜない。環境 fallback を成功扱いしない。
- **秘匿情報の扱い:** Git remote やローカルパスは必要最小限で扱い、認証情報や秘密鍵は表示しない。

### 試行タスク

1. `uv run pytest tests/test_colmap_processed.py -q` を実行し、processed COLMAP validation の基本を確認する。
2. `uv run python scripts/validate_colmap_dataset.py ... --expect-images 400 --expect-registered 400` を実行し、実データがある場合だけ FULL 証跡を確認する。
3. `output/tva_nyx650_400_smoke` がある場合、TensorBoard または `render.py` で smoke 結果を確認する。

### 運用ルール・変更管理

- ドキュメント更新時は、実コマンド結果・日付・対象 commit を更新履歴に残す。
- TBD は放置せず、外部データ待ち・legacy debt・ユーザー判断待ちのどれかに分類する。
- 大きな作業は workdoc を作り、worker 実施、auditor 承認、coordinator 受理の順で閉じる。
- push 前に `git status --short --branch` と `git show --stat --oneline HEAD` を確認する。

---

## 目次

0. [LLMオンボーディングサマリー](#0-llmオンボーディングサマリー)
1. [プロジェクト概要](#1-プロジェクト概要)
2. [現在のプロジェクト状態](#2-現在のプロジェクト状態)
3. [前提条件の確認](#3-前提条件の確認)
4. [環境セットアップ手順](#4-環境セットアップ手順)
5. [動作確認](#5-動作確認)
6. [トラブルシューティング](#6-トラブルシューティング)
7. [次のステップ](#7-次のステップ)
8. [環境セットアップ完了チェックリスト](#8-環境セットアップ完了チェックリスト)
9. [更新履歴](#9-更新履歴)

---

## 1. プロジェクト概要

### プロジェクト名

**DASH**
DASH: 4D Hash Encoding with Self-Supervised Decomposition for Real-Time Dynamic Scene Rendering の公式実装 fork です。

### 最終目標

動的シーン向け Gaussian Splatting の学習・レンダリングを、古い conda 環境に依存せず `uv` で再現可能にします。あわせて upstream issue 2/3 の分析で特定した不具合を回帰テスト付きで固定し、COLMAP/GLOMAP 処理済みデータを DASH 側で検証・読込・短時間学習できる状態にします。

### 主要コンポーネント

* `train.py`: DASH の学習エントリポイント
* `render.py`: レンダリングと FPS 計測
* `scene/gaussian_model.py`: GaussianModel 本体、dynamic mask、densify/prune 処理
* `scene/colmap_processed.py`: processed COLMAP/GLOMAP dataset の mapping/sparse/images 検証
* `scripts/validate_colmap_dataset.py`: processed dataset 検証 CLI
* `arguments/`: config 読み込みと dataset/config 別パラメータ
* `configs/train/`: Hydra YAML config
* `utils/optimizer_utils.py`: Adam / Schedule-Free / Muon optimizer helper
* `utils/tensorboard_utils.py`: TensorBoard metadata helper
* `submodules/depth-diff-gaussian-rasterization/`: CUDA rasterizer 拡張
* `submodules/simple-knn/`: CUDA KNN 拡張
* `hashencoder/`: JIT ビルドされる hash encoding CUDA 拡張
* `tests/`: Issue 2/3、Hydra、optimizer、TensorBoard、processed COLMAP の回帰テスト

---

## 2. 現在のプロジェクト状態

### 完了済み

| 分類 | 状態 | 説明 |
| --- | --- | --- |
| uv 移行 | 完了 | `pyproject.toml` と `uv.lock` を追加し、Python 3.10.13 / torch 2.5.1+cu124 で同期済み |
| CUDA 拡張 | 完了 | rasterizer、simple-knn、hashencoder JIT の 3 種を CUDA Toolkit 12.4 でビルド・import 検証済み |
| Issue 3 修正 | 完了 | `train.py` の dynamic render guard と `dynamic_image` None guard を修正 |
| Issue 2 関連修正 | 完了 | `_dynamic` dtype bool 化、prune 常時実行化、FPS 計測 synchronize 追加、N3DV config 追加 |
| Hydra / optimizer | 完了 | `configs/train/tva400_muon_schedulefree.yaml`、Muon + Schedule-Free optimizer、TensorBoard metadata を追加 |
| processed COLMAP | 完了 | `mapping.csv` と sparse model を検証する loader helper / CLI / tests を追加 |
| TVA 400-frame smoke | 完了 | ALIKED + LightGlue + GLOMAP/COLMAP 由来の 400-frame reconstruction を検証し、DASH short train 20 iteration を実行 |
| テスト基盤 | 完了 | `pytest` ベースの回帰テストを追加し、`uv run pytest -q` で 26 passed を確認 |
| オンボーディング | 完了 | 本ドキュメントに再現手順と既知の落とし穴を整理 |

### 依存パッケージのインストール状態

新しいセッションでは `.venv/` と CUDA 拡張ビルド成果物が存在しない可能性があります。必ずセクション 4 の `uv sync`、submodule install、hashencoder JIT import を順に実行してください。

### 未実装・これから着手する項目

* N3DV / Technicolor / TVA などの実データセットを使った長時間学習
* 論文値に対する PSNR/FPS の実測比較
* `arguments/n3dv.py` の仮説値の実験的検証と調整
* upstream への PR 作成や issue へのフィードバック

### 重要なファイル／ディレクトリ

```text
/home/kasm-user/Desktop/DASH/
├── README.md
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── DATA_COLMAP_WORKFLOW.md
│   └── ONBOARDING.md
├── arguments/
│   ├── default.py
│   └── n3dv.py
├── configs/train/
│   └── tva400_muon_schedulefree.yaml
├── scripts/
│   └── validate_colmap_dataset.py
├── scene/
│   ├── colmap_processed.py
│   └── gaussian_model.py
├── utils/
│   ├── optimizer_utils.py
│   └── tensorboard_utils.py
├── submodules/
│   ├── depth-diff-gaussian-rasterization/
│   └── simple-knn/
├── hashencoder/
├── tests/
│   ├── test_dynamic_dtype.py
│   ├── test_n3dv_config.py
│   ├── test_optimizer_utils.py
│   ├── test_prune.py
│   ├── test_report_guard.py
│   ├── test_tensorboard_utils.py
│   └── test_colmap_processed.py
├── temp/
│   ├── workdoc_Jun10-2026_dash_hydra_types_optimizer_refactor.md
│   └── workdoc_Jun10-2026_full_colmap_to_dash_training.md
├── train.py
└── render.py
```

---

## 3. 前提条件の確認

新しいセッションで環境を再現する前に、以下を確認してください。

### 3.1 システム情報の確認

```bash
cat /etc/os-release | grep -E "^(NAME|VERSION)="
uname -r
nproc
free -h
pwd
```

作成時点の検証環境は Ubuntu 20.04.6 LTS、kernel `6.8.0-57-generic`、64 cores、約 503 GiB RAM、作業ディレクトリ `/home/kasm-user/Desktop/DASH` です。

### 3.2 必須ツールの存在確認

```bash
which uv && uv --version
which git && git --version
which ninja && ninja --version
which /usr/local/cuda-12.4/bin/nvcc && /usr/local/cuda-12.4/bin/nvcc --version
nvidia-smi
```

期待値:

```text
uv 0.11.19 以上
CUDA compilation tools, release 12.4, V12.4.131
NVIDIA GPU が認識されていること
```

### 3.3 Git ブランチ・コミット確認

```bash
git branch --show-current
git log --oneline -1
git status --short
```

通常は `main` で作業します。作業前に未コミット差分がある場合は、自分の作業対象と衝突しないか確認してください。

---

## 4. 環境セットアップ手順

### 4.1 CUDA Toolkit 12.4 の確認／インストール

CUDA 拡張のビルドには driver だけでなく `nvcc` が必要です。

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

nvcc --version
```

`nvcc` がない場合は NVIDIA の CUDA apt repository を設定し、CUDA Toolkit 12.4 をインストールしてください。既存の `/etc/apt/sources.list.d/cuda.list` と keyring 付き repository が競合する場合があります。その場合は古いエントリを無効化してから `apt-get update` を実行します。

### 4.2 Python 依存パッケージのインストール

```bash
cd /home/kasm-user/Desktop/DASH
uv sync --no-install-project
```

主な dependencies:

* `torch==2.5.1` / `torchvision==0.20.1`: CUDA 12.4 wheel を使用
* `numpy<2`: 既存コードとの互換性維持
* `opencv-python`, `pillow`, `plyfile`, `pytorch-msssim`, `lpips`, `scipy`, `imageio`, `imageio-ffmpeg`, `tqdm`, `tensorboard`
* `ninja`, `setuptools`: CUDA extension / JIT build に必要
* `hydra-core`, `omegaconf`: Hydra YAML config merge
* `jaxtyping`, `beartype`: 型ヒントと runtime type check
* `schedulefree`, `muon-optimizer`: Schedule-Free / Muon optimizer

主な dev dependencies:

* `pytest`: 回帰テスト実行用
* `ty`: 変更ファイル単位の型チェック
* `rust`: ツール検証用 dependency

確認:

```bash
uv run python --version
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
uv run pytest --version
```

### 4.3 CUDA submodule と hashencoder のセットアップ

```bash
cd /home/kasm-user/Desktop/DASH
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

uv pip install --no-build-isolation -e ./submodules/depth-diff-gaussian-rasterization
uv pip install --no-build-isolation ./submodules/simple-knn
```

`simple-knn` は editable install では package import が壊れるため、非 editable install を使います。

hashencoder は import 時に JIT build されます。

```bash
uv run python - <<'PY'
import torch
from hashencoder.hashgrid import HashEncoder

enc = HashEncoder(
    input_dim=4,
    num_levels=2,
    level_dim=2,
    base_resolution=[8, 8, 8, 8],
    desired_resolution=[16, 16, 16, 16],
    log2_hashmap_size=12,
).cuda()
x = torch.rand(8, 4, device="cuda")
print(enc(x).shape)
PY
```

### 4.4 環境変数の永続化（任意）

毎回 CUDA 12.4 を使う場合のみ、shell 設定に追加してください。

```bash
printf '%s\n' \
  'export CUDA_HOME=/usr/local/cuda-12.4' \
  'export PATH="$CUDA_HOME/bin:$PATH"' \
  'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"' >> ~/.bashrc
```

---

## 5. 動作確認

### 5.1 GPU と torch の確認

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

### 5.2 CUDA 拡張の確認

```bash
uv run python - <<'PY'
import torch
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizer

pts = torch.rand(100, 3, device="cuda")
print(distCUDA2(pts).shape)
print(GaussianRasterizer)
PY
```

### 5.3 テスト実行

```bash
uv run pytest tests/ -q
```

期待値:

```text
26 passed
```

CUDA が使えない環境では GPU 依存テストが skip される場合があります。本プロジェクトの完了確認では CUDA 環境で実際に pass させることを基準にしてください。

### 5.4 processed COLMAP dataset の確認

TVA 400-frame のローカルデータが存在する環境では、次で DASH 側の processed COLMAP loader を確認できます。`data/` は Git 管理外なので、clone 直後の環境には存在しない場合があります。

```bash
uv run python scripts/validate_colmap_dataset.py \
  --source-path data/tva_nyx650_400_aliked_lg_glomap \
  --mapping data/tva_nyx650_400_aliked_lg_glomap/mapping.csv \
  --expect-images 400 \
  --expect-registered 400 \
  --validate-source-paths
```

期待値:

```text
images: 400
mapping_rows: 400
registered: 400
points: 41333
camera_models: PINHOLE
source_paths_validated: True
```

### 5.5 train.py の最小スモーク

データセットがない状態でも、config 読み込みと import chain の確認はできます。

```bash
timeout 60 uv run python train.py \
  -s /tmp/nonexistent_dataset \
  --model_path /tmp/dash-smoke \
  --conf arguments/n3dv.py
```

`Find Config: arguments/n3dv.py` と merged config が出力された後、データセット不在により `AssertionError: Could not recognize scene type!` で止まるのは想定内です。

### 5.6 TVA 400-frame short train

processed dataset がある場合は、Hydra config と Muon + Schedule-Free optimizer を含む short train を実行できます。

```bash
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

確認対象:

```text
output/tva_nyx650_400_smoke/events.out.tfevents*
output/tva_nyx650_400_smoke/point_cloud/iteration_20/point_cloud.ply
output/tva_nyx650_400_smoke/deform/iteration_20/deform.pth
```

---

## 6. トラブルシューティング

### 問題1: `nvcc: command not found`

**原因**: CUDA driver はあるが CUDA Toolkit が未導入、または PATH が通っていません。

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
```

### 問題2: CUDA apt repository の署名・重複エラー

**原因**: 古い `cuda.list` と keyring 付き repository が同時に存在しています。

**対処**: `/etc/apt/sources.list.d/` 配下の CUDA repository 設定を確認し、古い署名なしエントリを無効化してから `apt-get update` を再実行します。

### 問題3: `simple_knn` が import できない

**原因**: `simple-knn` を editable install すると package metadata と import path が合わないことがあります。

```bash
uv pip uninstall -y simple-knn simple_knn 2>/dev/null || true
uv pip install --no-build-isolation ./submodules/simple-knn
uv run python -c "from simple_knn._C import distCUDA2; print(distCUDA2)"
```

### 問題4: hashencoder JIT build が失敗する

**原因**: `ninja`、`setuptools`、`nvcc`、CUDA 環境変数のいずれかが不足している可能性があります。

```bash
uv pip list | grep -E "ninja|setuptools|torch"
nvcc --version
rm -rf tmp_build/
uv run python -c "import hashencoder.backend; print('hashencoder backend ok')"
```

### 問題5: `uv sync` で `pyproject.toml` が見つからない

**原因**: 作業ディレクトリがリポジトリルートではありません。

```bash
cd /home/kasm-user/Desktop/DASH
ls pyproject.toml uv.lock
```

### 問題6: N3DV / Technicolor の学習が始まらない

**原因**: dataset path が存在しない、または想定形式と異なります。リポジトリには実データセットは含まれていません。

```bash
ls /data
uv run python train.py -s <dataset_path> --model_path <output_path> --conf arguments/n3dv.py
```

### 問題7: processed COLMAP dataset の `source_path` が別環境で存在しない

通常の DASH loader は `source_path` の存在を必須にしていません。処理済み dataset を別環境へ移した場合でも、`images/`, `sparse/0`, `mapping.csv` が揃っていれば読込可能です。元 source image まで含めて provenance を検証したい場合だけ CLI に `--validate-source-paths` を付けてください。

### 問題8: `ty check scene/dataset_readers.py` が失敗する

`scene/dataset_readers.py` には既存 legacy 型診断が残っています。今回の品質ゲートでは、追加ファイル単位で次が通ることを確認しています。

```bash
uv run ty check scene/colmap_processed.py scripts/validate_colmap_dataset.py tests/test_colmap_processed.py
```

---

## 7. 次のステップ

### 7.1 修正内容の把握

```bash
git show --stat
git show -- train.py render.py scene/gaussian_model.py arguments/n3dv.py tests/
```

確認観点:

* `train.py`: dynamic render guard と `dynamic_image` None guard
* `scene/gaussian_model.py`: `_dynamic` bool 化と prune 常時実行
* `render.py`: FPS 計測前後の `torch.cuda.synchronize()`
* `arguments/n3dv.py`: N3DV 仮説 config
* `configs/train/tva400_muon_schedulefree.yaml`: Muon + Schedule-Free training config
* `scene/colmap_processed.py`: processed COLMAP 検証
* `scripts/validate_colmap_dataset.py`: processed COLMAP 検証 CLI
* `tests/`: F1/F2/F3/F4、Hydra、optimizer、TensorBoard、processed COLMAP の回帰テスト

### 7.2 実データセットでの検証

N3DV、Technicolor、TVA などの実データセットを配置したら、次を優先します。

1. `scripts/validate_colmap_dataset.py` で pose / sparse / mapping の整合性を確認する
2. `arguments/n3dv.py` と Hydra config で短時間学習を実行し、Gaussian 数と prune の挙動を確認する
3. TensorBoard で loss/config/optimizer metadata を確認する
4. 学習後に `render.py` で FPS と出力画像を確認する
5. `metrics.py` で PSNR/SSIM/LPIPS を測定する
6. N3DV 仮説値を必要に応じて調整する

### 7.3 便利なコマンド集

```bash
git status --short
uv pip list
uv run pytest -q
uv run python -m py_compile train.py render.py scene/gaussian_model.py arguments/n3dv.py
uv run ty check scene/colmap_processed.py scripts/validate_colmap_dataset.py tests/test_colmap_processed.py
env | grep -E "CUDA_HOME|LD_LIBRARY_PATH|PATH"
```

---

## 8. 環境セットアップ完了チェックリスト

以下をすべて確認してから、実装・検証作業に進んでください。

* [ ] システム情報を確認した
* [ ] `uv`、`git`、`ninja`、`nvcc`、`nvidia-smi` が利用できる
* [ ] `CUDA_HOME=/usr/local/cuda-12.4` を設定した
* [ ] `uv sync --no-install-project` が成功した
* [ ] `torch.cuda.is_available()` が `True` である
* [ ] rasterizer と simple-knn の CUDA 拡張をインストールした
* [ ] hashencoder JIT build が成功した
* [ ] `uv run pytest -q` が CUDA 環境で `26 passed` になった
* [ ] `arguments/n3dv.py` の仮説値が目的に合っていることを理解した
* [ ] processed COLMAP dataset がある場合は CLI 検証を実行した
* [ ] `data/` と `output/` は Git 管理外であることを確認した

---

## 9. 更新履歴

* `2026-06-10 06:48:21 UTC` 初版作成。uv/CUDA 12.4 セットアップ、Issue 2/3 修正、N3DV config、テスト手順を整理。
* `2026-06-10 11:35:00 UTC` 現状に合わせて更新。LLM オンボーディングサマリー、Hydra/Muon/Schedule-Free/TensorBoard、processed COLMAP loader/CLI、TVA 400-frame smoke、`26 passed` の検証状態を反映。
* `2026-06-10 14:35:00 UTC` `docs/DATA_COLMAP_WORKFLOW.md` を追加し、元データから COLMAP/GLOMAP、DASH、viewer/splaTV までの lineage と使用リポジトリへの導線を追記。

---

## このドキュメントについて

本ガイドは、新しいセッションや新規参加メンバーが、短時間で同一の開発・実行環境を再現し、DASH の修正・検証作業を引き継げるようにするためのものです。環境セットアップで問題が発生した場合は、本ドキュメントのトラブルシューティングと、必要に応じてリポジトリのコミットログを参照してください。
