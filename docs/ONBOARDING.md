# 環境セットアップ・オンボーディングガイド

**作成日**: `2026-06-10`
**対象**: `新しいセッション・エージェント、DASH の開発/検証を引き継ぐ開発メンバー`
**プロジェクト**: `DASH (yuki-inaho/DASH fork, upstream: chenj02/DASH)`
**目的**: `uv + CUDA 12.4 ベースの実行環境を再現し、Issue 2/3 修正後の検証と追加作業を安全に継続できるようにする`

---

## 目次

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

動的シーン向け Gaussian Splatting の学習・レンダリングを、古い conda 環境に依存せず `uv` で再現可能にします。あわせて upstream issue 2/3 の分析で特定した不具合を回帰テスト付きで固定し、N3DV 向けの仮説 config を利用できる状態にします。

### 主要コンポーネント

* `train.py`: DASH の学習エントリポイント
* `render.py`: レンダリングと FPS 計測
* `scene/gaussian_model.py`: GaussianModel 本体、dynamic mask、densify/prune 処理
* `arguments/`: config 読み込みと dataset/config 別パラメータ
* `submodules/depth-diff-gaussian-rasterization/`: CUDA rasterizer 拡張
* `submodules/simple-knn/`: CUDA KNN 拡張
* `hashencoder/`: JIT ビルドされる hash encoding CUDA 拡張
* `tests/`: Issue 2/3 修正の回帰テスト

---

## 2. 現在のプロジェクト状態

### 完了済み

| 分類 | 状態 | 説明 |
| --- | --- | --- |
| uv 移行 | 完了 | `pyproject.toml` と `uv.lock` を追加し、Python 3.10.13 / torch 2.5.1+cu124 で同期済み |
| CUDA 拡張 | 完了 | rasterizer、simple-knn、hashencoder JIT の 3 種を CUDA Toolkit 12.4 でビルド・import 検証済み |
| Issue 3 修正 | 完了 | `train.py` の dynamic render guard と `dynamic_image` None guard を修正 |
| Issue 2 関連修正 | 完了 | `_dynamic` dtype bool 化、prune 常時実行化、FPS 計測 synchronize 追加、N3DV config 追加 |
| テスト基盤 | 完了 | `pytest` ベースの回帰テストを追加し、`uv run pytest tests/ -q` で 10 passed を確認 |
| オンボーディング | 完了 | 本ドキュメントに再現手順と既知の落とし穴を整理 |

### 依存パッケージのインストール状態

新しいセッションでは `.venv/` と CUDA 拡張ビルド成果物が存在しない可能性があります。必ずセクション 4 の `uv sync`、submodule install、hashencoder JIT import を順に実行してください。

### 未実装・これから着手する項目

* N3DV / Technicolor 実データセットを配置した長時間学習
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
│   └── ONBOARDING.md
├── arguments/
│   ├── default.py
│   └── n3dv.py
├── scene/
│   └── gaussian_model.py
├── submodules/
│   ├── depth-diff-gaussian-rasterization/
│   └── simple-knn/
├── hashencoder/
├── tests/
│   ├── test_dynamic_dtype.py
│   ├── test_n3dv_config.py
│   ├── test_prune.py
│   └── test_report_guard.py
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

主な dev dependencies:

* `pytest`: 回帰テスト実行用

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
10 passed
```

CUDA が使えない環境では GPU 依存テストが skip されます。本プロジェクトの完了確認では CUDA 環境で実際に pass させることを基準にしてください。

### 5.4 train.py の最小スモーク

データセットがない状態でも、config 読み込みと import chain の確認はできます。

```bash
timeout 60 uv run python train.py \
  -s /tmp/nonexistent_dataset \
  --model_path /tmp/dash-smoke \
  --conf arguments/n3dv.py
```

`Find Config: arguments/n3dv.py` と merged config が出力された後、データセット不在により `AssertionError: Could not recognize scene type!` で止まるのは想定内です。

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
* `tests/`: F1/F2/F3/F4 の回帰テスト

### 7.2 実データセットでの検証

N3DV または Technicolor を配置したら、次を優先します。

1. `arguments/n3dv.py` で短時間学習を実行し、Gaussian 数と prune の挙動を確認する
2. 学習後に `render.py` で FPS を測定する
3. `metrics.py` で PSNR/SSIM/LPIPS を測定する
4. N3DV 仮説値を必要に応じて調整する

### 7.3 便利なコマンド集

```bash
git status --short
uv pip list
uv run pytest tests/ -q
uv run python -m py_compile train.py render.py scene/gaussian_model.py arguments/n3dv.py
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
* [ ] `uv run pytest tests/ -q` が CUDA 環境で `10 passed` になった
* [ ] `arguments/n3dv.py` の仮説値が目的に合っていることを理解した
* [ ] 実データセット検証は別作業であることを確認した

---

## 9. 更新履歴

* `2026-06-10 06:48:21 UTC` 初版作成。uv/CUDA 12.4 セットアップ、Issue 2/3 修正、N3DV config、テスト手順を整理。

---

## このドキュメントについて

本ガイドは、新しいセッションや新規参加メンバーが、短時間で同一の開発・実行環境を再現し、DASH の修正・検証作業を引き継げるようにするためのものです。環境セットアップで問題が発生した場合は、本ドキュメントのトラブルシューティングと、必要に応じてリポジトリのコミットログを参照してください。
