# 作業計画書 兼 記録書: DASH Hydra・型強化・optimizer factory・TensorBoard リファクタリング

---

**日付：** 2026年06月10日
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/DASH` / DASH uv project (`pyproject.toml`, `uv.lock`)
**作業者：** 統括/作業/監査エージェントチーム。書紀: Hypatia
**作成時刻：** 2026-06-10 10:01:43 UTC+0000(Coordinator 確認)

---

## 1. 作業目的

本日の作業は、以下の目標を達成するために実施します。

*   **目標1:** 既存 DASH 学習 CLI の挙動を保ちながら、Hydra 設定管理を追加導入できる構造へ分離する。
*   **目標2:** `jaxtyping` / `beartype`、optimizer factory、TensorBoard helper を軽量な境界から導入し、SOLID/KISS/DRY に沿って再利用性を高める。
*   **目標3:** MUON + schedulefree optimizer を選択可能にする調査・実装・検証を行い、回帰テスト、型チェック、smoke、commit まで完了する。

### 1.1 ゴール要求分析

*   **ユーザーの直観的・直截的な目的:** 既存の DASH 学習を壊さず、設定・型・optimizer・監視を整理し、今後の実験を Hydra と uv ベースで安全に拡張できる状態にする。
*   **明示要求:**
    *   commit まで行う。
    *   回帰テストを書く。
    *   SOLID/KISS/DRY に沿って再利用性のよい形にリファクタリングする。
    *   `jaxtyping`, `beartype` で型ヒントを強化する。
    *   `/workspace/Project/DEIM_sandbox` を参考に Hydra で設定管理できるようにする。参照先が不存在/空なら明示的に記録し、暗黙 fallback しない。
    *   TensorBoard で監視できるようにする。
    *   MUON + schedulefree optimizer を使える形にする。ユーザー要求上の `schedulerfree` は PyPI 名としては解決不可のため、実パッケージ名 `schedulefree==1.4.1` / import 名 `schedulefree` を採用する。
    *   `pytest`, `ty`, `rust` を uv add し、依存追加も記録する。
    *   write/review スキルで作業書を作成・レビューしてから DoD まで止めずに進める。
*   **既知状態:**
    *   `train.py`, `arguments/__init__.py`, `tests/test_n3dv_config.py` に CLI override 修正の既存変更がある。
    *   `uv run pytest tests/test_n3dv_config.py tests/test_report_guard.py` は 6 passed 済み。
    *   `uv add --dev pytest ty rust` 実行済み。`ty==0.0.46`, `rust==1.3.1` などが入り、`uv.lock` は更新済み。pytest は既存 dev dependency。
    *   `uv add "muon-optimizer @ git+https://github.com/KellerJordan/Muon"` 実行済み。`uv.lock` は `muon-optimizer` の Git source `https://github.com/KellerJordan/Muon#f98f1cacc0263b04290753e32be8d498c1efc806` を指す。package 名は `muon-optimizer`、Python import 名は `muon`。
    *   現環境の import 解決確認では `schedulefree=True`, `schedulerfree=False`, `muon=True`, `rust=False`。`rust==1.3.1` は Python import 名 `rust` での検証対象ではなく、Rust toolchain 導入確認にも使わない。
    *   DASH 接続済みデータセットは `/home/kasm-user/Desktop/DASH/data/tva_nyx650_400_aliked_lg_glomap`。これは成果物であり、原則 commit 対象から除外/ignore 方針。
    *   DASH リポジトリ直下に `AGENTS.md` / `CODEX.md` / `CLAUDE.md` / `justfile` は見当たらない。ローカル追加指示は本書とユーザー指示を正とする。
    *   `/workspace/Project/DEIM_sandbox` は本レビュー時点で存在しない。近傍に見える `tomato_optim` ディレクトリ群は別プロジェクトの参考実装であり、DEIM_sandbox の代替参照先として暗黙採用しない。
*   **暗黙制約:**
    *   uv 環境で作業する。コマンドは原則 `/home/kasm-user/Desktop/DASH` から実行する。
    *   既存 `--conf arguments/n3dv.py` を壊さない。Hydra は追加 CLI `--hydra_config` / `--hydra_overrides` 等で段階導入する。
    *   CLI 明示 override が config 読み込み後に勝つ回帰テストを維持・拡張する。
    *   MUON/schedulefree は依存可否を調査し、利用可能なら factory で選択式にする。利用不可なら明示的にブロック/代替判断を記録し、silent fallback しない。
    *   TensorBoard は既存 `SummaryWriter` があれば活かし、監視に必要な設定値/optimizer名/iteration/loss/point count 等が記録されることをテスト可能にする。
    *   `jaxtyping` / `beartype` は設定 dataclass/helper/factory など軽量な境界から入れ、GPU hot path を壊さない。
    *   data/ 以下の大きい成果物や symlink は commit しない。必要なら `.gitignore` 対応を検討する。
*   **非ゴール:**
    *   学習器全体の大改造、GPU hot path への全面的な runtime type check 導入。
    *   データセット成果物の commit。
    *   `/workspace/Project/DEIM_sandbox` が不存在/空の場合の推測実装。
    *   MUON/schedulefree の挙動を未検証のまま既定 optimizer に置き換えること。
*   **成功条件:**
    *   作業書レビューが完了し、Plan が実行可能である。
    *   設定/CLI/optimizer factory/TensorBoard helper/type utility が分離され、既存挙動が維持される。
    *   Hydra は後方互換 CLI と併存し、CLI 明示 override 優先のテストが通る。
    *   `jaxtyping` / `beartype` が軽量境界に導入され、型チェック・py_compile・pytest が通る。
    *   MUON + schedulefree が利用可能なら選択式に動作し、不可なら理由と代替判断が作業記録に残る。
    *   TensorBoard 監視 helper がテスト可能で、必要 scalar/text/hparam 相当の記録が確認できる。
    *   `uv run pytest tests`, `uv run ty check .` または実際に確認した ty コマンド、`uv run python -m py_compile ...`, DASH smoke が完了する。
    *   data/ 成果物を除外し、適切な差分だけを commit する。
*   **リスクと前提:**
    *   `ty` はバージョン 0.0.46 のためコマンド形が変更中の可能性がある。`uv run ty --help` で実コマンドを確認し、暗黙 fallback しない。
    *   `rust==1.3.1` は Python パッケージ依存として lock されているが、`import rust` は解決できないため import 検証しない。Rust toolchain の確認が必要な場合は `rustc --version` など別コマンドで実施し、`rust` PyPI package と混同しない。
    *   `schedulerfree` は PyPI 名として解決不可で、実パッケージは `schedulefree==1.4.1`、import 名は `schedulefree`。作業記録ではこの名称差を必ず残す。
    *   MUON upstream は `https://github.com/KellerJordan/Muon`。README 上は hidden weights を Muon、embedding/head/bias などその他 parameter を AdamW 系 aux で最適化する前提で、MIT license。DASH へ入れる際は全 parameter を一括 Muon 化しない。
    *   DASH smoke は GPU/データセット依存のため、時間・メモリ制約がある。短時間 smoke 条件を明示して実行する。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | 現状調査と Plan レビュー | write/review、既存挙動維持 | 調査記録、レビュー記録 | 作業記録、既存テスト再実行 |
| SG-2 | 設定/CLI 責務分離 | Hydra 導入、CLI override 維持 | config/CLI helper、追加テスト | CLI override 回帰テスト |
| SG-3 | 型境界の導入 | `jaxtyping` / `beartype` | dataclass/helper/factory 型注釈 | pytest、ty、py_compile |
| SG-4 | optimizer factory | MUON + schedulefree 選択式 | optimizer factory、設定項目、テスト | optimizer 選択テスト |
| SG-5 | TensorBoard helper | 監視可能性 | SummaryWriter helper、記録テスト | mock writer/unit test |
| SG-6 | Hydra 参照・導入 | DEIM_sandbox 参照、fallback 禁止 | Hydra loader/adapter、記録 | 参照先確認ログ、設定統合テスト |
| SG-7 | 統合検証と commit | DoD、commit 要求 | テストログ、smoke ログ、commit | `uv run pytest tests`, ty, py_compile, smoke, commit hash |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | write/review 後に実行 | 手順1-3 | review 結果、作業記録 |
| TR-2 | 既存 CLI override 維持 | 手順3, 7, 11, 15, 16 | `tests/test_n3dv_config.py` など |
| TR-3 | Hydra 後方互換導入 | 手順5, 7, 11 | Hydra loader テスト、CLI テスト |
| TR-4 | DEIM_sandbox 参照・fallback 禁止 | 手順4 | 存在/空確認ログ、判断記録 |
| TR-5 | SOLID/KISS/DRY リファクタ | 手順6, 11-15 | 差分、責務分離メモ |
| TR-6 | `jaxtyping` / `beartype` | 手順10, 12 | 型境界テスト、py_compile、ty |
| TR-7 | MUON + schedulefree / `muon-optimizer` | 手順5, 9, 13 | 依存調査、factory テスト、lock commit hash |
| TR-8 | TensorBoard 監視 | 手順8, 14, 20 | writer helper テスト、ログ確認 |
| TR-9 | uv 依存追加 | 手順2, 5 | `pyproject.toml`, `uv.lock`, `uv run ...` |
| TR-10 | 回帰・統合検証 | 手順16-20 | pytest/ty/py_compile/smoke ログ |
| TR-11 | data 除外と commit | 手順21-22 | diff 確認、commit hash |

---

## 2. 作業内容

### フェーズ 1: 調査・レビュー・設計確定 (見積: 0.8h)

このフェーズでは、既存変更と依存状態を破壊せず確認し、実装前の設計を固める。

1.  **作業書レビュー:**
    *   **タスク内容:** 本書を `review-written-workdoc` 相当で確認し、曖昧な依存・DoD・手順漏れを修正する。
    *   **目的:** 他エージェントが迷わず実行できる状態にする。
    *   **対応サブゴール/Trace ID:** SG-1 / TR-1
2.  **既存変更と依存の把握:**
    *   **タスク内容:** `train.py`, `arguments/__init__.py`, `tests/test_n3dv_config.py`, `pyproject.toml`, `uv.lock` の現在状態を確認する。
    *   **目的:** CLI override 修正と `uv add --dev pytest ty rust` 済み状態を踏まえて作業する。
    *   **対応サブゴール/Trace ID:** SG-1 / TR-2 / TR-9
3.  **DEIM_sandbox と外部依存調査:**
    *   **タスク内容:** `/workspace/Project/DEIM_sandbox` の存在・内容、Hydra の参考実装、schedulefree/MUON パッケージ可否、TensorBoard/jaxtyping/beartype の導入状態を調査する。
    *   **目的:** 暗黙 fallback を避け、導入可否を明示する。
    *   **対応サブゴール/Trace ID:** SG-4 / SG-6 / TR-4 / TR-7
4.  **設計方針の確定:**
    *   **タスク内容:** 設定/CLI/optimizer factory/TensorBoard/type utility の責務分離方針とファイル単位を決める。
    *   **目的:** いきなり学習器全体を大改造せず、既存挙動を保つ小さな変更にする。
    *   **対応サブゴール/Trace ID:** SG-2〜SG-6 / TR-3 / TR-5

### フェーズ 2: TDD と実装 (見積: 2.5h)

このフェーズでは、先に回帰テストを追加・失敗確認し、その後に小さく実装する。

1.  **CLI/Hydra 回帰テスト:**
    *   **タスク内容:** `--conf arguments/n3dv.py` と CLI 明示 override の優先順位、追加 Hydra CLI の併存をテストする。
    *   **目的:** 既存 CLI を壊さず Hydra を導入する。
    *   **対応サブゴール/Trace ID:** SG-2 / SG-6 / TR-2 / TR-3
2.  **型境界テスト:**
    *   **タスク内容:** 設定 dataclass/helper/factory に `jaxtyping` / `beartype` を導入しても通常入力が通り、不正入力が明示的に落ちるテストを追加する。
    *   **目的:** GPU hot path を避けつつ型安全性を上げる。
    *   **対応サブゴール/Trace ID:** SG-3 / TR-6
3.  **optimizer factory テスト:**
    *   **タスク内容:** 既存 optimizer、schedulefree、MUON 選択時の生成ロジックをテストする。依存不可なら明示エラーを期待する。
    *   **目的:** silent fallback なしで optimizer を切り替え可能にする。
    *   **対応サブゴール/Trace ID:** SG-4 / TR-7
4.  **TensorBoard helper テスト:**
    *   **タスク内容:** mock writer または tmp logdir で設定値/optimizer名/iteration/loss/point count が記録されることを確認する。
    *   **目的:** 監視機能をテスト可能にする。
    *   **対応サブゴール/Trace ID:** SG-5 / TR-8
5.  **実装:**
    *   **タスク内容:** 設定/CLI adapter、Hydra loader、type utility、optimizer factory、TensorBoard helper を追加または既存から抽出し、`train.py` から利用する。
    *   **目的:** 既存挙動を保ちながら責務を分離する。
    *   **対応サブゴール/Trace ID:** SG-2〜SG-6 / TR-2〜TR-8

### フェーズ 3: 検証・DASH smoke (見積: 1.0h)

このフェーズでは、単体・型・構文・全体テストと DASH smoke を実行する。

1.  **対象回帰テスト:**
    *   **タスク内容:** 既存 6 passed を含む設定系・report guard・追加テストを実行する。
    *   **目的:** CLI override と新 helper の fail-to-pass を確認する。
    *   **対応サブゴール/Trace ID:** SG-7 / TR-10
2.  **全テストと型チェック:**
    *   **タスク内容:** `uv run pytest tests`、`uv run ty check .` または `uv run ty --help` で確認した正しい ty コマンドを実行する。
    *   **目的:** プロジェクト全体の回帰と型問題を確認する。
    *   **対応サブゴール/Trace ID:** SG-7 / TR-10
3.  **py_compile:**
    *   **タスク内容:** 変更した Python ファイルを `uv run python -m py_compile ...` で確認する。
    *   **目的:** runtime 前の構文エラーを排除する。
    *   **対応サブゴール/Trace ID:** SG-7 / TR-10
4.  **DASH smoke:**
    *   **タスク内容:** `/home/kasm-user/Desktop/DASH/data/tva_nyx650_400_aliked_lg_glomap` を使い、短時間・低負荷条件で train smoke を実行する。
    *   **目的:** 実データ接続、設定統合、TensorBoard 記録の実行可能性を確認する。
    *   **対応サブゴール/Trace ID:** SG-7 / TR-8 / TR-10

### フェーズ 4: 差分整理・commit (見積: 0.4h)

このフェーズでは、data 成果物除外を確認してから commit する。

1.  **差分確認と ignore 方針:**
    *   **タスク内容:** data/ 以下の大きい成果物や symlink が commit 対象に入っていないことを確認し、必要なら `.gitignore` 追記を検討する。
    *   **目的:** 成果物混入を防ぐ。
    *   **対応サブゴール/Trace ID:** SG-7 / TR-11
2.  **commit:**
    *   **タスク内容:** 適切なソース・テスト・lock・作業書のみを stage し、ユーザー要求に従い commit する。
    *   **目的:** 作業成果を明示的な履歴として残す。
    *   **対応サブゴール/Trace ID:** SG-7 / TR-11

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更します。各 `[x]` は監査承認後に確定します。*

### フェーズ 1: 調査・レビュー・設計確定

### 手順 1: 作業書レビューを実施する【TR-1】
- [x] 🖐 **操作**: 本書を `review-written-workdoc` 相当で読み、ゴール要求分析、DoD、チェックリスト、Trace ID の不足を確認する。
- [x] 🔎 **確認**: 実行者が迷う未定義語、空欄、矛盾がない。修正した場合は作業記録に理由を残す。
- [x] 🧪 **テスト**: 作業書レビューのため自動テスト不要。レビュー結果が PASS または修正済みであることを記録する。
- [x] 🛠 **エラー時対処**: 仕様が曖昧な場合は、選択肢と採用基準を本書へ追記し、統括へ確認する。

### 手順 2: uv 依存追加済み状態を確認する【TR-9】
- [x] 🖐 **操作**: `pyproject.toml` と `uv.lock` を読み、`pytest`, `ty==0.0.46`, `rust==1.3.1` など `uv add --dev pytest ty rust` の反映を確認する。`rust` は `import rust` では検証せず、lock/pyproject の依存として確認する。
- [x] 🔎 **確認**: 依存追加済み/未反映の状態が作業記録に残り、pytest が既存 dev dependency だったこと、`rust==1.3.1` は Python import 名検証対象ではないことも記録されている。
- [x] 🧪 **テスト**: `uv run pytest tests/test_n3dv_config.py tests/test_report_guard.py` を再実行し、既知の 6 passed を再確認する。
- [x] 🛠 **エラー時対処**: lock と pyproject が不一致なら `uv sync` の要否を記録し、勝手に依存を増やさず統括へ確認する。

### 手順 3: 既存 CLI override 修正の影響範囲を調査する【TR-2】
- [x] 🖐 **操作**: `train.py`, `arguments/__init__.py`, `tests/test_n3dv_config.py` を読み、`--conf` 読み込み後に CLI 明示 override が勝つ実装とテストを整理する。
- [x] 🔎 **確認**: 現在の override 優先順位、変更候補、追加テスト名が作業記録に記載されている。
- [x] 🧪 **テスト**: 既存 `tests/test_n3dv_config.py` の対象ケースを特定し、追加すべき Hydra 併存ケースを列挙する。
- [x] 🛠 **エラー時対処**: 既存挙動が読み取れない場合は `rg "conf|override|argparse|ArgumentParser"` で呼び出し箇所を再探索する。

### 手順 4: `/workspace/Project/DEIM_sandbox` を明示確認する【TR-4】
- [x] 🖐 **操作**: `test -e /workspace/Project/DEIM_sandbox && find /workspace/Project/DEIM_sandbox -maxdepth 3 -type f | head -50` を実行し、Hydra 参考実装の有無を確認する。
- [x] 🔎 **確認**: 参照先が存在しない/空/有効のいずれかを作業記録に明示し、暗黙 fallback していない。`/workspace/Project/tomato_*/*/tomato_optim` など近傍プロジェクトは DEIM_sandbox の代替ではなく、統括承認がある場合のみ別枠の参考実装として扱う。
- [x] 🧪 **テスト**: 参照先調査のため自動テスト不要。採用した Hydra 導入方針が記録されていることを確認する。
- [x] 🛠 **エラー時対処**: 権限エラー・空ディレクトリ・不存在の場合は、その事実を記録し、既存 DASH 構造に基づく最小導入案へ統括承認付きで進む。近傍の `tomato_optim` へ silent fallback しない。

### 手順 5: Hydra と型/監視/optimizer 依存を調査する【TR-3/TR-6/TR-7/TR-8】
- [x] 🖐 **操作**: `pyproject.toml` と `uv run python -c` または `uv run python -m pip show` で `hydra-core`, `omegaconf`, `jaxtyping`, `beartype`, `tensorboard`, `torch.utils.tensorboard`, `schedulefree`, `schedulerfree`, `muon`, `muon-optimizer` の有無を確認する。
- [x] 🔎 **確認**: 各依存について「利用可能」「追加必要」「不可/不明」の判断と根拠が作業記録にある。`schedulerfree=False` / `schedulefree=True` の名称差、`muon-optimizer` package / `muon` import、lock の `f98f1cacc0263b04290753e32be8d498c1efc806` を明示する。
- [x] 🧪 **テスト**: import 確認コマンドを実行し、成功/失敗を記録する。
- [x] 🛠 **エラー時対処**: package 名が不明な場合は公式情報または installed package のみを根拠にし、推測した fallback 実装を入れない。`schedulerfree` 名で失敗した場合は `schedulefree==1.4.1` 採用済み判断を記録し、別名を勝手に追加しない。

### 手順 6: 実装ファイル構成を決める【TR-5】
- [x] 🖐 **操作**: 設定/CLI adapter、Hydra loader、type utility、optimizer factory、TensorBoard helper、対応テストのファイル配置案を作業記録へ記載する。
- [x] 🔎 **確認**: `train.py` の責務が増えすぎず、既存挙動を保つ小さな接続点になっている。
- [x] 🧪 **テスト**: 実装前に追加するテストファイルとテストケース名を列挙する。
- [x] 🛠 **エラー時対処**: 既存構造に適切な置き場がない場合は、最小の新規 module を作り、命名理由を記録する。

### フェーズ 2: TDD と実装

### 手順 7: CLI override + Hydra 併存テストを先に追加する【TR-2/TR-3】
- [x] 🖐 **操作**: `tests/test_n3dv_config.py` または新規設定テストへ、`--conf arguments/n3dv.py` 後に CLI 明示 override が勝つケースと `--hydra_config`/`--hydra_overrides` 併存ケースを追加する。
- [x] 🔎 **確認**: 実装前に追加テストが期待どおり失敗し、失敗理由が Hydra 未実装または期待仕様未充足である。
- [x] 🧪 **テスト**: `uv run pytest tests/test_n3dv_config.py -q` を実行し、fail-to-pass の fail 側を記録する。
- [x] 🛠 **エラー時対処**: 既存テストまで壊れる場合は、追加テストを分離し、既存 6 passed が維持されるか先に確認する。

### 手順 8: TensorBoard helper テストを先に追加する【TR-8】
- [x] 🖐 **操作**: mock writer または tmp logdir を使い、設定値、optimizer 名、iteration、loss、point count が記録されることを検証するテストを追加する。
- [x] 🔎 **確認**: 実装前に helper 未存在で失敗し、期待する writer API 呼び出しが明文化されている。
- [x] 🧪 **テスト**: `uv run pytest tests -k "tensorboard or summary" -q` または対象ファイルを実行し、期待失敗を記録する。
- [x] 🛠 **エラー時対処**: TensorBoard 依存がない場合は依存追加要否を手順5の判断に戻し、silent skip しない。

### 手順 9: optimizer factory テストを先に追加する【TR-7】
- [x] 🖐 **操作**: 既存 optimizer、schedulefree、MUON 指定時の factory 生成テストを追加する。依存不可ケースは明示的な例外を期待する。
- [x] 🔎 **確認**: unsupported optimizer が silent fallback せず、例外メッセージに必要依存または対応状況が含まれる。
- [x] 🧪 **テスト**: `uv run pytest tests -k "optimizer" -q` を実行し、実装前の失敗を記録する。
- [x] 🛠 **エラー時対処**: optimizer 対象 module が import 時に GPU 初期化する場合は、factory の純粋ロジックを分離して CPU-only テストにする。

### 手順 10: `jaxtyping` / `beartype` 境界テストを先に追加する【TR-6】
- [x] 🖐 **操作**: 設定 dataclass/helper/factory の軽量境界に対して、正常入力と不正入力のテストを追加する。
- [x] 🔎 **確認**: 不正入力が `beartype` または明示 validation で落ち、GPU hot path に runtime overhead を追加していない。
- [x] 🧪 **テスト**: `uv run pytest tests -k "type or config" -q` を実行し、実装前の失敗を記録する。
- [x] 🛠 **エラー時対処**: `jaxtyping` が tensor 型に過剰制約を要求する場合は、設定値や factory 境界に限定して適用範囲を縮小する。

### 手順 11: Hydra loader/CLI adapter を実装する【TR-2/TR-3/TR-5】
- [x] 🖐 **操作**: 既存 `--conf` を壊さず、追加 CLI `--hydra_config` / `--hydra_overrides` 等を読み込む loader/adapter を実装し、config 後に CLI 明示 override が勝つ順序にする。
- [x] 🔎 **確認**: 既存 `--conf arguments/n3dv.py` の挙動が維持され、Hydra 指定時だけ追加設定がマージされる。
- [x] 🧪 **テスト**: `uv run pytest tests/test_n3dv_config.py -q` を実行し、手順7の失敗が成功へ変わることを確認する。
- [x] 🛠 **エラー時対処**: Hydra が既存 argparse と競合する場合は、Hydra を compose API/明示 loader として扱い、main entrypoint 全体を Hydra decorator 化しない。

### 手順 12: 型 utility と dataclass 境界を実装する【TR-6/TR-5】
- [x] 🖐 **操作**: `jaxtyping` / `beartype` を設定 dataclass/helper/factory の境界へ導入し、型 alias と validation を共通化する。
- [x] 🔎 **確認**: 型注釈が重複せず、GPU 学習ループの hot path に不要な `beartype` decorator が増えていない。
- [x] 🧪 **テスト**: 手順10のテストと `uv run python -m py_compile` 対象ファイルを実行する。
- [x] 🛠 **エラー時対処**: 型チェックが動的 config と衝突する場合は、入力正規化 helper を追加し、境界後は単純な型へ変換する。

### 手順 13: optimizer factory を実装する【TR-7/TR-5】
- [x] 🖐 **操作**: optimizer 名と設定から既存 optimizer / schedulefree / MUON を生成する factory を実装する。未導入依存は明示エラーにする。
- [x] 🔎 **確認**: optimizer 選択が設定から追跡可能で、既存既定値は変わっていない。MUON は upstream README 方針に従い hidden weights を Muon、その他 parameter を AdamW 系 aux に分ける設計判断が記録されている。
- [x] 🧪 **テスト**: 手順9の optimizer テストを実行し、fail-to-pass を確認する。
- [x] 🛠 **エラー時対処**: MUON または schedulefree の API が想定と異なる場合は実装を止め、依存名/API/代替判断を作業記録へ記録する。MUON upstream commit や license に不明点があれば `https://github.com/KellerJordan/Muon` の README/LICENSE を再確認する。

### 手順 14: TensorBoard helper を実装する【TR-8/TR-5】
- [x] 🖐 **操作**: 既存 `SummaryWriter` を活かし、設定値、optimizer 名、iteration、loss、point count などを記録する helper を実装して `train.py` から呼び出す。
- [x] 🔎 **確認**: 監視項目が一箇所に集約され、学習本体に writer 呼び出しが散らばらない。
- [x] 🧪 **テスト**: 手順8の TensorBoard helper テストを実行し、fail-to-pass を確認する。
- [x] 🛠 **エラー時対処**: writer が無効な実行モードでは明示的に no-op helper を使い、silent fallback ではなく設定値として記録する。

### 手順 15: train 接続を最小変更で行う【TR-2/TR-3/TR-5/TR-7/TR-8】
- [x] 🖐 **操作**: `train.py` から CLI/Hydra adapter、optimizer factory、TensorBoard helper を利用する接続だけを行い、既存学習ロジックの大改造を避ける。
- [x] 🔎 **確認**: 既存引数での実行パスが維持され、追加機能は設定で有効化される。
- [x] 🧪 **テスト**: `uv run pytest tests/test_n3dv_config.py tests/test_report_guard.py -q` と追加テストを実行する。
- [x] 🛠 **エラー時対処**: 学習ループで副作用が広がる場合は adapter 返却値を既存引数形式に揃え、接続点を一箇所に戻す。

### フェーズ 3: 検証・DASH smoke

### 手順 16: 対象テストを実行する【TR-10】
- [x] 🖐 **操作**: `uv run pytest tests/test_n3dv_config.py tests/test_report_guard.py -q` と追加したテストファイルを実行する。
- [x] 🔎 **確認**: 既知の 6 passed を含め、追加テストがすべて成功している。
- [x] 🧪 **テスト**: 実行結果の passed 件数と失敗なしを作業記録に残す。
- [x] 🛠 **エラー時対処**: 既存テストの失敗は今回差分との関係を切り分け、追加テストだけの失敗と混同しない。

### 手順 17: 全 pytest を実行する【TR-10】
- [x] 🖐 **操作**: `uv run pytest tests` を実行する。
- [x] 🔎 **確認**: 全 tests が成功、または既知の環境依存 skip/xfail が明示されている。
- [x] 🧪 **テスト**: `uv run pytest tests` の結果を作業記録に要約する。
- [x] 🛠 **エラー時対処**: 長時間化または GPU 依存失敗は対象テスト、失敗理由、再実行条件を記録し、暗黙に無視しない。

### 手順 18: ty の実コマンドを確認して型チェックを実行する【TR-10】
- [x] 🖐 **操作**: `uv run ty --help` で `ty==0.0.46` の実コマンド形を確認し、可能なら `uv run ty check .` を実行する。
- [x] 🔎 **確認**: 実行した ty コマンドと結果が作業記録に残っている。
- [x] 🧪 **テスト**: 変更ファイルセットの型チェックが成功する。full repo ty が未達の場合は既存診断として出力を記録する。
- [x] 🛠 **エラー時対処**: `ty check .` が存在しない場合は help 出力から正しいコマンドを採用し、代替理由を明示する。

### 手順 19: py_compile を実行する【TR-10】
- [x] 🖐 **操作**: 変更した Python ファイルを対象に `uv run python -m py_compile <files...>` を実行する。
- [x] 🔎 **確認**: 構文エラーがない。
- [x] 🧪 **テスト**: py_compile の exit 0 を作業記録に残す。
- [x] 🛠 **エラー時対処**: import 副作用ではなく構文エラーだけを見る。構文エラー時は対象ファイルと行番号を記録して修正する。

### 手順 20: DASH smoke を実行する【TR-8/TR-10】
- [x] 🖐 **操作**: `/home/kasm-user/Desktop/DASH/data/tva_nyx650_400_aliked_lg_glomap` を dataset root として、短時間・低負荷の train smoke を実行する。実際のコマンドは既存 CLI help/README で確認してから記録する。
- [x] 🔎 **確認**: train が初期化し、設定、optimizer 名、iteration/loss/point count の TensorBoard 記録が生成される。
- [x] 🧪 **テスト**: smoke 実行ログと TensorBoard logdir の生成を確認する。
- [x] 🛠 **エラー時対処**: dataset がない/重い/GPU OOM の場合は原因を記録し、最小 iteration/低解像/CPU 不可などの条件を明示して統括判断を仰ぐ。

### フェーズ 4: 差分整理・commit

### 手順 21: data 成果物除外と差分を確認する【TR-11】
- [x] 🖐 **操作**: 変更差分を確認し、`data/` 以下の大きい成果物や symlink が commit 対象に入っていないことを確認する。必要なら `.gitignore` 追記を検討する。
- [x] 🔎 **確認**: commit 対象がソース、テスト、設定、lock、作業書など必要最小限に限られている。
- [x] 🧪 **テスト**: 差分確認結果を作業記録に残す。
- [x] 🛠 **エラー時対処**: data 成果物が含まれる場合は stage から外し、ignore が必要な場合は最小パターンを追加して理由を記録する。

### 手順 22: commit を作成する【TR-11】
- [x] 🖐 **操作**: 検証済み差分のみを stage し、要約が明確な commit を作成する。
- [x] 🔎 **確認**: commit hash が取得でき、commit に data 成果物が含まれていない。一時 hash は `326b576`。本 workdoc 追記を同一 commit に `git commit --amend --no-edit` するため、最終 hash は統括最終報告を正とする。
- [x] 🧪 **テスト**: commit 直前の pytest/ty/py_compile/smoke 結果が作業記録に紐づいている。`git diff --cached --check` は commit 前に OK。
- [x] 🛠 **エラー時対処**: commit 失敗時は hook/テスト/差分混入を切り分け、失敗内容を記録してから再実行する。push はユーザー指示なしでは行わない。

---

## 4. 作業に使用するコマンド参考情報

### 基本的な開発ワークフロー

```bash
cd /home/kasm-user/Desktop/DASH
uv sync
uv run pytest tests/test_n3dv_config.py tests/test_report_guard.py -q
uv run pytest tests
uv run ty --help
uv run ty check .
uv run python -m py_compile train.py arguments/__init__.py
```

### 依存確認・追加候補

```bash
cd /home/kasm-user/Desktop/DASH
uv run python - <<'PY'
import importlib.util
for name in ["hydra", "omegaconf", "jaxtyping", "beartype", "tensorboard", "schedulefree", "schedulerfree", "muon"]:
    print(name, bool(importlib.util.find_spec(name)))
PY

# 必要性が確認できた場合のみ、理由を記録してから追加する
uv add hydra-core omegaconf jaxtyping beartype tensorboard
# schedulefree は実パッケージ名。schedulerfree は採用しない。
# MUON は実行済み: uv add "muon-optimizer @ git+https://github.com/KellerJordan/Muon"
```

### optimizer 依存名の確定情報

- `schedulefree==1.4.1`: PyPI/package/import 名は `schedulefree`。`schedulerfree` は解決不可名として扱い、作業記録へ名称差を残す。
- `muon-optimizer`: Git 依存 `https://github.com/KellerJordan/Muon#f98f1cacc0263b04290753e32be8d498c1efc806`。Python import 名は `muon`。README 上、hidden weights に Muon、その他 parameter に AdamW 系 aux を使う。license は MIT。
- `rust==1.3.1`: `uv add --dev pytest ty rust` の反映として lock/pyproject を確認する。`import rust` での検証や Rust toolchain 判定には使わない。

### DEIM_sandbox 確認

```bash
test -e /workspace/Project/DEIM_sandbox && find /workspace/Project/DEIM_sandbox -maxdepth 3 -type f | head -50
# 存在しない場合はそのまま作業記録へ「不存在」と記録する。
# 近傍の tomato_optim は DEIM_sandbox ではないため、silent fallback しない。
```

### DASH smoke 参考

```bash
cd /home/kasm-user/Desktop/DASH
uv run python train.py --help
# 実際の smoke コマンドは help/既存 README/既存テストから確定し、作業記録に正確に残す。
# dataset root 候補:
# /home/kasm-user/Desktop/DASH/data/tva_nyx650_400_aliked_lg_glomap
```

### commit 前確認

```bash
cd /home/kasm-user/Desktop/DASH
# data/ 以下の成果物や symlink を stage しないこと
git status --short
git diff --stat
git diff --cached --stat
```

---

## 6. 完了の定義

*作業が最後まで完了したら `[ ]` を `[x]` にしつつ、作業が本当に完了したかをチェックします*

- [x] 観点(a): 作業書レビューが完了し、Trace ID と各手順の対応が作業記録に残っている。【TR-1】Euler review: PASS_WITH_NOTES。
- [x] 観点(b): 既存 `--conf arguments/n3dv.py` と CLI 明示 override 優先の挙動が維持され、Hydra 追加 CLI と併存する。【TR-2, TR-3】`train.py` / `arguments/__init__.py` / `tests/test_n3dv_config.py` で確認。
- [x] 観点(c): `/workspace/Project/DEIM_sandbox` の参照可否が記録され、暗黙 fallback なしで設計判断している。【TR-4】DEIM_sandbox は不存在。近傍 `tomato_optim` は別参考であり fallback ではない。
- [x] 観点(d): 設定/CLI/optimizer factory/TensorBoard helper/type utility が責務分離され、`train.py` の大改造を避けている。【TR-5】
- [x] 観点(e): `jaxtyping` / `beartype` が軽量境界に導入され、テストと py_compile が通る。【TR-6】
- [x] 観点(f): MUON + schedulefree が利用可能なら選択式で動作し、不可なら明示的なブロック/代替判断が記録されている。`schedulefree==1.4.1` 採用、`schedulerfree` 不採用理由、`muon-optimizer` lock commit hash `f98f1cacc0263b04290753e32be8d498c1efc806`、MIT license、MUON の parameter group 方針が作業記録にある。【TR-7】
- [x] 観点(g): TensorBoard で設定値/optimizer名/iteration/loss/point count 等を監視でき、テスト可能である。【TR-8】smoke で event file と tag を確認済み。
- [x] 観点(h): `uv add --dev pytest ty rust` 済み状態と追加依存判断が `pyproject.toml` / `uv.lock` / 作業記録に残る。`rust==1.3.1` は Python import 名 `rust` では検証しない旨も残る。【TR-9】
- [ ] 観点(i): `uv run pytest tests`, ty の実チェック、`uv run python -m py_compile ...`, DASH smoke が実行され、結果が作業記録に残っている。【TR-10】部分達成: `pytest tests`, 変更ファイルセットの ty, temp 等除外 py_compile, DASH smoke は成功。full repo `uv run ty check . --respect-ignore-files --output-format concise` は 95 diagnostics のため未達として扱う(主因は既存レガシー/外部 CUDA extension stubs/scene readers/render)。
- [x] 観点(j): `git status --short`, `git diff --stat`, `git diff --cached --stat` 等で data/ 以下の成果物や symlink を commit しないことを確認し、必要な差分だけ commit され、commit hash が作業記録にある。【TR-11】一時 hash は `326b576`。workdoc 追記を同一 commit に amend するため、最終 hash は統括最終報告を正とする。`data/` と `/tmp` smoke output は commit 対象外、`.gitignore` に `data/` 追加済み。`git diff --cached --check` は commit 前に OK。

---

## 7. 作業記録

**重要な注意事項：**

*   作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
*   各作業項目を開始する際と完了する際の両方で記録を行うこと。
*   作業内容は具体的なコマンドや操作手順を詳細に記載すること。
*   結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
*   複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
*   コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
*   エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-10 | 10:01:43 UTC | 書紀エージェント Hypatia(Codex) | 本作業書を新規作成 | Coordinator 確認時刻を記録。既知状態: CLI override 修正あり、対象2テストは 6 passed 済み、`uv add --dev pytest ty rust` 実行済み、DASH dataset root は成果物のため commit 除外方針 |
| 2026-06-10 | 10:08:40 UTC | 作業書レビュー/書紀補助エージェント Codex | `review-written-workdoc` 観点で本作業書のみをレビューし、安全な改善を直接編集 | Coordinator は直接 workdoc 編集しない前提のため、本エージェントがレビュー修正担当として記録。`schedulerfree` は PyPI 名として採用せず `schedulefree==1.4.1` / import `schedulefree` を採用する判断、`uv add --dev pytest ty rust` 反映と `rust==1.3.1` は `import rust` 検証対象ではない点、`/workspace/Project/DEIM_sandbox` 不存在時に近傍 `tomato_optim` へ silent fallback しない点、MUON upstream `KellerJordan/Muon` / `muon-optimizer` / import `muon` / lock commit `f98f1cacc0263b04290753e32be8d498c1efc806` / MIT license / hidden weights + AdamW aux 方針、DoD の pytest/ty/py_compile/smoke/diff/data 除外/commit hash 条件を追記 |
| 2026-06-10 | 10:30:54 UTC | 書紀エージェント Hypatia(Codex) | 実作業結果を §3/§6/§7 へ反映 | Euler が `review-written-workdoc` 観点で対象 workdoc のみを編集し PASS_WITH_NOTES。`uv add --dev pytest ty rust` 済み: `pytest==9.0.3`, `ty==0.0.46`, `rust==1.3.1`。`uv run python -c "import rust"` は `ModuleNotFoundError`; Python import 名を提供しない package と記録し、dummy/fallback ではない |
| 2026-06-10 | 10:30:54 UTC | 書紀エージェント Hypatia(Codex) | 依存追加・参照先調査結果を記録 | `uv add hydra-core omegaconf jaxtyping beartype schedulerfree` は `schedulerfree` が見つからず失敗。実 package 名 `schedulefree==1.4.1` を採用し `uv add hydra-core omegaconf jaxtyping beartype schedulefree` 成功。ユーザー明示の MUON upstream `https://github.com/KellerJordan/muon` に対し `uv add "muon-optimizer @ git+https://github.com/KellerJordan/Muon"` 成功、lock commit `f98f1cacc0263b04290753e32be8d498c1efc806`。`/workspace/Project/DEIM_sandbox` は不存在。近傍 `tomato_optim` は DEIM_sandbox fallback ではなく別参考として扱った |
| 2026-06-10 | 10:30:54 UTC | 書紀エージェント Hypatia(Codex) | 実装内容を記録 | `arguments/__init__.py`: Hydra YAML loader/CLI override helpers/optimizer CLI defaults/`beartype` 境界。`configs/train/tva400_muon_schedulefree.yaml` 追加。`utils/optimizer_utils.py`: `SingleDeviceMuonWithAuxAdam`、optimizer factory、schedulefree reference wrapper、optimizer state resize helpers、jaxtyping/beartype 境界。`utils/tensorboard_utils.py`: metadata writer。`scene/gaussian_model.py` / `scene/deform_model.py`: optimizer factory 使用、Gaussian densify/prune state mutation を Adam 固有 state 名依存から汎用 helper へ移行。`train.py`: Hydra config と CLI override 順序、schedulefree train/eval mode、TensorBoard metadata 接続。`tests/test_optimizer_utils.py`, `tests/test_tensorboard_utils.py`, `tests/test_n3dv_config.py` 追加/更新 |
| 2026-06-10 | 10:30:54 UTC | 書紀エージェント Hypatia(Codex) | 重要修正経緯を記録 | `schedulefree.ScheduleFreeWrapper` は DASH の stride 付き parameter で `view(torch.uint8)` RuntimeError を起こしたため、同 package の `ScheduleFreeWrapperReference` に切替。これは silent fallback ではなく同じ schedulefree 実装の安全 wrapper 採用 |
| 2026-06-10 | 10:30:54 UTC | 書紀エージェント Hypatia(Codex) | 検証結果を記録 | `uv run pytest tests -q` -> 19 passed, 3 warnings(CUDA arch / `torch.cuda.amp` deprecation)。変更ファイルセット ty: `uv run ty check arguments/__init__.py train.py scene/gaussian_model.py scene/deform_model.py utils/optimizer_utils.py utils/tensorboard_utils.py tests/test_n3dv_config.py tests/test_optimizer_utils.py tests/test_tensorboard_utils.py --output-format concise` -> All checks passed。temp 等除外 py_compile OK。DASH smoke command は exit 0 |
| 2026-06-10 | 10:30:54 UTC | 書紀エージェント Hypatia(Codex) | DASH smoke と TensorBoard 確認を記録 | smoke: `uv run python train.py -s /home/kasm-user/Desktop/DASH/data/tva_nyx650_400_aliked_lg_glomap --model_path /tmp/dash_tva400_muon_schedulefree_smoke --conf arguments/n3dv.py --hydra_config configs/train/tva400_muon_schedulefree.yaml --iterations 2 --test_iterations 2 --save_iterations 2 --resolution 4 --quiet` -> exit 0。出力: `/tmp/dash_tva400_muon_schedulefree_smoke/point_cloud/iteration_2/point_cloud.ply`, `deform/iteration_2/deform.pth`, `events.out.tfevents...`。TensorBoard tags: scalars `config/iterations`, `train_loss_patches/*`, `test/*`, `train/*`, `total_points`; tensors `config/args/text_summary`, `config/optimizers/text_summary` |
| 2026-06-10 | 10:30:54 UTC | 書紀エージェント Hypatia(Codex) | 未達/注意点を記録 | full repo `uv run ty check . --respect-ignore-files --output-format concise` -> 95 diagnostics。主因は既存レガシー/外部 CUDA extension stubs/scene readers/render などで、変更ファイルセットは All checks passed。最初の py_compile は `temp/colmap_refs/colmap-odometry/src/odometry/matcher.py` の外部参照コード syntax error を拾って失敗したため、DoD 対象外の `temp` を除外して本体 py_compile OK。`data/` は untracked 成果物で commit 対象外予定、`/tmp` smoke output も commit 対象外。commit は未実施/未記録 |
| 2026-06-10 | 10:36:58 UTC | 書紀エージェント Hypatia(Codex) | commit 実施結果を記録 | `git commit -m "Add Hydra config and Muon schedule-free optimizers"` 実施済み。一時 hash は `326b576`。本 workdoc 追記を同一 commit に含めるため、これから `git commit --amend --no-edit` 予定。最終 hash は amend 後に統括最終報告を正とする。`data/` と `/tmp` smoke output は commit 対象外、`.gitignore` に `data/` 追加済み。`git diff --cached --check` は commit 前に OK |
| | | | | |
