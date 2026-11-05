# 入玉スカラーつき Simple の GPU 検証手順 (手元の WSL2)

[NNUE 新アーキテクチャの仕様書](nnue-architecture-ideas.md) のアイデア A2 (HalfKP＋入玉スカラー、
L1 入力 608) の学習器 (段階4) は、GPU の無い環境で実装し、CPU で確かめられる部分だけを
確かめてある。この文書は、残りの GPU でしか確かめられない項目 (仕様書 §7.1 の「GPU 検証待ちの
項目」) を手元の WSL2 で片付けるための手順書である。上から順に実行する。

各手順には「期待される結果」と「失敗したときに Claude Code に渡す情報」を書いた。失敗したら
その手順で止め、渡す情報をまとめて Claude Code のセッションに貼る。共通して渡すもの:

- `git -C /path/to/tatara rev-parse HEAD` と `git -C /path/to/tatara status --short` の出力
- 失敗したコマンドそのものと、その出力全体 (`2>&1 | tee <name>.log` で残したもの)
- `nvidia-smi` の出力

## 0. 前提

### GPU

| GPU | compute capability | VRAM | 使えるか |
|---|---|---|---|
| Quadro T1000 (主) | 7.5 (Turing) | 4 GB | 使える。段階4のテストと smoke は足りる。`local-ci.sh` の既存テストの一部は VRAM が足りない可能性がある (§6) |
| RTX 2070 | 7.5 (Turing) | 8 GB | 使える |
| RTX 3050 | 8.6 (Ampere) | 8 GB (デスクトップ) | 使える |
| RTX 4090 | 8.9 (Ada) | 24 GB | 使える。`local-ci.sh` を全部通すのに最も確実 |

- 既定の CUDA C++ (native) backend は、`nvidia-smi` が報告する compute capability の PTX を
  nvcc で作る (`crates/cuda-native-runtime/build.rs`)。kernel は倍精度の `atomicAdd` (sm_60 以上)・
  `__shfl_*_sync`・FP16 の変換だけを使い、sm_80 以上の命令 (`cp.async` など) は使わないので、
  sm_75 で動く。段階4で変えた kernel は、クラウドで clang により sm_75 の PTX に compile できる
  ことまで確かめた (実行はしていない)。
- 比較用の cuda-oxide backend は既定では sm_80 の PTX を作るので、Turing では
  `CUDA_OXIDE_TARGET=sm_75` が要る。`scripts/build-kernels.sh` と `scripts/local-ci.sh` は
  `nvidia-smi` を見て自動で設定する。手で `cargo-oxide build` するときだけ自分で付ける。
- GPU を 2 枚以上挿しているときは、native backend は最も低い compute capability に合わせて
  PTX を作り、テストは device 0 を使う。検証する GPU を `CUDA_VISIBLE_DEVICES=<番号>` で 1 枚に
  絞る。GPU を差し替えたら `cargo clean -p cuda-native-runtime` してからビルドし直す。

### ソフトウェア

[docs/setup.ja.md](setup.ja.md) の「Windows (WSL2) の準備」と「システム install」に従う。要点:

| もの | 版 | 備考 |
|---|---|---|
| NVIDIA driver | Windows host 側に入れる (WSL 内には入れない) | WSL 用の CUDA に対応した版 |
| CUDA Toolkit | 12.x (12.9 で確認済み) | WSL 用の `cuda-toolkit-12-x` だけを入れる。nvcc・libNVVM・nvJitLink・libcublas を含む。`nvcc --list-gpu-arch` に `compute_75` があること |
| LLVM | 21 以上 (22 推奨) | `llc` / `opt` / `llvm-link` (NVPTX target つき)。cuda-oxide の PTX 生成に使う |
| clang | 21 か 22 と `libclang-common-{21,22}-dev` | cuda-oxide の bindgen に使う |
| Rust | nightly-2026-04-03 | `rust-toolchain.toml` で固定。リポジトリ内で初めて `cargo` を打つと入る |
| cargo-oxide | Cargo.lock の pin と同じ rev | `bash scripts/setup-cuda-oxide.sh` で入れる |

### VRAM の見積り (HalfKP 256x2-32-32＋入玉スカラー、FP32 既定)

- FT の weight は 125,388 × 256 = 3,210 万個 (f32 で 122 MB)。weight・lookahead の slow weight・
  勾配・Ranger の 1 次と 2 次のモーメントの 5 本で約 612 MB。`--fp16-opt-state` で 2 本を f16 に
  すると約 490 MB。dense 層 (L1 は 608 × 32) は無視できる大きさ。
- batch に比例する作業領域は 1 局面あたり約 11 KB (`--ft-fp16-out` で +2 KB)。うち入玉スカラーの
  分は u8 の入力 2 本 (active / back) の 192 B と、`combined` / `dcombined` の 96 列 × 4 B × 2。
- CUDA context と cuBLAS で 0.3〜0.6 GB、T1000 を画面表示にも使うならその分も要る。

| batch | 作業領域 | 合計の目安 |
|---:|---:|---:|
| 16,384 (既定) | 約 170 MB | 約 1.2〜1.5 GB |
| 32,768 | 約 340 MB | 約 1.4〜1.7 GB |
| 65,536 | 約 690 MB | 約 1.8〜2.1 GB |

4 GB の T1000 では batch 65,536 までが目安。OOM のときはエラーメッセージに出る見積りに従って
batch を下げるか `--fp16-opt-state` を付ける。

## 1. 環境の確認

```bash
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv
```

```bash
nvcc --version
```

```bash
nvcc --list-gpu-arch | grep compute_75
```

```bash
ls "${CUDA_HOME:-/usr/local/cuda}"/lib64/libcublas.so
```

```bash
(llc-22 --version || llc-21 --version) | grep -i nvptx
```

```bash
(clang-22 --version || clang-21 --version) | head -1
```

```bash
cd /path/to/tatara && rustc -V && cargo-oxide --version
```

**期待される結果**: T1000 の行に `7.5` と 4 GB 前後が出る。nvcc は 12.x。`compute_75` が出る。
libcublas.so が見つかる。llc の target 一覧に `nvptx64` がある。rustc は
`nightly-2026-04-03` 系。cargo-oxide の版が出る。

**失敗したときに渡す情報**: 上の全コマンドの出力、`uname -a`、`cat /etc/os-release`、
Windows 側の driver の版。

## 2. GPU 限定コードの型検査 (§7.1 の 1 項目め)

```bash
cd /path/to/tatara && git checkout claude/nyugyoku-scalar-trainer
```

```bash
cargo check -p nnue-trainer --all-targets 2>&1 | tee check-native.log
```

```bash
cargo check -p nnue-trainer --all-targets --no-default-features --features oxide-parity 2>&1 | tee check-parity.log
```

```bash
cargo check -p nnue-trainer --all-targets --no-default-features --features oxide 2>&1 | tee check-oxide.log
```

**期待される結果**: 3 つとも `Finished` で終わり、warning と error が無い。クラウドでは nvcc を
空のファイルを書く代役に置き換えて Rust 側の型検査だけを通してある (3 構成とも通過)。手元では
nvcc が本物の CUDA C++ を compile するので、ここで初めて CUDA C++ の compile error が出る可能性が
ある (クラウドでは clang で sm_75 の PTX に compile できることまで確認済み)。

**失敗したときに渡す情報**: 失敗したコマンドの log 全体。nvcc の error なら `nvcc --version` も。

## 3. kernel のビルド

```bash
bash scripts/build-kernels.sh 2>&1 | tee build-kernels.log
```

```bash
grep -E '(cp\.async|wgmma|tcgen05|tma\.|cluster\.)' bins/nnue_train/nnue_train.ll
```

```bash
cargo build -p nnue-trainer --release 2>&1 | tee build-release.log
```

**期待される結果**: build-kernels.sh が `compute_cap 7.5 (sub-Ampere) → CUDA_OXIDE_TARGET=sm_75 を設定`
と `完了。` を出す (Ampere 以降の GPU なら `Ampere+` の行)。grep は何も出さない。release
ビルドが通る。

**失敗したときに渡す情報**: build-kernels.log と build-release.log の全体、`echo $CUDA_OXIDE_TARGET`。

## 4. GPU/CPU 同等性テスト (§7.1 の 3 項目め)

4 GB の GPU では、並列に走るテストの VRAM の予算を下げる (既定は 12 GB の GPU 向けの 8 GiB)。

```bash
export TATARA_TEST_GPU_BUDGET_GIB=2
```

### 4a. kernel 単位 (cuda-oxide の kernel と CPU reference)

```bash
cargo test -p nnue-trainer --release --no-default-features --features oxide -- --test-threads=1 scalar 2>&1 | tee test-kernels.log
```

**期待される結果**: 次の 3 つを含め、名前が一致した全テストが `ok` (CLI の CPU テストも一緒に走る)。

| テスト | 見ること |
|---|---|
| `simple_ft_post_fused_with_scalar_columns_matches_cpu` | FT post の融合 kernel (CReLU / SCReLU) が行幅 608 で CPU reference と一致し、行末の 96 列を書き換えない |
| `simple_bwd_ft_act_fused_with_scalar_columns_matches_cpu` | FT の活性化 backward の融合 kernel が行幅 608 で一致し、`dcombined` の入玉スカラーの列 (NaN を置く) を読まない |
| `simple_nyugyoku_scalar_to_combined_matches_cpu` | 入玉スカラーを u8 / 127 で `combined` に書く kernel が CPU reference と bit 単位で一致する |

### 4b. trainer 単位 (既定の native backend)

```bash
cargo test -p nnue-trainer --release -- --test-threads=1 simple_nyugyoku 2>&1 | tee test-trainer.log
```

**期待される結果**: 次の 3 つを含め、名前が一致した全テストが `ok` (CLI の CPU テストも一緒に走る)。

| テスト | 見ること |
|---|---|
| `simple_nyugyoku_forward_matches_cpu_reference` | 入玉スカラーつきの forward (FP32 と `--ft-fp16-out`) が、同じ weight から f32 で計算した値と一致する。入玉スカラーを 0 にすると出力が変わる |
| `simple_nyugyoku_warm_start_keeps_outputs` | 入玉スカラーなしのネットを weight 0 の列で広げて読み込むと、元のネットと同じ出力・loss になる (ワームスタート直後の関数保存) |
| `simple_nyugyoku_training_moves_scalar_columns` | FT の列の L1 初期値が入玉スカラーなしと bit 単位で同じ。40 step で loss が下がり、0 で始めた入玉スカラーの列の weight が動く |

### 4c. CUDA C++ と cuda-oxide の一致

```bash
cargo test -p nnue-trainer --release --no-default-features --features oxide-parity -- --test-threads=1 simple_nyugyoku 2>&1 | tee test-parity.log
```

**期待される結果**: 4b の 3 つに加えて `simple_nyugyoku_native_matches_cuda_oxide` が `ok`
(入玉スカラーつきの 3 step 後の L1 weight が両 backend で一致する)。

**4a〜4c が失敗したときに渡す情報**: 該当 log 全体 (assertion のメッセージに局面番号と
GPU / CPU の値が出る)。`RUST_BACKTRACE=1` を付けて再実行した出力。

## 5. smoke (学習データなしで GPU 経路を通す)

```bash
target/release/nnue-train simple 2>&1 | tee smoke.log
```

**期待される結果**: 次の行が出て、最後に `[smoke/simple] PASSED` が出る。

```text
[smoke/simple] entering-king scalars (L1 input 608): loss A -> B, scalar-column L1 weight norm 0.000e0 -> C
[smoke/simple] entering-king scalars OK ✓ (quantised round-trip loss D)
```

`B < A`、`C > 0`、`D` は有限。

**失敗したときに渡す情報**: smoke.log 全体。

## 6. `scripts/local-ci.sh` (§7.1 の 2 項目め、tatara の push 前規約)

```bash
TATARA_TEST_GPU_BUDGET_GIB=2 bash scripts/local-ci.sh 2>&1 | tee local-ci.log
```

**期待される結果**: 最後の行が `PASS`。

T1000 (4 GB) では、段階4と無関係な既存の大きなテスト (ft_out 1536 の LayerStack など) が
`CUDA_ERROR_OUT_OF_MEMORY` で落ちる可能性がある。その場合はテスト名を控え、同じ commit で
RTX 4090 (または 8 GB の 2070 / 3050) に切り替えて `local-ci.sh` を通す。段階4の追加テストは
4 で T1000 で通っていればよい。

**失敗したときに渡す情報**: local-ci.log 全体 (どの step で止まったか)。OOM なら、落ちた
テスト名と `nvidia-smi` の memory.total / 実行中の他 process。

## 7. ワームスタート直後の出力一致 (CLI の経路)

4b の `simple_nyugyoku_warm_start_keeps_outputs` は trainer の中で確かめる。ここでは
`--init-from` の CLI の経路 (入玉スカラーなしの `.bin` を広げて学習を始め、書き出す) を通し、
書き出したネットが量子化 CPU forward で元のネットと同じ値を返すことを見る。

1. 入玉スカラーなしの HalfKP 256x2-32-32 のネットを用意する。学習済みの tatara の `.bin` が
   あればそれを使う。無ければ乱数のネットを作る:

   ```bash
   cargo run -p nnue-format --release --example simple_quantised_eval -- random-net --out /path/to/work/halfkp.bin
   ```

2. 評価に使う局面を用意する。やねうら王 (入玉スカラーつきの edition) の `test nn gen_sfens`
   で 10,000 局面を作る:

   ```text
   isready
   test nn gen_sfens 10000 1 /path/to/work/pos.sfen 40
   ```

   やねうら王が手元に無ければ、golden の局面 (32 局面) を使う:

   ```bash
   grep -v '^#' crates/shogi-features/tests/data/nyugyoku_scalar_golden.tsv | cut -f4 | sort -u > /path/to/work/pos.sfen
   ```

3. 学習率を極小にして 1 superbatch だけ学習し、書き出す (`--lr` は 0 を受け付けないので
   `1e-12` にする。f32 の weight は丸めで変わらない)。教師データは何でもよい:

   ```bash
   target/release/nnue-train \
     --data /path/to/train.psv \
     --output /path/to/work/ws --net-id ws \
     --feature-set halfkp \
     --superbatches 1 --batches-per-superbatch 3 --save-rate 1 --threads 4 \
     --lr 1e-12 \
     --init-from /path/to/work/halfkp.bin \
     --scale 290 --win-rate-model \
     --wrm-in-offset 0 --wrm-target-offset 0 \
     --wrm-in-scaling 290 --wrm-target-scaling 290 --wrm-nnue2score 290 \
     simple --nyugyoku-rule csa27 --nyugyoku-init-noise 0 --fv-scale 16 2>&1 | tee ws.log
   ```

4. 元のネットと書き出したネットを同じ局面で評価して比べる:

   ```bash
   cargo run -p nnue-format --release --example simple_quantised_eval -- eval --net /path/to/work/halfkp.bin --sfens /path/to/work/pos.sfen > /path/to/work/a.tsv
   ```

   ```bash
   cargo run -p nnue-format --release --example simple_quantised_eval -- eval --net /path/to/work/ws/ws-1.bin --sfens /path/to/work/pos.sfen --nyugyoku-rule csa27 > /path/to/work/b.tsv
   ```

   ```bash
   cmp /path/to/work/a.tsv /path/to/work/b.tsv && echo SAME
   ```

   ```bash
   cargo run -p nnue-format --release --example simple_quantised_eval -- l1-norms --net /path/to/work/ws/ws-1.bin --nyugyoku-rule csa27
   ```

**期待される結果**: ws.log に `[train] --init-from: the net has no entering-king scalar inputs;
widened L1 from 512 to 608 inputs` が出る。`cmp` が `SAME` を出す。`l1-norms` の
`scalar_stm` / `scalar_nstm` / `scalar_pad` が `0.000000` で、`ft_stm` / `ft_nstm` は
元のネットと同じ値 (`l1-norms --net /path/to/work/halfkp.bin` と比べる)。

**失敗したときに渡す情報**: ws.log、`cmp` が最初に違うと言った行番号と、a.tsv / b.tsv のその行、
pos.sfen のその行、両ネットの `l1-norms` の出力。

## 8. 数 superbatch の smoke 学習

実データで数 superbatch 学習し、loss が下がることと、入玉スカラーの列の weight が 0 から動く
ことを見る。比較のため、同じデータ・同じ設定で入玉スカラーなしの run も回す。

```bash
target/release/nnue-train \
  --data /path/to/train.psv \
  --output /path/to/work/sb --net-id sb \
  --feature-set halfkp \
  --superbatches 4 --batches-per-superbatch 500 --save-rate 1 --threads 8 \
  --scale 290 --win-rate-model \
  --wrm-in-offset 0 --wrm-target-offset 0 \
  --wrm-in-scaling 290 --wrm-target-scaling 290 --wrm-nnue2score 290 \
  simple --nyugyoku-rule csa27 --nyugyoku-init-noise 0 2>&1 | tee sb.log
```

```bash
for n in 1 2 3 4; do echo "== sb $n"; cargo run -q -p nnue-format --release --example simple_quantised_eval -- l1-norms --net /path/to/work/sb/sb-$n.bin --nyugyoku-rule csa27; done
```

比較用 (入玉スカラーなし): 上の学習のコマンドから `--nyugyoku-rule csa27 --nyugyoku-init-noise 0`
を外し、`--output /path/to/work/base --net-id base` にして同じように回す。

**期待される結果**:

- sb.log の `[train] superbatch k/4 | loss ...` の loss が superbatch ごとに下がる (少なくとも
  1 → 4 で下がる)。比較用の run と同程度か、少し低い。
- `l1-norms` の `scalar_stm` / `scalar_nstm` が sb 1 で 0 より大きく、sb ごとに変化する。
  `scalar_pad` (入力が常に 0 の 8 列) は 0 のまま。`ft_stm` / `ft_nstm` は有限。
- sb.log に `non-finite` が出ない。

**失敗したときに渡す情報**: sb.log と比較用の log の全体、4 つの `l1-norms` の出力、
学習中の `nvidia-smi` (VRAM)、教師データの局面数と入手元。

## 9. §7.1 のチェックリストとの対応

| §7.1 の項目 | 手順 |
|---|---|
| `cargo check -p nnue-trainer` で GPU 限定コードを型検査する (段階3・段階5の変更を含む) | 2 |
| `bash scripts/local-ci.sh` が全 crate で PASS | 6 (T1000 で OOM したら RTX 4090 で) |
| 段階4の GPU/CPU 同等性 | 3, 4a, 4b, 4c |
| 段階4の smoke | 5, 8 |
| 段階4のワームスタートの関数保存 | 4b (`simple_nyugyoku_warm_start_keeps_outputs`), 7 |

全部通ったら、tatara の段階4の draft PR に結果 (各手順の期待どおりだったか、8 の loss と
`l1-norms` の値) を書き、§7.1 のチェックを付けてから本マージに進む。
