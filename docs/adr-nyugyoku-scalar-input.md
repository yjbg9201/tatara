# ADR: 入玉スカラー特徴を Simple アーキの L1 入力へ連結する

[docs/nnue-architecture-ideas.md](nnue-architecture-ideas.md) のアイデア A2
(HalfKP＋入玉スカラー特徴) について、実装方針の「段階1 (調査とADR)」の結論を記録する。
対象は tatara (学習器) と `yjbg9201/YaneuraOu` (推論エンジン)。GPU は使えない前提で調査した
(仕様書 §7.1)。

## 1. 決定事項 (要約)

| # | 決定 |
|---|---|
| D1 | 入玉スカラーは **Simple アーキの拡張オプション**として実装する (新サブコマンドにしない)。`SimpleId` に「L1 に連結する密入力」の種別を足し、既定値 (なし) では既存挙動と bit-identical |
| D2 | 学習器では、既存の `combined` バッファ (FT 出力の連結、L1 の入力) の末尾 96 列にスカラーを書く。L1 の forward/backward は既存の cuBLAS Sgemm がそのまま扱う |
| D3 | 行ストライドを `2*ft_out` に固定している FP32 融合カーネル 4 本に、ストライド引数を足す (CUDA C++ と cuda-oxide の両方) |
| D4 | 初版の活性化は **CReLU のみ**。SCReLU / Pairwise とスカラーの併用は CLI で拒否する |
| D5 | スカラーはホスト側で `round(x*127)` の u8 値に量子化してから、学習では `u8/127` の f32 として渡す。学習とエンジンで同じ値を見る |
| D6 | ルール表・区切り表は Rust と C++ に別々に実装し、**共有 golden ファイル**で一致を保証する。tatara 側の golden ファイルを正とする |
| D7 | ネットのハッシュ・arch 文字列に「スカラー種別・ルール・符号化」を含める。YaneuraOu 側は新アーキのビルドでハッシュ不一致を**エラー**にする (現状は警告のみ、§3.4) |
| D8 | スカラー計算に使う入玉ルールは**ネットに固定**する (初版 CSA 27点法・平手)。エンジンの `EnteringKingRule` オプションとは独立に計算する |
| D9 | クラウド (GPU なし) で検証するため、Simple 用の **CPU 量子化 forward** と、`gpu-kernels` の `*_cpu` 関数を組み合わせた **CPU float forward/backward (テスト専用)** を追加する |

## 2. 調査結果: tatara

### 2.1 Simple trainer で FT 出力が L1 にどう渡るか

結論: **連結済みの中間バッファを経由する**。カーネル内で FT 出力を直接読む構造ではないため、
L1 入力に列を足す改造は容易な部類。

- `bins/nnue_train/src/trainer_simple.rs` の `SimpleGpuWorkspace` が
  `combined: DeviceBuffer<f32>` (`batch × id.combined_dim()`) を確保している。
- forward:
  1. `sparse_ft_forward` が `ft_stm_out` / `ft_nstm_out` (`batch × ft_out`) を書く
  2. FT post カーネル (bias 加算＋活性化) が `combined` の視点別の区間に直接書く。
     FP32 既定経路は `simple_ft_post_fused_{crelu,screlu}` を視点ごとに 2 回起動し、
     列オフセット 0 / `ft_out` で書き分ける
  3. L1 は `cublas.sgemm_fwd_rowmajor(b, l1_out, l1_in, combined, l1_w, l1_pre)`
     (`l1_in = combined_dim`)＋ `bias_add_per_row`
- backward:
  - `dcombined = dl1_pre @ l1_w^T`、`l1_w_grad = combined^T @ dl1_pre` を cuBLAS で計算
  - FT 側の勾配は `simple_bwd_ft_act_{crelu,screlu}_fused` が `dcombined` の視点別区間を読む
- `l1_w` のデバイス上の配置は `[in=combined_dim][out=l1_out]`。量子化出力
  (`SimpleWeights::save_quantised`) では `[out][pad32(in)]` の i8 に並べ替えて書く。
- `SimpleId::combined_dim()` (`crates/nnue-format/src/simple_weights.rs`) が L1 入力次元の
  唯一の定義で、ワークスペース確保・重み初期化の fan-in・`l1_w` サイズ・export・
  ネットワークハッシュ (`compute_fc_hash`)・arch 文字列の `InputSlice[...]` すべてがここを
  参照する。

**L1 入力を `[FT_stm(256) | FT_nstm(256) | scalar(96)]` にするために必要な変更**

| 箇所 | 変更 |
|---|---|
| `SimpleId` | 密入力の種別フィールドを追加し、`combined_dim()` に 96 を加える |
| FP32 融合カーネル 4 本 | `simple_ft_post_fused_{crelu,screlu}`、`simple_bwd_ft_act_{crelu,screlu}_fused` は行ストライドを `2 * ft_columns` に**固定**している (`crates/cuda-native-runtime/kernels/native_kernels.cu` と `bins/nnue_train/src/kernels/simple.rs` の両方)。ストライド引数を追加し、呼び出し側は `combined_dim` を渡す。`stride == 2*ft_out` なら計算は従来と同一 |
| FP16 出力経路 | `simple_bias_act_fwd_fp16_in_*` / `simple_act_grad_to_fp16_*` は既にストライド (`l1_in`) を引数で受けている。変更不要 |
| Pairwise 経路 | `ft_post_perspective_fwd` はストライド `ft_out` 前提。D4 により初版は対象外 |
| スカラーの書き込み | アップロード済みのスカラーバッファ (`batch × 96`) を `combined` の列 `[2*ft_out, 2*ft_out+96)` へ写す小さなカーネル (またはストライド付きコピー) を FT post の後に追加 |
| L1 backward | 変更不要。`l1_w_grad` はスカラー列の分も既存 Sgemm で求まる。`dcombined` のスカラー列は計算されるが使わない (入力側に勾配は流さない) |
| 重み初期化 | スカラー列だけ「0＋微小ノイズ」にする。既存の一様初期化は fan-in に `combined_dim` を使うため、スカラー列を足すと FT 列の初期化幅も変わる点に注意 (FT 列の fan-in は従来の値を使う) |
| `--init-from` | 既存 HalfKP ネットからのワームスタートでは、ロード時のハッシュ照合が厳密なので、「スカラーなしのネットを読み、`l1_w` の列を拡張する」経路が別途必要 |

### 2.2 `--arch` の任意次元と L1 入力次元

- `simple --arch <l1>x2-<l2>-<l3>` と `--l1/--l2/--l3` で FT 幅と隠れ層を指定する。
  L1 入力次元は CLI から直接は指定せず、`SimpleId::combined_dim()` が導出する。
- したがって「L1 入力 = 2×FT幅＋96」は CLI の次元指定を変えず、`SimpleId` の新フィールドで
  表すのが自然 (D1)。arch 文字列の `AffineTransformSparseInput[32<-608](InputSlice[608(0:608)])`
  も `combined_dim()` から自動で変わる。

### 2.3 データローダ

- `crates/nnue-train/src/dataloader.rs` の `Batch` は `stm_indices` / `nstm_indices` /
  `nnz` / `score` / `wdl` / `per_pos_norm` を持つ。特徴抽出は `Batch::push_decoded(&ShogiBoard)`
  で、prefetch worker が 1 局面につき 1 回だけデコードした `ShogiBoard` を使い回す。
- `ShogiBoard` (`crates/shogi-format/src/packed_sfen.rs`) は盤面 81 マス・両者の持ち駒・
  手番・両玉の位置を持つ。スカラー計算に必要な情報はすべてここにあり、王手判定は不要
  (「宣言条件の成立」は王手を除外した定義のため)。
- 追加するもの: `Batch` に `Option<Vec<f32>>` (`batch × 96`) の密入力、
  `InputUploadRing` (`trainer_common.rs`) に pinned バッファ 1 組、ワークスペースに
  active / back のデバイスバッファ 1 組。スカラーなしの構成では確保しない。
- スカラー計算は `shogi-features` クレートに置く。このクレートは GitHub Actions の CI で
  ビルド・テストされる (CUDA 不要)。

### 2.4 CPU 版の forward/backward・CPU で学習を回すモード

結論: **CPU で学習を回すモードは無い**。ただし部品は揃っている。

| 有無 | 内容 |
|---|---|
| なし | CPU 学習モード。`TrainerBackend` を実装しているのは GPU トレーナ (`trainer_common.rs` のマクロ経由) とテスト用の `MockBackend` だけ。`nnue-trainer --no-default-features` は CI 用に CPU でテストできる部分だけをビルドする構成で、学習は実行できない |
| あり | カーネル単位の CPU 参照実装。`gpu-kernels` クレートの `*_cpu` 関数 (`sparse_ft_forward_cpu` / `sparse_ft_backward_cpu` / `dense_mm_fwd_cpu` / `dense_mm_bwd_{input,weight}_cpu` / `bias_grad_cpu` / `crelu_fwd_cpu` / `crelu_grad_cpu` / `loss_wdl_cpu` / `loss_wrm_cpu` / `radam_step_cpu` ほか)。`gpu-kernels` は CUDA なしでビルドでき、CI の workspace に含まれる |
| あり | `gpu_cpu_equivalence_tests` (GPU 必須)。上記 `*_cpu` と GPU カーネルの数値一致を確認する |
| LayerStack のみ | 量子化 CPU forward。`bins/eval_sfen` の `forward_one_raw` は LayerStack の `.bin` だけを読む |
| なし | Simple の量子化 CPU forward、Simple 全体の float CPU forward |

**影響**: GPU 未構築の間に integer-exact 突合と「数ステップで loss が減る」確認を
クラウドで行うには、次の 2 つを追加する必要がある (D9)。

1. Simple(＋スカラー) の量子化 CPU forward。FT i16 累積 → u8 化 (0〜127) → i8 アフィン
   (`>> 6`) → … と、エンジンと同じ整数演算をする。`nnue-format` に置けば CI で回る。
2. テスト専用の Simple float CPU forward/backward。`*_cpu` 関数の合成で書ける。
   HalfKP 256 は FT が約 3,200 万パラメータなので、小さい batch なら CPU でも数ステップ回せる。

### 2.5 その他の tatara 側の事実

- `FeatureSet` は `HalfKp` / `HalfKaSplit` / `HalfKaMerged` / `HalfKaHmSplit` /
  `HalfKaHmMerged` の 5 種。`HalfKp` のハッシュは `0x5D69D5B8` で、YaneuraOu の
  `HalfKP<Friend>` (`0x5D69D5B9 ^ 1`) と一致する。
- Simple の量子化 `.bin` は YaneuraOu 従来 NNUE と同じ枠組み
  (`NNUE_VERSION = 0x7AF32F16`、ヘッダ・FT ハッシュ・密層ハッシュ、L1 は i8 ×
  `l1_out × pad32(l1_in)`) で書かれる。YaneuraOu の `AffineTransformSparseInput` も
  `kOutputDimensions × kPaddedInputDimensions` を読み、`kMaxSimdWidth = 32` 固定なので
  608 列は追加パディングなしで一致する。**実際に YaneuraOu で読み込めるかは本調査では
  確認していない** (export・推論の段階で byte-golden テストと実ロードで確認する)。
- ライブラリとしての SFEN パーサが無い (`bins/eval_sfen` と
  `crates/shogi-features/examples/dump_effect_bucket_indices.rs` に個別実装がある)。
  golden ファイル (SFEN＋期待値) を読むテストのために `shogi-format` へ寄せる必要がある。

## 3. 調査結果: YaneuraOu

### 3.1 推論経路

- `source/eval/nnue/evaluate_nnue.cpp` の `ComputeScore` (従来 NNUE 経路):
  `feature_transformer.Transform(pos, transformed_features, refresh)` で
  `transformed_features[2×256]` (u8、0〜127) を作り、`network[0].Propagate(transformed_features, buffer)`
  に渡す。出力は `output[0] / FV_SCALE`。
- ネットワーク定義 (`architectures/halfkp_256x2-32-32.h`):
  `InputSlice<512>` → `AffineTransformSparseInput<…, 32>` → `ClippedReLU` →
  `AffineTransform<…, 32>` → `ClippedReLU` → `AffineTransform<…, 1>`。
- 改造方針: 新アーキのヘッダで `InputSlice<608>` を使い、`ComputeScore` では
  `transformed_features` を 608 バイト確保して、`Transform` の後ろ 96 バイトにスカラーを書く。
  ハッシュと構造文字列にスカラー種別を入れるため、`InputSlice` 相当の新しい入力層クラスを
  作る (D7)。

### 3.2 `EnteringKingRule` と仕様書の表の照合

実装: `source/types.h` (`enum EnteringKingRule`)、`source/position.cpp`
(`Position::update_entering_point()`、`Position::DeclarationWin()`)。

| 項目 | 仕様書 | YaneuraOu の実装 | 判定 |
|---|---|---|---|
| 27点法の必要点数 | 先手28・後手27 | `EKR_27_POINT`: `points[BLACK]=28`、`points[WHITE]=27` | 一致 |
| 24点法の必要点数 | 31点以上で勝ち | `EKR_24_POINT`: 先後とも 31 | 一致 |
| 24点法の引き分け帯 (24〜30点) | 引き分け | 実装なし (31未満は宣言不可として扱うだけ) | 仕様書どおり初版では特徴化しないので問題なし |
| 点数の対象 | 敵陣内の玉以外の自駒＋持ち駒 | `score = 敵陣内の自駒数 + 敵陣内の大駒数×4 − 1 (玉) + 持ち駒点` | 一致 |
| 点数の重み | 大駒5点・小駒1点 | 大駒は `BISHOP_HORSE` / `ROOK_DRAGON` (角・馬・飛・竜)。と金など成駒は小駒扱い | 一致 (成駒の扱いを明記すべき) |
| 敵陣内の駒数 | 玉以外10枚以上 | `(pieces(us) & enemy_field) >= 11` (玉込み) | 一致 |
| 玉の位置 | 敵陣三段目以内 | `enemy_field(us) & king_square` | 一致 |
| 王手 | 王手されていない | `checkers()` が空 | 一致 |
| 持ち時間 | 残っている | `DeclarationWin` では判定しない | 特徴にも入れないので問題なし |
| 手番 | 記載なし | (a) 宣言側の手番であること。`DeclarationWin` は手番側についてだけ呼ばれる前提 | **差異**。特徴の「宣言条件の成立」は両玉について計算するので、手番条件も除外した (b)(c)(d) の論理積と定義する |
| 駒落ち | 記載なし | `EKR_27_POINT_H` / `EKR_24_POINT_H` は盤上＋持ち駒の総点が 56 に満たない分を後手 (上手) の必要点数から引く | **差異**。初版の `Csa27` は `EKR_27_POINT` (平手) に対応させ、駒落ち対応版は扱わない |
| トライルール | 記載なし | `EKR_TRY_RULE` がある | 対象外 |
| 既定値 | 27点法 | エンジンオプション `EnteringKingRule` の既定は `EKR_27_POINT` | 一致 |

必要点数はエンジン内では `Position::enteringKingPoint[]` (`set_ekr()` →
`update_entering_point()` で設定) に入る。D8 のとおり、評価関数のスカラー計算はこの値を
参照せず、ネットに固定したルール表 (`Csa27`: 28 / 27) を使う。エンジンオプションと
ネットのルールが異なる場合は、ロード時に警告を出す。

### 3.3 FV_SCALE

- `FV_SCALE` はエンジンオプション (既定 16、範囲 1〜128) で、固定値ではない。
- tatara の Simple は `fv_scale = round(127 × 64 / --scale)` を arch 文字列に書くが、
  YaneuraOu は arch 文字列の `fv_scale` を読まない。

したがって `--scale 508` は「`FV_SCALE` を既定の 16 のまま使う場合」の条件。別の
`--scale` で学習した場合は、エンジンの `FV_SCALE` オプションを合わせれば使える。

### 3.4 ハッシュ不一致の扱い

従来 NNUE のロード (`LoadAndShare` と `Detail::ReadParameters`) は、ネットワークハッシュ・
FT ハッシュ・密層ハッシュの不一致を **警告 (`info string Warning: ...`) にするだけで
読み込みを続ける**。仕様書 §1.1 の「既存 `.nnue` の誤ロードをロード時に弾く」は、
ハッシュを新しく採番するだけでは成立しない。新アーキのビルドフラグ下ではハッシュ不一致を
`FileMismatch` で返すようにする (D7)。

### 3.5 名前の衝突

`source/eval/nnue/features/a2.h` に upstream 由来の `Features::A2` (玉を含む全駒の
BonaPiece、ハッシュ `0xA20DCB9B`) が既にある。仕様書の「A2」とは無関係。新規コードでは
`A2` を識別子・ファイル名に使わず、`nyugyoku_scalar` のように内容を表す名前にする。

## 4. 決定の詳細と理由

### D1. Simple の拡張オプションにする

- Simple は L1 入力が `combined` バッファで実体化されており、L1 は入力次元を引数で受ける
  cuBLAS Sgemm なので、列の追加がほぼ `combined_dim()` の変更で伝播する。
- LayerStack は L1 をバケット別の融合カーネルで計算しており (L1 forward をバケット層と
  共有層で 1 カーネルにまとめている)、列の追加はカーネル改造が大きい。LayerStack との
  併用はアイデア A3 で改めて扱う。
- 新サブコマンドにすると、CLI・チェックポイント・export の重複が増える。
- CLI の形 (案): `simple --dense-input nyugyoku --nyugyoku-rule csa27 --nyugyoku-encoding thermometer`。
  既定は密入力なし。

### D2–D3. `combined` の末尾に書き、融合カーネルにストライドを足す

- 既存の FP32 融合カーネルは `combined` の行ストライドを `2*ft_out` と仮定しているので、
  そのまま `combined_dim` を 608 にすると行がずれる。ストライド引数を足せば、既存構成
  (`stride = 2*ft_out`) は同じアドレス計算になり bit-identical を保てる。
- カーネルの変更は CUDA C++ (既定の native backend) と cuda-oxide (比較用 oracle) の
  両方に入れる。`scripts/check-native-cuda-parity.sh` と GPU 同等性テストは GPU 環境で確認する。

### D4. 初版は CReLU のみ

- 目標の `256x2-32-32` と YaneuraOu の `ClippedReLU` に合う最小構成。
- SCReLU / Pairwise はストライド・量子化スケールの確認範囲が広がるので後回しにする。

### D5. ホスト側で u8 量子化してから学習に渡す

- エンジンの L1 入力は u8 (0〜127)。「持ち駒点数 / 必要点」のような連続値も
  `round(x*127)` に丸めた値を学習でも使えば、スカラー部分の量子化誤差がなくなる。
- 仕様書の「int8 化」はエンジン側の型に合わせて **u8** と読み替える (値域 0〜127 なので
  実質同じ)。

### D6. 共有 golden ファイル

- Rust と C++ でコードは共有できないため、ルール表・区切り表・ブロック順はそれぞれに
  定数として持ち、同じ golden ファイル (SFEN と期待 u8 列 96 個) を両リポのテストが読む。
- 正本は tatara 側に置き、YaneuraOu 側には同じ内容を複製する。両方のテストで
  ファイルの SHA-256 も確認し、片方だけの更新を検出する。

### D7. ハッシュと arch 文字列

- L1 入力次元が 608 になるだけで `compute_fc_hash` / `InputSlice::GetHashValue` は変わるが、
  ルールや符号化が違っても同じ値になる。そこで「スカラー種別・ルール・符号化・ブロック配置の
  版」から作る 32bit 値をさらに XOR し、arch 文字列にもトークン
  (例: `DenseInput=NyugyokuScalar(csa27,thermometer)[96]`) を入れる。
- YaneuraOu 側は新しい入力層クラスの `GetHashValue` / `GetStructureString` に同じ値・文字列を
  持たせ、ハッシュ不一致はエラーにする。

### D8. 入玉ルールはネットに固定

- 学習時のスカラーは学習データ生成時のルールに依存しないよう、ネットのルールで計算する。
- エンジンで別ルール (24点法など) を選んでも評価値の意味は変わらない。ルールの差異は
  探索側の宣言判定が受け持つ。

### D9. CPU で検証できる経路を足す

- 量子化 CPU forward があれば、GPU なしで「export した `.bin` を tatara の CPU forward と
  YaneuraOu の `Evaluate()` で計算して ±0 で一致」まで確認できる。
- float CPU forward/backward は、ワームスタート直後の出力一致 (関数保存) と、数ステップで
  loss が減ることを CPU で確認するのに使う。GPU 版との一致は GPU 環境で確認する。

## 5. 段階計画への反映

実施順は **1 → 2 → 3 → 5 → 4 → 6** (hs 決定)。段階5の量子化 CPU forward と YaneuraOu 側の
推論は乱数の重みで integer-exact 突合でき、GPU なしで検証を完了できるため先に行う。
段階5で使う L1 入力次元のデータ定義 (`SimpleId::combined_dim()` の 608 化など) は段階5で
入れ、FP32 融合カーネルの行幅引数化など GPU 経路の変更は段階4に残す。

| 段階 | 変更点 |
|---|---|
| 2 (スカラー計算) | `shogi-features` にスカラー計算、`shogi-format` に SFEN パーサを追加。YaneuraOu は `source/eval/nnue/features/nyugyoku_scalar.h` (仮)。golden ファイルを作る |
| 3 (データローダ) | `Batch` の密入力、`InputUploadRing` の pinned バッファ |
| 4 (trainer) | 融合カーネル 4 本のストライド引数 (CUDA C++ と cuda-oxide)、スカラー列の書き込み、初期化、`--init-from` の列拡張。GPU 未検証のまま draft |
| 5 (export・推論) | `SimpleId` の新フィールド、ハッシュ・arch 文字列、Simple の量子化 CPU forward。YaneuraOu の新入力層クラス・新アーキヘッダ・ビルドフラグ・ハッシュ不一致のエラー化 |
| 6 (検証) | スカラー値の突合と integer-exact 突合は CPU で実施 |

## 6. hs の回答

| # | 論点 | 回答 |
|---|---|---|
| 1 | ADR と仕様書の置き場所 | `docs/` に置いたままにする。tatara の `CLAUDE.md` に `docs/nnue-*` / `docs/adr-*` を「dated header 禁止」「ADR は repo に置かない」の例外として明記した |
| 2 | 初版の活性化 (D4) | CReLU のみでよい |
| 3 | 入玉ルールの扱い (D8) | ネット側に固定 (27点法)。ルール種別をハッシュと arch 文字列に含め、エンジンの `EnteringKingRule` と食い違う場合は警告を出す (export・推論の段階で実装) |
| 4 | 駒落ち (`EKR_27_POINT_H`) | 扱わない |
| 5 | 「宣言条件の成立」ビット | 手番・王手・持ち時間を除いた条件で判定する |

## 7. 段階3: `Batch` 上の入玉スカラーの型とメモリ配置

### 決定

- `Batch` に `nyugyoku_scalar: Option<NyugyokuScalarConfig>` と `nyugyoku_scalars: Vec<u8>` を
  足す。スカラーは **u8 (0〜127) のまま**持ち、f32 には変換しない。
- 配置は `[batch_size, 96]` の row-major (`bi * 96 + k`)、1 局面 96 バイト連続。パディングは
  入れない (96 は 32 の倍数)。
- `nyugyoku_scalar` が `None` の構成では buffer を確保せず (`Vec` は容量 0)、
  `push_decoded` もスカラーを計算しない。`[n_positions, batch_size)` の行や `reset` の扱いは
  既存 buffer と同じ契約 (`n_positions` までしか読まない) に揃える。
- 設定は `TrainingConfig::nyugyoku_scalar` から学習 loader (`BucketedPrefetchedLoader`) と
  held-out 検証集合 (`HeldoutSet`) に渡す。CLI はまだ無く、常に `None`
  (`SimpleId` の密入力種別と一緒に段階5で入れる)。rescore と単一 worker の
  `PrefetchedLoader` は対象外 (スカラー付きネットの推論を使う段階で必要なら足す)。
- スカラー有効時に玉が欠けた局面は `InvalidData` で学習を止める。判定は特徴抽出の前に行い、
  batch の buffer を書き換えない。

### 理由

| 観点 | u8 で持つ場合 | f32 で持つ場合 |
|---|---|---|
| H2D 転送量 (1 局面) | 96 B | 384 B |
| 既存の H2D (HalfKP: stm/nstm index 38 × 4 B × 2 + score/wdl/nnz) | 約 316 B に対し +30% | +122% |
| 段階5 (量子化 CPU forward・integer-exact 突合) | エンジンの L1 入力 (`uint8_t`) と同じ値をそのまま使える | u8 に戻す必要がある |
| 段階4 (GPU) | `combined` の列 `[2*ft_out, 2*ft_out+96)` へ書くカーネルで `u8 / 127` に変換する。変換はスカラー列の書き込みと同じカーネルで済む | 変換不要だが転送が 4 倍 |
| float の参照実装 (CPU) | 行ごとに `u8 / 127` にすればよい (`nyugyoku_scalars_f32` と同じ値) | そのまま使える |

- 学習とエンジンで同じ値を見るため、スカラーは量子化済みの u8 が正 (D5)。f32 はその派生でしかない。
- row-major の 96 バイト連続にしておくと、段階4では `InputUploadRing` に u8 の pinned buffer を
  1 組足して 1 回の H2D で送れ、GPU 側は `b × 96` の 2D 配列として読める。

### スループットの計測

`crates/nnue-train/examples/dataloader_throughput.rs` で計測した (sample.psv の 100 局面を
2,000 回繰り返した 20 万局面の PSV、HalfKP、batch 16,384、各 7 回の中央値)。計測環境は
クラウドの 4 vCPU (Intel Xeon 2.1 GHz) で、同じ構成でも run 間で ±10% 程度ぶれる。

| 経路 | スカラーなし | スカラーあり (27点法・thermometer) | 変化 |
|---|---|---|---|
| 学習 loader、4 worker | 2.49 M pos/s | 2.36 M pos/s | -5.2% |
| 学習 loader、1 worker | 637 k pos/s | 589 k pos/s | -7.5% |
| `push_decoded` のみ (1 thread、decode 済み局面) | 3.38 M pos/s | 2.46 M pos/s | -27% |
| 同上、HalfKA_hm (merged) | 3.69 M pos/s | 2.84 M pos/s | -23% |

- 最初の素直な実装 (マスごとに分岐し、色で添字を引く配列に累積) では学習 loader が
  -11〜-13% だった。盤面走査を「(駒種, 色) → 寄与」と「マス → 敵陣 field の mask」の表引きで
  1 つの u64 に累積する形に、thermometer を表のコピーにして、1 局面あたりのスカラー計算を
  約 220 ns から約 80 ns に減らした。
- 学習 loader 全体では decode と I/O の比重が大きいので低下は 5〜8%。GPU 学習では loader の
  worker 数を増やせば吸収できる範囲と見ている (実際の学習速度への影響は段階4で GPU を使って確認する)。

### 判断 (hs)

- 学習 loader 全体で -5〜-8% の低下は**現時点では許容**する。本番学習で dataloader が GPU の
  律速になった場合に再検討する (候補: worker 数の増加、スカラー計算の更なる表引き化、
  decode と同じ盤面走査への統合)。
- rescore と単一 worker の `PrefetchedLoader` は、スカラーありのアーキが指定されたら黙って
  スカラーなしで動かさず、エラーにする (CLI を入れる段階5で実装)。
- HCPE 経路は、HCPE エンコーダを作らず「loader が詰めたスカラー」と「同じ record を復号した
  盤面から直接計算したスカラー」の一致で検証する (`hcpe_loader_matches_direct_computation`)。

## 8. 段階5: export と推論、integer-exact 突合

### 実装

- tatara
  - `SimpleId::nyugyoku_scalar` (`Option<NyugyokuScalarConfig>`)。`Some` のとき `combined_dim()` は
    FT 出力の連結＋96 (HalfKP 256x2 で 608)、`ft_output_dim()` が FT 部分。CReLU のみ。
  - hash: L1 入力層の hash `0xEC42E90D ^ l1_input` に、トークン
    `NyugyokuScalar(<rule>,<encoding>)[96]` の FNV-1a 32bit を XOR する (`fc_hash`)。arch 文字列の
    入力層は `InputSlice[608(0:512)+NyugyokuScalar(csa27,thermometer)[96](512:608)]`。
    密入力なしの構成の hash・arch 文字列・保存バイト列は変更前に固定したテストどおり不変。
  - `SimpleQuantised`: `.bin` を整数のまま読み (header 検査は `SimpleWeights::load` と共通)、
    YaneuraOu の従来 NNUE と同じ整数演算で forward する。
  - CLI: `simple --nyugyoku-rule csa27|csa24 [--nyugyoku-encoding thermometer|onehot]`。GPU で
    学習する経路は段階4なので、指定すると CUDA 初期化の前にエラーにする。
  - `examples/simple_quantised_eval`: 乱数ネットの書き出しと SFEN の評価 (`--stats` で層ごとの
    飽和率)。
- YaneuraOu
  - edition `YANEURAOU_ENGINE_NNUE_HALFKP_NYUGYOKU_256X2_32_32`、`NYUGYOKU_RULE` /
    `NYUGYOKU_ENCODING` でネットのルール・符号化を選ぶ。
  - 入力層 `NyugyokuScalarInputSlice` が L1 入力次元・hash・構造文字列を定め、
    `TransformInput()` が FT 出力の直後に入玉スカラーを書く。
  - 新アーキのビルドではハッシュ不一致を `FileMismatch` にする。ネットのルールとエンジンの
    `EnteringKingRule` が違うと `isready` で警告する。
  - スカラー計算は段階2の bitboard 実装 (popcount、81 マス走査なし) のまま。

### integer-exact 突合

局面: YaneuraOu の `test nn gen_sfens` による平手からのランダムプレイアウト 10,000 局面
(40% の対局は手番側の玉を前に進める手を優先) ＋ golden 8 局面。玉が敵陣にいる局面 3,346
(うち相入玉 2,263)、宣言条件を満たす玉がある局面 6。

| ネット | ルール・符号化 | スカラー列の weight | エンジン | 一致 (駒が揃った 10,006 局面) |
|---|---|---|---|---|
| A | csa27・thermometer | ±127 | 新アーキ (csa27) | 10,006 / 10,006 |
| B | csa27・thermometer | ±8 (線形域) | 新アーキ (csa27) | 10,006 / 10,006 |
| C | csa24・one-hot | ±127 | 新アーキ (csa24・onehot) | 10,006 / 10,006 |
| D | 入玉スカラーなし | – | 従来 HalfKP | 10,006 / 10,006 |

- 出力層の生の値・評価値の両方を比較した。A では hidden1 の 53% が 0、44% が 127 に張り付き、
  飽和・クリップの経路を通っている。B は線形域を多く通す構成。
- 駒の欠けた golden 2 局面 (`both_entered`、`black_entered`) は B・C・D で値がずれた。原因は
  下記の HalfKP の既存の差異で、YaneuraOu 側の挙動を tatara で再現すると 5 件すべて一致した
  (入玉スカラーの部分は一致している)。
- 新アーキは別ルール・別符号化のネットと従来 HalfKP のネットを読み込み時にエラーにすることを
  確認した (従来 HalfKP のビルドは従来どおり警告のみ)。

### 既存の差異 (段階5とは無関係、未修正)

- **HalfKP の駒の欠け**: YaneuraOu の `HalfKP::AppendActiveIndices` は常に 38 個の index を出し、
  盤上・手駒に無い駒は `BONA_PIECE_ZERO` (index `玉マス * fe_end`) として数える。tatara の
  `extract_active_features` は存在する駒だけを出す。駒落ちや駒箱のある局面でだけ FT が
  食い違い、平手の対局から作った教師データ (常に 40 枚) には影響しない。直すなら tatara 側で
  欠けた駒の分の index を出すか、駒の欠けた局面を教師から除く。
- YaneuraOu (従来 HalfKP のビルドでも) は駒の欠けた手作りの局面 (`black_entered`) の探索中に
  segfault した。NPS の計測局面から駒の欠けた局面は除いた。
- YaneuraOu の NNUE ビルドの `unittest` と `test nn ...` は、`isready` の前に打つと評価関数が
  未読み込みのため segfault する。

### NPS

`bench 64 1 1000 <局面> movetime` (1 thread、乱数ネット)、局面はプレイアウトの 40 局面＋駒の
揃った golden 6 局面、各 3 回。計測環境は 4 vCPU Intel Xeon 2.1 GHz。

| エンジン | run 1 | run 2 | run 3 | 中央値 |
|---|---|---|---|---|
| 従来 HalfKP 256x2-32-32 | 966,590 | 931,607 | 950,383 | 950,383 |
| 新アーキ (入玉スカラー) | 833,816 | 810,265 | 813,122 | 813,122 |

**NPS は -14.4%** で、5% を超える。乱数ネット同士では探索木の形も変わるので、評価関数だけの
時間 (`test nn bench_eval`、プレイアウト 10,000 局面の合法手ごとに do_move → evaluate →
undo_move、3 回の中央値) でも比べた。

| エンジン | 評価 1 回 (差分更新つき) |
|---|---|
| 従来 HalfKP | 477 ns |
| 新アーキでスカラーを 0 で埋めた計測用ビルド (L1 608 のみ) | 496 ns (+4%) |
| 新アーキ | 563 ns (+18%) |

スカラー計算 1 回 (両玉・96 次元、`test nn bench_scalar`) は約 64 ns。

### 原因

1. **スカラー計算** (増分の約 8 割、約 65 ns/評価): 評価のたびに両玉の状態 (popcount 約 8 回と
   持ち駒点数) を求め、44 次元のブロックを 2 つ書く。差分更新が無い。
2. **L1 の入力幅** (約 2 割、約 19 ns): sparse 入力の affine は 0 でない 4 バイト単位ごとに
   32 出力の積和をするので、スカラー 96 個 (約 37% が 0 でない) の分だけ仕事が増える。

### 改善案 (A と B を段階5bで実装。§9)

- **A. スカラーの L1 への寄与を表引きにする (本命)**: 各ブロックは玉ごとの小さな整数
  (前進段数 0〜8、敵陣内の駒数 0〜10、残り点数 0〜29 ×2、持ち駒比率 0〜必要点、フラグ 3 つ)
  だけで決まる。ネット読み込み時に「値 → L1 の 32 出力への寄与 (i32)」の表を部分ブロックごとに
  作れば、スカラー部分の L1 は 1 玉あたり 8 回程度の 32 要素ベクトル加算で済み、スカラーの
  書き出しと sparse affine のスカラー部分が無くなる。表は 1 玉あたり約 120 項目 × 32 × 4 B ≈
  15 KB (両玉で約 30 KB)。L1 の積和は ClippedReLU の前の整数の和なので、加算の順序を変えても
  結果は変わらず integer-exact を保てる。見込みはスカラー＋L1 スカラー部分が約 85 ns → 10〜20 ns、
  NPS の低下は数 % 程度 (未計測の見積り)。
- **B. 玉ごとの状態を StateInfo にキャッシュする**: 状態が変わるのは、玉が動く・敵陣に駒が
  出入りする・持ち駒が変わる手だけなので、それ以外は親局面の値を使う。A と組み合わせられる。
- **C. 計算そのものの軽量化**: thermometer を表のコピーにし両玉をまとめて求める (tatara 側で
  220 → 80 ns にした方法)。効果は A より小さい。

推奨は A (必要なら B を追加)。

## 9. 段階5b: 新アーキの NPS 低下への対策

YaneuraOu のエンジン内部だけを変えた。評価関数ファイルの形式 (ハッシュ値・構造文字列・
バイト列) と tatara は変えていない。

### 結論

**NPS の低下は約 4% (同じ探索木で比べて -3.3〜-3.8%)** になり、目標の 5% 以内に入った。ただし
余裕は小さく、計測環境の揺れ (同じ源の HalfKP のビルド同士で約 1%、逐次実行では回ごとに
±5%) と同じ程度の大きさである。

### 計測の分解 (実装前)

`test nn bench_scalar <playout 10,000 局面> 20 full|states|write`、7 回の中央値。

| 部分 | 時間 |
|---|---|
| full (両玉の状態の計算＋96 次元の書き込み) | 63.9 ns |
| states (両玉の状態の計算: popcount 約 8 回と持ち駒点数) | 37.1 ns |
| write (求めた状態から 96 次元を書くだけ) | 32.9 ns |

### 実装

1. **L1 への寄与の表引き** (`AffineTransformNyugyokuScalar`、`L1ContributionTable`)
   - 評価関数ファイル上の L1 は従来どおり `AffineTransformSparseInput[32<-608]` で、そのまま
     読む。推論では 96 次元を作らず、FT 出力 512 の部分だけを sparse affine で計算し、入玉
     スカラーの部分は読み込み時に作る表を引いて足す。
   - 玉 1 つ分のブロック (44 列) を、玉の状態の小さな整数だけで決まる「表引きの field」と、
     値をそのまま weight に掛ける「線形の列」に分ける。
     - thermometer: field 4 つ (前進段数＋先手＋宣言条件の成立 36 行 / 敵陣内の駒数 11 /
       宣言点数の残り 30 / 潜在宣言点数の残り 30)、線形の列 1 つ (持ち駒点数の比率)
     - one-hot: field 1 つ (前進段数＋先手＋宣言条件の成立 36 行)、線形の列 4 つ (駒数・
       宣言点数・残り・持ち駒点数の比率)
   - 表引きの列は 0 か 127 のフラグだけなので、表の行は「立つ列の weight の和」を int16 で
     持ち (32 出力で 64 バイト = 1 キャッシュライン)、全 field を足してから 127 を掛ける。
     両玉の全列が立って weight が全部 -128 でも和は int16 に収まる (単体テストで確認)。
     加算の順序を変えても ClippedReLU の前の整数の和なので、integer-exact は保たれる。
   - 符号化はネットに固定なので表の型に持たせ、field の数と行の位置をコンパイル時の定数に
     して、評価のたびのループを展開させる。
   - **表のサイズ: thermometer 13,952 バイト** (表引き 2 側 × 107 行 × 64 B ＋ 線形の列
     2 側 × 1 列 × 128 B)、one-hot 5,632 バイト。評価 1 回で読むのは thermometer で 12
     キャッシュライン (表引きの 8 行で 8、線形の列の 2 行で 4)。
   - 表の行は探索中に L1 キャッシュから追い出されていることが多い (cachegrind で、表引きを
     int32 で入れた時点の新アーキは HalfKP より 1 node あたり D1 の読み込みミスが 19 回多く、
     うち 16 回が表の読み込み)。玉の状態を FT の差分更新より先に求め、引く行を prefetch
     してから FT を更新する。
2. **親局面からの再利用** (`UpdateNyugyokuInputs`): accumulator に玉ごとの表引きの入力
   (`L1Inputs`) を持たせ、直前の手 (`StateInfo::dirtyPiece`) で変わり得ない玉は親局面の値を
   使う。指した側の玉は「駒を取らない・打たない・玉以外を動かす・敵陣のマスに出入りしない」
   手のとき、相手の玉は駒を取られないとき引き継ぐ。親局面が評価されていなければ計算する。
   `test nn test_nyugyoku_reuse 1000` (ランダムプレイアウト 1,000 局、半数は玉を前に進める
   対局、ときどき null move) で毎回計算した値と一致することを確かめた: 305,065 局面
   (null move の後 27,223)。探索中 (bench) の内訳は、評価 1 回あたりの玉の状態の計算が
   2 回 → 平均 1.18 回 (親が未評価 22%、駒を取る手 24%、駒を打つ手 13%、玉が動く 4%、
   敵陣への出入り 7%、両玉とも引き継ぎ 29%)。
3. L1 層を FT の差分更新の前から参照するため、`AffineTransform` と `ClippedReLU` に直前の層を
   返す const の getter を足した (既存アーキの挙動は変わらない)。

### 検証

- integer-exact (段階5と同じ 4 ネット × 10,008 局面): A 0 件、B・C は 2 件、D は 1 件の
  不一致で、すべて段階5で記録した駒の欠けた golden 局面 (HalfKP の既存の差異)。値も段階5と
  同じ。
- `unittest` 157/157 (golden、表引きと 96 次元を作って掛けた和の一致を乱数の weight と全
  -128 の weight で、両符号化・両ルールで確認)。`test nn test_accumulator 50` のチェックサムは
  変更前と同じ (新アーキ 2 構成・従来 HalfKP)。

### NPS

計測環境は 4 vCPU Intel Xeon 2.1 GHz (共有 VM)。逐次に 1 本ずつ測ると回ごとに ±5% 揺れ、
数 % の差を測れなかったので、次の方法を使った。

- **同じ探索木で比べる**: ネット D (HalfKP の乱数ネット) の L1 にスカラー 96 列を weight 0 で
  足したネット A0 は D と同じ評価値を返すので、ノード数を固定した bench で HalfKP (D) と
  新アーキ (A0) は同じ木を探索する (ノード数 12,348,820 で一致)。評価にかかる時間の差だけが
  NPS の差になる。
- **同時に走らせて比べる**: 2 つのエンジンを別のコアで同時に走らせ (回ごとにコアを入れ替える)、
  回ごとの比の中央値を取る。環境の揺れが両方に同じようにかかる。

`bench 64 1 300000 <局面> nodes` (1 thread、局面は段階5と同じ 46 局面)。

| 版 | 同時実行 (回ごとの比の中央値) | 逐次 |
|---|---|---|
| 段階5 (96 次元を作って L1 608) | – | -14.4% (movetime、別の木) |
| 表引き (int32)＋再利用 | – | -7.1% (5 回) |
| 同、計測用に表引きの加算を外す (A0 なので同じ木) | – | +2% (HalfKP と差なし) |
| 同、加算と玉の状態の計算を外す | -1.3% (8 回) | – |
| 同、加算だけ外す | -2.3% (8 回) | – |
| 表引き (int16)＋L1 の直前で prefetch | -5.7% (12 回) | -4.6% (8 回) |
| ＋FT の更新の前に prefetch | -5.2% (12 回) | – |
| ＋表の型に符号化を持たせる | -4.9% (12 回) | – |
| **最終** (＋宣言条件の成立を field 0 にまとめる) | **-3.8%** (16 回) | **-3.3%** (6 回) |

- 同じ源の HalfKP の 2 つのビルド (段階5bの途中と最後) を同時実行で比べると -0.8% (8 回)。
  上の表の 1 % 前後の差はこの揺れと区別できない。最終の行は最後のビルド同士の比較。
- 再利用を外したビルドは、表引き (int32)＋prefetch の時点の逐次計測で再利用ありより約 1%
  遅かった (-5.6% と -4.6%)。
- 段階5と同じ方法 (`bench 64 1 1000 <局面> movetime`、HalfKP は D、新アーキは A、各 4 回の
  中央値): HalfKP 938,722、新アーキ 866,802 (**-7.7%**)。同じ新アーキのビルドでネットを A0 に
  替えると 920,242 (-2.0%) になる。A と A0 は評価の計算量が同じなので、差の大半はネットの
  違いによる探索木の形の違いで、評価の速さではない。
- `test nn bench_eval` (局面ごとに合法手で do_move → evaluate → undo_move) は環境の揺れが
  大きく (同じエンジンで 495〜657 ns)、5 回の中央値は HalfKP 542 ns、新アーキ 540 ns で差が
  見えなかった (段階5は +18%)。この計測では親局面を評価しないので再利用は効かない。

### 残る差と、さらに下げる案 (未実装)

最終版で残る約 4% の内訳 (同時実行の差から推定): 新アーキの配線 (accumulator の追加分、
入力 buffer への玉の状態の書き込み) 約 1%、玉の状態の計算 約 1%、表引きの加算と prefetch
約 2%。

- 玉ごとの寄与 (int16 の和) を accumulator に持たせ、引き継げる玉は表を引かない。ただし玉の
  側 (手番側・相手側) が 1 手ごとに入れ替わり、側ごとに表が違うので、引き継げるのは 2 手前
  以前の値になり、効果は小さいと見込む。
- 親局面が未評価のとき、さらに前の評価済みの局面まで遡って引き継ぐ (評価の 22% が該当)。
- 駒を取る手 (評価の 24%) で、取られた側の状態を差分で更新する (盤上の駒数・大駒数・敵陣内の
  駒数が 1 つ減るだけ)。
