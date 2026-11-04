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

| 段階 | 変更点 |
|---|---|
| 2 (スカラー計算) | `shogi-features` にスカラー計算、`shogi-format` に SFEN パーサを追加。YaneuraOu は `source/eval/nnue/features/nyugyoku_scalar.h` (仮)。golden ファイルを作る |
| 3 (データローダ) | `Batch` の密入力、`InputUploadRing` の pinned バッファ |
| 4 (trainer) | 融合カーネル 4 本のストライド引数 (CUDA C++ と cuda-oxide)、スカラー列の書き込み、初期化、`--init-from` の列拡張。GPU 未検証のまま draft |
| 5 (export・推論) | `SimpleId` の新フィールド、ハッシュ・arch 文字列、Simple の量子化 CPU forward。YaneuraOu の新入力層クラス・新アーキヘッダ・ビルドフラグ・ハッシュ不一致のエラー化 |
| 6 (検証) | スカラー値の突合と integer-exact 突合は CPU で実施 |

## 6. 未決事項 (hs に確認)

1. 本 ADR と仕様書のリポジトリ内の置き場所。tatara の `CLAUDE.md` は「ADR・設計判断は repo に
   置かない」「doc 冒頭の日付ヘッダ禁止」としており、本 ADR と仕様書 (版の日付・出典表を含む)
   はこれに反する。hs の依頼に従って `docs/` に置いている。代案は PR description への移動、
   または gitignore 済みの `docs/issue/` への移動。
2. D4: 初版を CReLU のみにしてよいか。
3. D8: 入玉ルールをネットに固定し、エンジンの `EnteringKingRule` とは独立に計算してよいか。
4. 駒落ち対応版 (`EKR_27_POINT_H`) を扱わないことでよいか (教師データが平手のみなら影響なし)。
5. 「宣言条件の成立」ビットの定義を「玉が敵陣・敵陣内10枚以上・必要点数以上」の論理積
   (手番・王手・持ち時間を除く) としてよいか。
