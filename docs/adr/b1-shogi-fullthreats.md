# ADR: B1 将棋版 FullThreats（HalfKP＋利き関係特徴）の実装方針

- 状態: 提案（調査完了・実装未着手。GPU 工程は未検証）
- 対象: [docs/nnue-architecture-ideas.md](../nnue-architecture-ideas.md) §4 B1、§6 の B1 関連未決事項
- 対象リポジトリ: tatara（学習器）と YaneuraOu（エンジン）。どちらも `yjbg9201` のフォーク

## 1. 結論（要約）

1. **学習器は既存の threat 実装を流用する**。tatara には将棋向けの利き関係特徴（`--threat-profile`）が
   既にあり、仕様書 B1 の特徴設計（攻撃駒・被攻撃駒・マスの組、金類の集約、冗長ペアの機械的除去）を
   ほぼそのまま満たしている。新しい index 体系は作らない。
2. **ネットは Simple（HalfKP 256x2-32-32 CReLU）＋threat で作る**。threat は今 LayerStack 専用なので、
   Simple に開放する。Simple の `.bin` はやねうら王の通常 NNUE と同じ形式であることを実測で確認した（§3.3）。
3. **初期スコープは全駒種**（profile `full-symdedup`：全駒種＋冗長な相互利きの片側を落とす）。
   縮小版は既存 profile（`step-attacker` など）を先に試し、必要なら「大駒・金銀限定」profile を追加する。
4. **エンジンは「threat 部分だけ毎手全計算」で始める**。やねうら王の特徴量セットは refresh trigger ごとに
   別アキュムレータを持つため、threat 特徴の trigger を `kAnyPieceMoved` にすれば変換器本体を変えずに
   全リフレッシュ版が作れる。差分更新（dirty threats）は固定ノードの Elo 上限を測ってから着手する。

## 2. 背景

仕様書 B1 は「駒 A が駒 B に利いている」関係を 1 特徴とし、HalfKP の FT と同じ幅の別 FT を活性化前に
加算する構成を提案している。§6 には B1 向けの未決事項として次の 2 つが残っていた。

- B1 の初期スコープ（全駒種か、大駒・金銀限定か）
- tatara の threat 拡張が将棋の利き（飛び駒 5 種・打ち駒）にどこまで対応しているか

本 ADR はこの 2 点に答え、実装の段階分けを決める。

## 3. 調査結果

### 3.1 tatara の threat 実装（`crates/shogi-features/src/threat*.rs`）

| 項目 | 実装 |
|---|---|
| 特徴の単位 | (視点, 攻撃側の味方/敵, 攻撃駒 class, 被攻撃側の味方/敵, 被攻撃駒 class, from, to) |
| 駒種 class | 9 種。金・と・成香・成桂・成銀は `GoldLike` に集約（攻撃側・被攻撃側の両方）。**玉は攻撃側・被攻撃側のどちらにも入らない** |
| 利きの計算 | 盤面配列上の ray-walk。飛び駒 5 種（香・角・飛・馬・竜）は占有マスで止まり、止めた駒自身は攻撃対象になる。玉も遮蔽物として扱う |
| 持ち駒・打ち駒 | 持ち駒は対象外（盤上に利きがない）。学習側は局面ごとに全列挙するので、打ち駒による遮りは自然に反映される |
| index | `pair_base + from_offset + attack_order`。空盤面の利きから作った表で詰めて並べる（`threat.rs` 冒頭の式） |
| マスの正規化 | 視点で 180° 回転した後、**自玉が 6〜9 筋にあれば左右反転**（HalfKA_hm と同じ半面ミラー） |
| 1 局面の最大 active 数 | 学習側の上限 `THREAT_MAX_ACTIVE = 128`（視点あたり）。コメントによると約 9 億局面の実測最大は 105。超えるとデータローダが停止する |
| 冗長ペア除去 | `full-symdedup`：逆向きの利きが必ず同時に成立する組（空盤面の利き表で機械判定）のうち、class の小さい側を emit 時に落とす。同 class 同士は落とさない（半面ミラーで「残す側」が一意にならないため）。index 空間は `full` と同じで、落とした側の行は常に非 active |
| factorizer | threat の pair ごとに 1 本の仮想行（「駒種ペアのみ」に相当）。LayerStack で使用 |
| ハッシュ | `feature_hash = base ^ fnv1a32("threat-<profile>")` |

profile ごとの threat 次元（`threat_dimensions_of`）:

| profile | 除外規則 | threat 次元 |
|---|---|---|
| `full` | なし | 216,720 |
| `full-symdedup` | なし（冗長な片側を emit 時に落とす） | 216,720 |
| `same-class` | 同 class の組 | 192,640 |
| `same-class-major-pawn` | 同 class、大駒→歩 | 173,568 |
| `cross-side` | 同じ側同士、同 class | 96,320 |
| `step-attacker` | 飛び駒の攻撃（香・角・飛・馬・竜） | 33,408 |

同梱の `crates/shogi-format/tests/data/sample.psv`（100 局面）で `threat-symmetric-survey` を実行すると、
1 局面あたりの active な利き関係は平均 28.3 本、`full-symdedup` で落ちるのはそのうち 7.7%
（大半が「銀→金類」「金類→馬」「歩→金類」）だった。100 局面なので分布の目安にとどまる。

### 3.2 tatara で threat が使える範囲

- `--threat-profile` は `layerstack` サブコマンドの引数で、`simple` には無い。
- やねうら王形式への書き出し（`--output-format yaneuraou`、`net_to_yo`）は threat 付きネットを明示的に拒否する。
  LayerStack＋threat の推論先は今のところ rshogi だけ。
- Simple の GPU 学習経路（`bins/nnue_train/src/trainer_simple.rs` と native カーネル）は入力次元 `ft_in` と
  1 局面の最大特徴数 `max_active` を実行時引数で受けており、threat 行の連結は feature set の spec 経由で
  入る作り。factorizer の threat 仮想行用テーブル（`threat_pair_starts`）も既に Simple へ渡されている
  （今は threat 行が無いので番兵 1 要素だけ）。
- Simple の arch 文字列には threat を表すトークンが無い。threat 付き spec で組むと
  `Features=HalfKP(Friend)[342108->256x2]` となり、`full` と `full-symdedup` が同じ文字列になる（ハッシュは異なる）。
- `--init-from` は identity（feature set・次元）が完全一致する `.bin` しか読めない。HalfKP の `.bin` から
  HalfKP＋threat へ移植する経路は無い。
- 学習は GPU 必須。`--no-default-features` は CI 用の CPU テスト部分だけをビルドし、学習は実行できない。
  CPU で量子化 forward を計算できるのは LayerStack 用の `eval_sfen`（threat 対応）だけで、Simple 用は無い。

### 3.3 Simple 形式とやねうら王の互換性（実測）

クラウド環境（GPU なし）で次を確認した。

1. tatara の `nnue-format` で乱数重みの HalfKP 256x2-32-32 CReLU ネットを `SimpleWeights::save_quantised` で書き出した。
2. やねうら王を `make normal TARGET_CPU=AVX2 YANEURAOU_ENGINE_NNUE COMPILER=clang++` でビルドし（約 1 分）、
   その `nn.bin` を読ませた。**ハッシュ不一致の警告は出ず、ロードは成功した**。
3. tatara の `FeatureSet::HalfKp` が出す index と `.bin` から読み戻した整数重みで、やねうら王と同じ整数演算の
   forward を書いて評価値を計算し、やねうら王の `e` コマンド（`evaluate()` の生値）と比べた。
   平手・先手番/後手番・持ち駒あり・成駒ありの 5 局面（`source/benchmark.cpp` の局面を含む）で
   **全局面が整数単位で一致**した（例: 平手 −401、他 −211 / −452 / −432 / −552）。

検証用の一時コードは PR 本文に添付した（リポジトリには入れていない）。この結果から、Simple＋threat の
`.bin` は「FT の行が増えるだけ」でやねうら王の通常 NNUE と同じ形式にできる。

### 3.4 やねうら王の NNUE 実装（`source/eval/nnue/`）

| 項目 | 実装 | B1 への影響 |
|---|---|---|
| 特徴量の合成 | `FeatureSet<A, B>` は後ろの `B` が低位の index 空間 | tatara の行順 `[HalfKP | threat]` に合わせるには `FeatureSet<Threat, HalfKP<kFriend>>` と宣言する |
| アキュムレータ | refresh trigger ごとに別の部分和 `accumulation[視点][trigger][次元]` を持ち、変換時に合算する | threat 特徴を `kAnyPieceMoved` にすると、HalfKP 部分は従来の差分更新のまま、threat 部分だけ毎手全計算になる。変換器本体の変更は不要 |
| 差分情報 | `DirtyPiece` は動いた駒と取られた駒の最大 2 個だけ | 利き関係の差分（開き利き・打ち駒による遮り）は自前で計算する必要がある。差分更新版は別段階にする |
| index リストの容量 | `IndexList` は `RawFeatures::kMaxActiveDimensions` 固定長 | 超えるとメモリ破壊になる。threat の上限は理論値で取る（§4 D5） |
| Finny テーブル | `-DUSE_FINNY_TABLES`（＋`-DUSE_FINNY_RESET`）で、trigger ごとのキャッシュとの差分で refresh する。`kAnyPieceMoved` はキャッシュ 1 枠 | 全計算版でも、キャッシュとの差分だけ加減算できる可能性がある。並び順がずれると差分が膨らむので `-DNNUE_FINNY_CANCEL` 併用で測る |
| ハッシュ検査 | 不一致は警告のみでロード続行。サイズ違いはファイル末尾の検査で失敗する | `full` と `full-symdedup` はサイズが同じなので、取り違えを弾けない |
| 検査コマンド | `test nn test_features`（差分 index と全計算の一致）、`test nn test_accumulator`（差分・キャッシュ経由と全再計算の一致）、`e`（生の評価値） | 検証プロトコル §1.2 の 1・3 にそのまま使える |
| 利きの計算 | `effects_from(Piece, Square, occupied)` で駒ごとの利き Bitboard が取れる | 全計算版の index 列挙はこれと盤上の駒 Bitboard の積で書ける |

### 3.5 仕様書との差異

| 仕様書の記述 | 実ソース | 扱い |
|---|---|---|
| 玉非依存で開始（視点の 180° 回転のみ）、左右ミラーは第 2 段階 | tatara の threat index は自玉の筋で半面ミラーする | 実ソースを優先（D2）。エンジンの差分更新版では、自玉がミラー境界をまたいだときに threat 部分を全計算する |
| 次元は圧縮後 10 万〜20 万程度 | `full` / `full-symdedup` は 216,720 | 実ソースの値を使う |
| 被攻撃側を集約するかは実験で決める | 攻撃側・被攻撃側とも `GoldLike` に集約済み | 既存の集約を使う |
| 冗長ペアを機械的に全列挙して削る | 機械判定はあるが、index 空間からは削らず emit 時に落とす。同 class の組は残す | 既存の `full-symdedup` を使う。落とした側の行は常に 0 勾配で、`.bin` には残る |
| PSQ の FT と同じ幅の別 FT、重みは int8 | 同じ FT 行列の後ろに threat 行を連結し、重みは int16 | 活性化前の加算と数学的に同じなので連結のまま進める（D4）。int8 化は後続の最適化 |
| factorizer は「駒種ペアのみ」「相対位置のみ」 | 「駒種ペアのみ」に相当する pair 仮想行だけある | 既存のものを使い、「相対位置のみ」は保留 |
| ハッシュを新規採番して誤ロードをロード時に弾く | やねうら王はハッシュ不一致を警告だけで通す | B1 のビルドでは不一致を拒否する（D6） |
| 初期化は `nnue_surgery.py` の `ft-transplant` 相当 | このツールは両リポジトリに無い | tatara に移植経路を追加する（D7） |
| §3「threat 拡張・factorizer は A3・B1 の起点にできる可能性が高い」 | threat は LayerStack 専用、Simple・やねうら王形式の書き出しは非対応 | Simple に開放する（D1） |
| §7.1「CPU で数ステップ学習して loss が減るまで確認」 | tatara に CPU 学習モードは無い | 学習の確認は GPU 環境が整ってから |

## 4. 決定

### D1. 学習経路は Simple＋threat

`simple` サブコマンドに `--threat-profile` を開放し、HalfKP 256x2-32-32 CReLU に threat 行を連結する。

- 理由: 仕様書 B1 は HalfKP 前提で、比較用ベースライン（既存 HalfKP ネット）からの移植を初期化方法にしている。
  Simple の `.bin` はやねうら王の通常 NNUE と互換であることを実測した（§3.3）。
- LayerStack＋threat を SFNN に書き出す経路は、やねうら王の SFNN 側にも threat を入れる必要があり範囲が広い。
  B1 の効果を確認してからの別段階とする。

### D2. 特徴定義は tatara の既存 threat index を正とする

index 式・class 集約・半面ミラー・冗長除去の規則は `crates/shogi-features/src/threat*.rs` を唯一の定義とし、
やねうら王側はこれを移植する。既存の profile の挙動は変えない（bit-identical）。

### D3. 初期スコープは `full-symdedup`（全駒種）

- 最初の測定は固定ノードの Elo 上限（仕様書 B1「計測順」2）なので、情報量が最大の全駒種で測る。
- `full` と情報量は同じで active 数だけ少ない。冗長判定は空盤面の利き表の参照だけなのでエンジンでも安い。
- 撤退基準（固定ノード +30 Elo 程度で NPS が 15% 以上落ちる等）に当たったら、既存の `step-attacker`
  （飛び駒を攻撃側から外す。33,408 次元で、エンジンは飛び駒の ray 列挙を省ける）→ `cross-side` の順に試す。
  仕様書の「大駒・金銀限定」はどの既存 profile とも一致しないので、必要になった時点で新しい profile として足す。

### D4. FT は 1 本のまま threat 行を後ろに連結する（int16）

- FT 入力は 125,388（HalfKP）＋216,720（threat）＝ 342,108 行。重みは int16 のまま。
- `.bin` は約 167 MiB（HalfKP 単体は約 61 MiB）。threat 部分だけ int8 にすれば約 53 MiB 減るが、
  学習・書き出し・推論の 3 か所に量子化の分岐が増えるので初版では行わない。

### D5. エンジンは全計算版から始める

- 特徴量 `Threat`（仮称）を `source/eval/nnue/features/` に追加し、`kRefreshTrigger = kAnyPieceMoved` とする。
- アーキテクチャヘッダ（例: `halfkp_threat_256x2-32-32.h`）で `FeatureSet<Threat, HalfKP<Side::kFriend>>` を宣言し、
  Makefile の edition で既存アーキと切り替える。
- `kMaxActiveDimensions` は視点あたりの**理論上限 236** とする。盤上の玉以外の駒は最大 38 枚で、1 枚が持つ
  利きの方向は最大 8（馬・竜）、金類 6、銀 5 なので、全駒が成駒・馬・竜でも
  18×6 + 4×6 + 4×6 + 4×6 + 4×6 + 2×8 + 2×8 = 236 を超えない（飛び駒は 1 方向あたり最初の 1 駒にしか当たらない）。
  学習側の 128 は教師データ用の上限で、探索中の局面には使えない。
- 利きの列挙は `effects_from(pc, sq, pieces())` と「玉以外の盤上の駒」の積で行う。占有には玉を含める（遮蔽物）。
- NPS は次の 3 通りで測る: 素の全計算 / `-DUSE_LAZY_EVALUATE -DUSE_FINNY_TABLES -DUSE_FINNY_RESET` /
  それに `-DNNUE_FINNY_CANCEL` を足したもの。
- 差分更新（dirty threats）は、固定ノードで効果が確認できてから別段階で実装する。

### D6. ハッシュと arch 文字列

- feature hash は tatara の値（`base ^ fnv1a32("threat-<profile>")`）を正とする。やねうら王の `FeatureSet` は
  `Head ^ (Tail << 1) ^ (Tail >> 31)` で合成するので、`Threat::kHashValue` はそこから逆算した定数を置く。
  例: HalfKP（`0x5D69D5B8`）＋`full-symdedup` では tatara の feature hash が `0x9CD066E9`、
  `Threat::kHashValue` は `0x2603CD99`。値は両リポのテストで固定する。
- tatara の Simple の arch 文字列に LayerStack と同じ `Threat=<dims>,ThreatProfile=<id>,` トークンを入れる。
  ストリームには LayerStack のような profile id の u32 を**書かない**（やねうら王の通常 NNUE の読み込み順を保つため）。
- B1 のビルドでは、やねうら王のハッシュ不一致を警告ではなくロード失敗にする（既存アーキの挙動は変えない）。

### D7. 既存 HalfKP ネットからの移植は tatara に追加する

HalfKP の Simple `.bin` を読み、HalfKP 行・後段はそのまま、threat 行は 0＋微小ノイズで初期化する経路を
明示的な別オプションとして足す（`--init-from` の意味は変えない）。移植直後の量子化出力が元ネットと一致
すること（threat 行が 0 のとき）をテストで確認する。threat 行の学習率を一時的に上げる等の扱い（仕様書 §1.4）は
学習段階で決める。

### D8. 検証の道具

| 仕様書 §1.2 | 道具 |
|---|---|
| 1. index 整合 | tatara: SFEN から threat index を出す example（`crates/shogi-features/examples/` の既存 dump と同じ形）。やねうら王: 一時的な index dump。ランダム対局の 1 万局面で突合 |
| 2. integer-exact | tatara に Simple の CPU 量子化 forward を追加し（§3.3 の検証コードを正式化）、やねうら王の `e` と比較 |
| 3. 差分更新整合 | やねうら王の `test nn test_features` / `test nn test_accumulator` |
| 4. 棋力 | 固定ノード（Elo 上限）→ 固定時間（NPS 込み）。対局は手元環境 |

仕様書 B1「計測順」1 の脅威特徴数の分布は、教師データに対して `threat-symmetric-survey`、
1 手あたりの変化数は `test nn test_features` の出力で取る。

## 5. 段階分け（各段階を独立 PR にする。T＝tatara、Y＝やねうら王）

| 段階 | 内容 | GPU |
|---|---|---|
| 1 | 調査と本 ADR（この PR） | 不要 |
| 2 | T: Simple に `--threat-profile` を開放。arch 文字列のトークン、保存・読み込み、CLI テスト、byte-golden テスト | CPU 部分は不要。学習 smoke・GPU/CPU 同等性は保留 |
| 3 | T: Simple の CPU 量子化 forward と threat index の dump example | 不要 |
| 4 | Y: `Threat` 特徴量（全計算版）、アーキテクチャヘッダ、ビルド切り替え、ハッシュ不一致の拒否 | 不要 |
| 5 | T・Y: index 突合（1 万局面）、integer-exact 突合、`test nn test_features` / `test_accumulator` | 不要 |
| 6 | T: HalfKP からの移植オプション | 移植直後の一致は CPU で確認可。学習は GPU |
| 7 | 学習と測定（固定ノード → 固定時間、NPS 3 通り） | 必要（手元） |
| 後続 | 差分更新（dirty threats）、threat 重みの int8 化、LayerStack/SFNN＋threat、縮小 profile | — |

GPU が要る項目（段階 2 の学習 smoke と GPU/CPU 同等性、段階 6・7 の学習）は draft PR のチェックリストに残し、
GPU 環境が整ってから消化する。

## 6. リスクと未解決事項

- **学習時の GPU メモリ**: FT が 342,108 行になり、float の重み・勾配・Adam のモーメントで 1 GiB を超える
  （factorizer の fold バッファも加わる）。手元の Quadro T1000（4 GB）では batch を下げるか
  `--fp16-opt-state` が要る可能性がある。
- **NPS**: 全計算版は threat 部分で毎手、視点あたり 20〜40 行程度（局面依存）の加算と利き列挙が増える。
  HalfKP の差分（視点あたり 1 手 2〜4 行）より重く、固定時間での効果は固定ノードより小さく出る見込み。
- **Simple の factorizer と threat の組み合わせ**: LayerStack では検証済みだが Simple では未検証。
- **`.bin` が大きい**: やねうら王は評価関数を共有メモリに置くので、多数プロセスの対局で問題にならないか確認する。
- **upstream との関係**: upstream（SH11235/tatara）の CLAUDE.md は ADR や作業計画文書をリポジトリに置かない
  規約になっている。本 ADR と仕様書はフォーク内の作業用として置いており、upstream 向けの PR には含めない。
