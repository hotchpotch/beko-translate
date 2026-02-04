# 🐈 beko-translate

beko-translate は、Mac の Apple Silicon で動く mlx フレームワークを用いて、翻訳を行う cli アプリケーションです。推論には [mlx-lm](https://github.com/ml-explore/mlx-lm) で最適化しています。

PDF の見開き翻訳コマンドも同梱されており、たとえば論文のページを原文・訳文と交互に読み進めるのに便利です。

![PDF bilingual example](https://github.com/hotchpotch/beko-translate/raw/main/assets/pdf_translated_example.jpg)

## できること

- 日英翻訳
  - モデルによっては多言語翻訳にも対応
- サーバーモードで常駐可能・起動コストを削減
- 対話モード翻訳
- ストリーミング出力 (対話モードはデフォルト ON )
- レイアウトを維持した PDF 翻訳 (見開き翻訳対応)

## インストール

### uv tool（おすすめ）

まず python パッケージマネージャの uv のインストールが必要です。

- https://docs.astral.sh/uv/getting-started/installation/

その後

```bash
uv tool install beko-translate
```

でインストールすると、 `beko-translate` と `beko-translate-pdf` コマンドが使えます。

### 更新方法

```bash
uv tool upgrade beko-translate
```

## 使い方

### 1) cli からの翻訳

```bash
beko-translate --text "こんにちは、可愛い🐈ですね"
# Hello.
```

なお、初回はモデルのダウンロードが行われるため、時間がかかるでしょう。言語を明示したい場合

```bash
beko-translate --text "Hello" --input-lang en --output-lang ja
# こんにちは。
```

なお、標準入力で文章が渡されると、その翻訳文を出力します。

### 2) 対話モード

引数なしで起動すると対話モードになります。

```bash
beko-translate
```

```
>> こんにちは
Hello.
>> exit
```

### 3) ストリーミング出力

```bash
beko-translate --stream --server never --text "こんにちは"
```

ストリーミングはサーバー経由では使えません。`--stream` を付けると直起動に切り替わります。

### 4) サーバーモード

起動コストが気になる場合はサーバーとして立ち上げることで、モデルのロードを省略することができます。

```bash
beko-translate server start
beko-translate --text "こんにちは"
beko-translate server stop
```

サーバーは `~/.config/beko-translate/` に通信用の socket とログを出力します。

- socket: `~/.config/beko-translate/beko-translate.sock`
- log: `~/.config/beko-translate/server.log`

任意の場所を使いたい場合:

```bash
beko-translate server start \
  --socket ~/.config/beko-translate/test.sock \
  --log-file ~/.config/beko-translate/test.log
```

状態確認:

```bash
beko-translate server status
```

### 5) PDF 翻訳

PDF を丸ごと翻訳します。翻訳は beko-translate サーバー経由でできるだけ高速に行います。なお内部では[PDFMathTranslate-next](https://github.com/PDFMathTranslate-next/PDFMathTranslate-next)を利用しています。


```bash
beko-translate-pdf paper.pdf
# 見開き翻訳しない場合
beko-translate-pdf paper.pdf --no-dual
```

デフォルト言語は `--input en --output ja` です。なお、出力ファイル/ディレクトリを指定も可能です。

```bash
beko-translate-pdf paper.pdf --output-pdf translated.pdf
beko-translate-pdf paper.pdf --output-dir ./out/
```

高品質な PLaMo 翻訳モデルも利用可能です。なお PLaMo モデルの利用には、[PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-ja) への同意が必要です。

```bash
uv run beko-translate-pdf paper.pdf \
    --output-dir ./out \
    --model plamo
```

なお、PDF 翻訳には時間がかかります。`mlx-community/plamo-2-translate` を使って、論文を翻訳する場合、ページ数にもよりますが M4 Max で5分〜20分ほどかかります。

### 自動でダウンロードフォルダの pdf を翻訳

[scripts/auto_pdf_translate.py](scripts/auto_pdf_translate.py) に、ブラウザのダウンロードフォルダの pdf で未翻訳のものがあれば、自動で翻訳するスクリプトがあります。

このコマンドに、`auto_pdf_translate.py --arxiv` と `--arxiv` オプションをつけると、arxiv の PDF のようなファイル名だけを翻訳します。そのため、ブラウザで論文をダウンロード、コマンド一発で複数の論文を自動翻訳、のような用途で便利に活用できます。なお、このコマンドの標準の利用モデルは `plamo` となるので、plamo のライセンスが適用されます。


## 翻訳モデルの選択

何も指定しない場合、デフォルトは小型高速な以下の[CAT-Translate](https://huggingface.co/collections/cyberagent/cat-translate)モデルです。

- `hotchpotch/CAT-Translate-0.8b-mlx-q4`

短いエイリアスも用意しています（例: `cat`）。

### おすすめモデル

- **PLaMo 2 Translate（おすすめ）**
  - 論文や技術文書の翻訳が特に良いです。
  - 利用には **PLaMo Community License** への同意が必要です(なお条件付きで商用利用も可能)。
  - ライセンス: [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-ja)
- **CAT-Translate**
  - MIT ライセンスのため、商用利用でも扱いやすいです。
  - なお q8 が 8bit, q4 が 4bit モデルです。

なおこのプロジェクト(beko-translate)のソースコードは MIT ですが、利用する翻訳モデルのライセンスはモデルごとに異なります。利用の際は、必ず各モデルのライセンスを確認してください。

### 動作確認済みの MLX 翻訳モデル

| Model | Hugging Face | License |
| --- | --- | --- |
| [CAT-Translate](https://huggingface.co/collections/cyberagent/cat-translate) | `hotchpotch/CAT-Translate-0.8b-mlx-q4` | MIT |
| [CAT-Translate](https://huggingface.co/collections/cyberagent/cat-translate) | `hotchpotch/CAT-Translate-0.8b-mlx-q8` | MIT |
| [CAT-Translate](https://huggingface.co/collections/cyberagent/cat-translate) | `hotchpotch/CAT-Translate-1.4b-mlx-q4` | MIT |
| [CAT-Translate](https://huggingface.co/collections/cyberagent/cat-translate) | `hotchpotch/CAT-Translate-1.4b-mlx-q8` | MIT |
| [PLaMo 2 Translate](https://huggingface.co/pfnet/plamo-2-translate) | `mlx-community/plamo-2-translate` | [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-ja) |
| [HY-MT 1.5](https://github.com/Tencent-Hunyuan/HY-MT) | `mlx-community/HY-MT1.5-1.8B-4bit` / `mlx-community/HY-MT1.5-1.8B-8bit` / `mlx-community/HY-MT1.5-7B-4bit` / `mlx-community/HY-MT1.5-7B-8bit` | [HY-MT License](https://github.com/Tencent-Hunyuan/HY-MT/blob/main/License.txt) |

## 開発

```bash
uv run tox
```

`tox` は pytest / ruff / ty をまとめて実行します。MLX 統合テストも走ります。

## 注意点

- Apple Silicon (macOS) での使用を想定しています。他の OS は想定しておりません。
- 初回はモデルのダウンロードが走ります。

```bash
uv run --no-sync beko-translate --text "こんにちは"
```

## FAQ

- なんで server は mcp じゃないの？
  - mcp 使わないので、つけることはないと思います

## ライセンス

- ソースコード: MIT

## 謝辞

小型で使いやすそうな MIT ライセンスの翻訳モデル、[CAT Translate](https://huggingface.co/collections/cyberagent/cat-translate)モデルを mac からサクッと使ってみたくて作成しました。プロジェクト名もインスパイアされています。CAT Translate プロジェクトの関係者の方々、ありがとうございます。

また cli は [plamo-translate-cli](https://github.com/pfnet/plamo-translate-cli) が便利だったので、同じように起動しっぱなしで翻訳できる実装を作ってみました。PLaMo は翻訳も cli も便利大変便利です、PLaMo 関連の方々、ありがとうございます。

## Author

- Yuichi Tateno (@hotchpotch)
