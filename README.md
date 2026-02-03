# CAT-Translate CLI

Apple Silicon + MLX で日本語/英語の翻訳を回すための小さな CLI です。Hugging Face 上の MLX 量子化モデルを使います。速さ重視なので、常駐サーバーを立てて使うモードも入れました。

## できること

- 日本語 <-> 英語の翻訳（入力を自動判定）
- MLX モデルの自動利用
- 常駐サーバーで起動コストを削減
- 対話モード（引数なしで起動すると REPL）
- ストリーミング出力（対話モードはデフォルト ON）
- PDF 翻訳（pdf2zh_next + cat-translate）

## インストール

依存関係は `uv` で管理しています。

```bash
uv sync
```

## 使い方

### 1) ワンショット翻訳

```bash
uv run cat-translate --text "こんにちは"
```

言語を明示したい場合:

```bash
uv run cat-translate --text "Hello" --input-lang en --output-lang ja
```

### 2) 対話モード

引数なしで起動すると対話モードになります。

```bash
uv run cat-translate
```

```
>> こんにちは
Hello.
>> exit
```

### 3) ストリーミング出力

```bash
uv run cat-translate --stream --server never --text "こんにちは"
```

ストリーミングはサーバー経由では使えません。`--stream` を付けると自動的に直起動に切り替わります。

### 4) サーバーモード

起動コストが気になる場合はサーバーを起動して使ってください。

```bash
uv run cat-translate server start
uv run cat-translate --text "こんにちは"
uv run cat-translate server stop
```

サーバーは `~/.config/cat-translate/` にソケットとログを作ります。

- socket: `~/.config/cat-translate/cat-translate.sock`
- log: `~/.config/cat-translate/server.log`

任意の場所を使いたい場合:

```bash
uv run cat-translate server start \
  --socket ~/.config/cat-translate/test.sock \
  --log-file ~/.config/cat-translate/test.log
```

状態確認:

```bash
uv run cat-translate server status
```

### 5) PDF 翻訳

pdf2zh_next を使って PDF を丸ごと翻訳します。翻訳は cat-translate サーバー経由です。

```bash
uv run cat-translate-pdf paper.pdf
```

デフォルトは `--input en --output ja` です。自動判定したい場合:

```bash
uv run cat-translate-pdf paper.pdf --input auto
```

和英:

```bash
uv run cat-translate-pdf paper_ja.pdf --input ja --output en
```

出力ファイル/ディレクトリを指定:

```bash
uv run cat-translate-pdf paper.pdf --output-pdf translated.pdf
uv run cat-translate-pdf paper.pdf --output-dir ./out
```

サーバーは 1.4b q8 がデフォルトで、別モデルが動いていたら自動的に停止して起動し直します。

## モデル

デフォルトは以下です。

- `hotchpotch/CAT-Translate-0.8b-mlx-q4`

他のモデルも指定できます。

- `hotchpotch/CAT-Translate-0.8b-mlx-q8`
- `hotchpotch/CAT-Translate-1.4b-mlx-q4`
- `hotchpotch/CAT-Translate-1.4b-mlx-q8`

例:

```bash
uv run cat-translate --model hotchpotch/CAT-Translate-1.4b-mlx-q8 --text "こんにちは"
```

PDF 翻訳 (`cat-translate-pdf`) のデフォルトは以下です。

- `hotchpotch/CAT-Translate-1.4b-mlx-q8`

## オプション

主要なものだけ。

- `--input-lang` / `--output-lang` : en / ja
- `--max-new-tokens` : 既定 4096
- `--temperature` / `--top-p` / `--top-k` : サンプリング設定
- `--server` : `auto` / `always` / `never`
- `--socket` / `--log-file` : サーバー用
- `--verbose` : どの経路を使ったか表示

## 開発

```bash
uv run tox
```

`tox` は pytest / ruff / ty をまとめて実行します。MLX 統合テストも走ります。

## 注意点

- Apple Silicon (macOS) での使用を想定しています。
- 初回はモデルのダウンロードが走ります。
- `uv run` は毎回同期するので、実行前にインストールログが出ます。気になる場合は `--no-sync` を使ってください。

```bash
uv run --no-sync cat-translate --text "こんにちは"
```
