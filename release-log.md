# Release Log

## HEAD

- (unreleased)

## 0.0.4

- `beko-translate-pdf --no-dual` 時に `*mono*.pdf` を優先して選ぶよう修正。
- `--no-dual` と通常の `dual` 出力選択を固定する回帰テストを追加。

## 0.0.3

- `torch>=2` を依存関係に追加し、`--model plamo` 初回実行時の `torch` 未導入警告を回避。

## 0.0.2

- 依存関係に numba を追加。

## 0.0.1

- 初期リリース準備: README 整備、モデル一覧/ライセンス表記、例画像の追加。
- MLX 翻訳モデルの抽象化・Hunyuan/Plamo 対応、繰り返し抑止、KV cache などの改善。
- PDF 翻訳の既定動作・サーバー制御周りの整理。
