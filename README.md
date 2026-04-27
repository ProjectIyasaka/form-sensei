# form-sensei

武道の型を自分で解析し、過去の記録と比較するためのフロントエンド PoC です。
Astro + React で構成されており、`@mediapipe/tasks-vision` の Pose Landmarker を使って骨格推定を行います。

## 機能

- 静止画解析: 画像をアップロードして骨格と関節角度を表示
- 動画解析: 動画をフレーム単位で解析し、骨格と手首の軌跡を表示
- 自己比較: 保存済みの解析結果を 2 件選び、同期再生しながら差分を確認
- ローカル保存: 解析結果のメタデータは `localStorage`、動画 Blob は `IndexedDB` に保存

## 画面

- `/`: ホーム
- `/poc`: 静止画解析
- `/video`: 動画解析
- `/compare`: 保存済み解析の比較

## 技術スタック

- Astro 6
- React 19
- Tailwind CSS 4
- MediaPipe Tasks Vision

## セットアップ

要件:

- Node.js `>=22.12.0`

インストール:

```sh
npm install
```

開発サーバ:

```sh
npm run dev
```

本番ビルド:

```sh
npm run build
```

プレビュー:

```sh
npm run preview
```

## プロジェクト構成

```text
.
├─ public/
├─ scripts/
│  ├─ extract_frames.py
│  └─ train_rtdetr.py
├─ src/
│  ├─ components/
│  │  ├─ PoseAnalyzer.tsx
│  │  ├─ SelfCompare.tsx
│  │  └─ VideoAnalyzer.tsx
│  ├─ lib/
│  │  ├─ storage.ts
│  │  └─ video-analysis.ts
│  ├─ pages/
│  │  ├─ compare.astro
│  │  ├─ index.astro
│  │  ├─ poc.astro
│  │  └─ video.astro
│  └─ styles/
└─ package.json
```

## 実装メモ

- 関節角度は肩、肘、股関節、膝の 8 箇所を算出
- 動画比較ではフレーム補間を使って差分を平滑化
- 比較 UI は横並び表示と重ね表示を切り替え可能
- 保存データはブラウザローカルなので、別ブラウザや別端末には引き継がれません

## 制約

- MediaPipe のモデルと WASM を CDN から読み込むため、初回利用時にネットワーク接続が必要です
- 長い動画や高解像度動画ではブラウザ負荷が高くなります
- 解析結果は推定値であり、武道動作の正確な採点を保証するものではありません

## 補助スクリプト

`scripts/` には、刀検知モデル学習のための補助スクリプトが含まれています。

- `scripts/extract_frames.py`: 動画から学習用フレームを抽出
- `scripts/train_rtdetr.py`: RT-DETR の fine-tuning と ONNX エクスポート

これらは Web アプリ本体とは独立した実験用ワークフローです。
