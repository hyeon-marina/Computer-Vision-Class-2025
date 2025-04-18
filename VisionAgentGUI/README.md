# Computer-Vision-Class-2025

🧠 2025年の大学コンピュータビジョン授業の実践記録
Practical projects and labs from Computer Vision class (2025)
=======
# 🎥 ビジョンエージェント GUI - PyQt5 × OpenCV

---

## 🧠 概要

このアプリケーションは、**PyQt5とOpenCVを用いて構築された画像・映像処理支援ツール**です。GUIベースで直感的に操作可能であり、主に以下の4つのビジョン機能をサポートしています：

- オリム（前景抽出）
- 交通弱者ゾーン認識（SIFT特徴量マッチング）
- パノラマ合成（Stitcherによる画像連結）
- 特殊効果（油絵風、スケッチ、カートゥーンなど）

本プロジェクトは、**画像処理アルゴリズムの学習・可視化**を目的としており、日本でのAI・CV関連職種を目指すポートフォリオ作品の一つとして制作されました。

---

## ✨ 主な機能

| 機能名 | 概要 |
|--------|------|
| 🖌️ オリム | 画像上に描画 → GrabCutで前景抽出、保存可能 |
| 🚸 交通弱者ゾーン | 標識（子供・高齢者・障がい者）をテンプレートに、SIFT特徴点マッチングで道路画像内の標識を検出 |
| 🖼️ パノラマ合成 | 動画からフレームを収集し、複数画像をパノラマ合成して表示・保存 |
| 🎨 特殊効果 | エンボス、カートゥーン、鉛筆スケッチ（グレー/カラー）、油絵風の画像変換 |

---

## 🧪 開発環境

| ツール | バージョン |
|--------|------------|
| Python | 3.8以上（推奨） |
| PyQt5 | 5.x |
| OpenCV | `opencv-python` + `opencv-contrib-python` |
| NumPy | 1.x |
| OS | Windows（※ `winsound` 使用のため） |

```bash
# 必要なパッケージのインストール
pip install pyqt5 opencv-python opencv-contrib-python numpy
```

---

## 🚀 実行方法

```bash
python VisionAgentGUI.py
```

---

## 📁 ディレクトリ構成

```
VisionAgentGUI/
│
├── visiongui.py         # メインGUIアプリケーション
├── child.png                 # 標識画像：子供
├── elder.png                 # 標識画像：高齢者
├── disabled.png              # 標識画像：障がい者
├── README_ja.md              # 日本語README（本ファイル）
└── requirements.txt          # （必要であれば）依存パッケージ一覧
```

---

## 📌 補足事項

- 交通弱者ゾーン機能には、`child.png`, `elder.png`, `disabled.png` の画像ファイルが必要です。
- スケッチや油絵機能を使用するには `opencv-contrib-python` がインストールされている必要があります。
- `winsound` モジュールを使用しているため、**Windows環境限定での動作**となります。

---

## 💼 作成者

**hyeon-marina**  
- 専攻：人工知能 / コンピュータビジョン  
- 目的：日本のAI系企業での新卒就職  
- GitHub: [github.com/hyeon-marina](https://github.com/hyeon-marina)

---

## 📚 ライセンス

MIT License
>>>>>>> 0f83922 (初期バージョンのアップロード: VisionAgentGUI with README.md)
