# VRC AI Agent

VRChat 向けの AI エージェントです。  
音声認識（STT）・LLM（自然言語生成）・音声合成（TTS）・OSC（Open Sound Control）・DB（SQLite）を組み合わせ、VRChat 内でリアルタイムに AI アバターとして動作します。

---

## 動作要件

| 項目 | 要件 |
|------|------|
| OS | Windows 10 / 11 |
| Python | 3.11（[uv](https://docs.astral.sh/uv/) で管理） |
| VRChat | VRChat がログイン済みで起動していること |
| VOICEVOX | [VOICEVOX](https://voicevox.hiroshiba.jp/) が起動していること（デフォルト: `http://127.0.0.1:50021`） |
| 映像入力 | [Spout](https://spout.zeal.co/) 経由で VRChat の映像が送信されていること |
| 音声認識 | [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) を使用（GPU 推奨） |

---

## VRChat AI エージェントができること

AIエージェントは、ユーザーからの音声による呼びかけを受け取って、以下のツールを使って VRChat 内で自律的に行動します。

| ツール | 分類 | できること |
|--------|------|------------|
| `say` | 🗣️ 発話 | VOICEVOX（TTS）を使って音声で発話する |
| `chat_box` | 💬 テキスト | VRChat のチャットボックスにテキストを表示してコミュニケーションを取る |
| `move` | 🚶 移動 | 前後左右に歩いて移動する |
| `rotate` | 🔄 方向転換 | 左右に体を回転させる |
| `jump` | ⬆️ ジャンプ | ジャンプする |
| `get_current_view` | 👁️ 視覚 | Spout 経由で VRChat の現在の視野を画像として取得し、周囲の状況を認識する |
| `remember` | 🧠 記憶 | 重要な出来事・会話・感情を長期記憶（SQLite）に保存する |
| `recall` | 🧠 記憶 | キーワードや話題をもとに過去の記憶を検索して呼び出す |
| `end_action` | ⏸️ 制御 | 現在のアクションサイクルを終了し、ユーザーの応答を待つ |

---

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone <このリポジトリのURL>
cd vrc-ai-agent
```

### 2. aiavatarkit をダウンロードして `lib/` フォルダに配置

このプロジェクトは [aiavatarkit v0.8.9](https://github.com/uezo/aiavatarkit/releases/tag/v0.8.9) をローカルライブラリとして使用しています。  
`lib/` フォルダは `.gitignore` に含まれているため、**クローン後に手動で取得する必要があります**。

```bash
git clone --branch v0.8.9 --depth 1 https://github.com/uezo/aiavatarkit.git lib/aiavatarkit
```

> **注意**: `lib/aiavatarkit` として配置してください。`pyproject.toml` でパスが固定されています。

### 3. 依存パッケージのインストール

[uv](https://docs.astral.sh/uv/) を使用して仮想環境と依存パッケージをインストールします。

```bash
uv sync
```

> **uv がインストールされていない場合** は以下でインストールできます:
> ```powershell
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```

### 4. 環境変数の設定

`.env.example` をコピーして `.env` を作成します。

```bash
copy .env.example .env
```

`.env` を開いて、必要な項目を設定してください。

#### 使用するLLMのモデルとAPIキーの設定

```env
# LLM モデル（LiteLLM 形式で指定）
# 例: LITELLM_MODEL=gemini/gemini-3.1-flash-lite-preview
LITELLM_MODEL= 

# API キー（使用するLLMに応じて設定）
GEMINI_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
```

※LLMモデルの注意点  
`gemini-2.5-flash-lite`等の古いモデルでは、安定して動作しないことがあります。  
（内部で function calling を多用しているため、 安定的に function calling が使えるモデルである必要があります）  
動作確認では、`gemini-3.1-flash-lite-preview`を用いて 安定的に動作することを確認しています。  

#### 音声ルーティング設定

```env
# --- Audio Settings ---
# デバイス名は部分一致で検索されます（例: "Voicemeeter Out A1" など）
# 指定がない（空の）場合は AUDIO_INPUT_DEVICE_INDEX が使用されます（未設定の場合、既定のデバイス）
AUDIO_INPUT_DEVICE_NAME=
AUDIO_INPUT_DEVICE_INDEX=

# 出力側も同様に部分一致で検索されます
AUDIO_OUTPUT_DEVICE_NAME=
AUDIO_OUTPUT_DEVICE_INDEX=
```

以下のコマンドで認識されているデバイス一覧を取得できます。  
使用するデバイスの index または name の値を設定してください。
```bash
uv run ./scripts/list_devices.py
```

※参考：作者の環境では以下の設定で運用しました。  
VRChatの出力デバイス：Voicemeeter Input (VB-Audio Voicemeeter VAIO)  
VRChatの入力デバイス：CABLE Output (VB-Audio Virtual Cable)  
AUDIO_INPUT_DEVICE_NAME="Voicemeeter Out B1"  
AUDIO_OUTPUT_DEVICE_NAME="CABLE Input"  

#### Spout設定の補足
```env
SPOUT_SENDER_NAME=VRCSender1
```

デフォルト設定は`VRCSender1`として、VRChatの`ストリームカメラ`から `Spoutストリーム`を on にするだけで画面キャプチャができるようにしています。
が、視野が狭いので、OBSで作者は画像を取得させました。  


### 5. キャラクタープロンプトの配置

キャラクター設定ファイルを配置してください（`.gitignore` で除外されています）。

```
prompts/charactor.txt
```

ファイルの内容はAIエージェントのキャラクター設定を記述したプレーンテキストです。
自由に編集可能です。

---

## 起動方法

1. VRChat AI エージェントの起動
```bash
uv run main.py
```

2. AIエージェント用のVRChatの起動とログイン  
※私の環境ではVRChatを一つのPCから別アカウントで多重起動できなかったため、メインPCはAIエージェントに明け渡し、Quest 3を使ってAIエージェントと会話していました。  

---

## 設定一覧（`.env`）

各設定項目の詳細は [`.env.example`](./.env.example) を参照してください。  
コメントで各変数の説明・デフォルト値・設定例を記載しています。  

---

## ライセンス
MIT License  

---

## 謝辞
サードパーティライブラリのライセンスは [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md) を参照してください。  

長期記憶の保存については familiar-ai の記憶機能を参考に利用させていただきました。  
https://github.com/susumuota
