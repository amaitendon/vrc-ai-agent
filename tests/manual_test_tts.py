"""
tests/manual_test_tts.py

Voicevoxとローカルオーディオ出力デバイス（スピーカ、Voicemeeter等）の
結合を手動で確認するためのスクリプト。
"""

import asyncio
import os
from dotenv import load_dotenv

# プロジェクトルートの.envを読み込むため、親ディレクトリへパスを通す
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actuators.speech import get_audio_output_pipeline


async def main():
    load_dotenv()

    from utils.audio import get_device_name_by_index

    pipeline = get_audio_output_pipeline()
    device_name = get_device_name_by_index(pipeline.output_device)

    print("✅ AudioOutputPipeline generated.")
    print(f"🔊 Output Device Index: {pipeline.output_device} (Name: {device_name})")

    test_text = "音声テストです。マイクテスト、マイクテスト。正しく聞こえていますか？"
    print(f"▶️ Speaking: '{test_text}'")

    await pipeline.speak(test_text)

    # aiavatarkit の AudioPlayer は 別スレッドでエンキューされた音声を再生するため、
    # メインスレッドが即終了しないようにしばらく待つ
    print("⏳ Waiting for playback to complete (10 seconds)...")
    await asyncio.sleep(10)
    print("✅ Done.")


if __name__ == "__main__":
    asyncio.run(main())
