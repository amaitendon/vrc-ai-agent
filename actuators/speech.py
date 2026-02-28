"""
actuators/speech.py

TTS（Voicevox等）を用いた発声機能のセットアップを行う。
"""

import asyncio
import io
import os
import wave
from loguru import logger

from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from aiavatar.device.audio import AudioPlayer
from core.context import AppContext
from langchain_core.tools import tool
from utils.audio import get_device_index_by_name


class AudioOutputPipeline:
    """
    音声出力パイプライン設定。
    TTS (Voicevox) -> AudioPlayer の制御基盤。
    """

    def __init__(self):
        # 出力デバイスの解決 (ID優先、次に名前)
        device_index_str = os.getenv("AUDIO_OUTPUT_DEVICE_INDEX", "")
        device_name_str = os.getenv("AUDIO_OUTPUT_DEVICE_NAME", "")

        if device_index_str.isdigit() and int(device_index_str) >= 0:
            self.output_device = int(device_index_str)
        elif device_name_str:
            self.output_device = get_device_index_by_name(
                device_name_str, is_input=False
            )
        else:
            self.output_device = -1

        logger.info(
            f"[AudioOutputPipeline] Initializing with output device index: {self.output_device}"
        )

        # Voicevoxの設定
        voicevox_url = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")
        # TODO: speaker_id を.envから取得できるようにする
        speaker_id = int(os.getenv("VOICEVOX_SPEAKER_ID", "46"))

        logger.info(
            f"[AudioOutputPipeline] Connecting to Voicevox at {voicevox_url} with speaker {speaker_id}"
        )
        self.tts = VoicevoxSpeechSynthesizer(base_url=voicevox_url, speaker=speaker_id)

        # 音声再生プレイヤのセットアップ
        self.player = AudioPlayer(device_index=self.output_device)


# グローバルなインスタンス
_pipeline_instance = None


def get_audio_output_pipeline() -> AudioOutputPipeline:
    """
    AudioOutputPipeline のシングルトンインスタンスを取得する。
    ※現在内部は同期処理のためスレッドセーフだが、将来非同期化（awaitの混入）する場合は
    二重初期化リスクに注意すること。
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AudioOutputPipeline()
    return _pipeline_instance


async def _wait_for_playback(duration_sec: float, player: AudioPlayer) -> None:
    """
    指定秒数待機するタスク。キャンセルされた場合（割り込み時）は再生を停止する。
    """
    try:
        await asyncio.sleep(duration_sec)
    except asyncio.CancelledError:
        try:
            player.stop()
        except Exception as e:
            logger.warning(f"[say] player stop failed: {e}")
        raise  # キャンセル伝播を保証


@tool
async def say(text: str) -> str:
    """
    テキストを音声合成し、ユーザーに対して発話する。
    このツールは直ちに終了し、再生はバックグラウンドで行われる。
    """
    pipeline = get_audio_output_pipeline()

    # 1. TTS合成
    try:
        audio_data = await pipeline.tts.synthesize(text)
    except Exception as e:
        logger.error(f"[say] TTS synthesis failed: {e}")
        return f"failed to synthesize speech: {e}"

    # 2. 再生キューへ投入
    pipeline.player.add(audio_data, has_wave_header=True)

    # 3. 再生時間の推定（WAVヘッダ情報から動的取得）
    try:
        with io.BytesIO(audio_data) as wav_io:
            with wave.open(wav_io, "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duration_sec = frames / float(rate)
    except Exception as e:
        logger.warning(
            f"[say] Failed to parse wave header: {e}. Using fallback duration estimation."
        )
        WAV_HEADER_SIZE = 44
        # Voicevox default fallback: 24kHz, 1ch, 16bit(2 bytes)
        duration_sec = (len(audio_data) - WAV_HEADER_SIZE) / (24000 * 1 * 2)
        if duration_sec < 0:
            duration_sec = 0.0

    # 4. バックグラウンド再生待機タスクを登録
    ctx = AppContext.get()
    ctx.say_task = ctx.spawn_background_task(
        _wait_for_playback(duration_sec, pipeline.player)
    )

    return "say_started"
