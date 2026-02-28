"""
actuators/speech.py

TTS（Voicevox等）を用いた発声機能のセットアップを行う。
"""

import os
from loguru import logger

from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from aiavatar.device.audio import AudioPlayer
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
            self.output_device = get_device_index_by_name(device_name_str, is_input=False)
        else:
            self.output_device = -1

        logger.info(f"[AudioOutputPipeline] Initializing with output device index: {self.output_device}")

        # Voicevoxの設定
        voicevox_url = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")
        # TODO: speaker_id を.envから取得できるようにする
        speaker_id = int(os.getenv("VOICEVOX_SPEAKER_ID", "46")) 
        
        logger.info(f"[AudioOutputPipeline] Connecting to Voicevox at {voicevox_url} with speaker {speaker_id}")
        self.tts = VoicevoxSpeechSynthesizer(
            base_url=voicevox_url,
            speaker=speaker_id
        )

        # 音声再生プレイヤのセットアップ
        self.player = AudioPlayer(device_index=self.output_device)

    async def speak(self, text: str):
        """
        テキストを音声に合成し、再生する（Step 4以降のアクションから呼ばれる）
        """
        try:
            # TTSでWAVバイナリを生成
            logger.debug(f"[AudioOutputPipeline] Synthesizing speech for: '{text}'")
            audio_data = await self.tts.synthesize(text)
            
            # 再生
            logger.debug(f"[AudioOutputPipeline] Playing back synthesized audio ({len(audio_data)} bytes)...")
            # aiavatarkitの実装ではAudioPlayer.add(bytes)でキューに追加し勝手に別スレッドで再生する
            self.player.add(audio_data, has_wave_header=True)
            logger.debug("[AudioOutputPipeline] Playback enqueued.")

        except Exception as e:
            logger.error(f"[AudioOutputPipeline] Error during speech synthesis or playback: {e}", exc_info=True)


# グローバルなインスタンス（またはAppContext経由での管理）
# PHASE1 の Step4 で実際に `say` ツールを作るときに利用する想定のスタブ関数
_pipeline_instance = None

def get_audio_output_pipeline() -> AudioOutputPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AudioOutputPipeline()
    return _pipeline_instance
