"""
inputs/audio.py

マイク入力をVADで監視し、発話完了後にSTTで文字起こしして
メインの優先度付きキューへエンキューする処理を提供する。
"""

import asyncio
import os
from loguru import logger

from aiavatar.device.audio import AudioDevice
from aiavatar.sts.vad.silero import SileroSpeechDetector
from inputs.stt_faster_whisper import FasterWhisperSpeechRecognizer

from agent.core.context import AppContext, QueueEvent, PRIORITY_VOICE


class AudioInputPipeline:
    """
    音声入力パイプライン。
    VAD (Silero) -> STT (Faster-Whisper) -> AppContext.priority_queue
    の流れを管理する。
    """

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        
        # デバイスIDの取得（.envで指定されていなければデフォルトデバイスを使用）
        device_index_str = os.getenv("AUDIO_INPUT_DEVICE_INDEX", "")
        self.input_device = int(device_index_str) if device_index_str.isdigit() else -1

        logger.info(f"[AudioInputPipeline] Initializing with input device index: {self.input_device}")

        # VADのセットアップ (aiavatarkit)
        self.vad = SileroSpeechDetector(device_index=self.input_device)

        # STTのセットアップ (Faster-Whisper)
        # TODO: モデルサイズ等は必要に応じて.env化する
        self.stt = FasterWhisperSpeechRecognizer(
            model_size="base",
            device="cuda" if self._has_cuda() else "cpu", # 可能な限りGPUを使用
            language="ja"
        )

        # 発話検知時のフックを登録
        self.vad.on_speech_detected = self._on_speech_detected

    def _has_cuda(self) -> bool:
        """簡単なCUDA利用可否チェック（精度は必要に応じて調整）"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def _on_speech_detected(self, audio_data: bytes):
        """
        VADが発話を検知し終わった際に呼ばれるコールバック。
        音声データを受け取り、STTで文字起こしを行ってキューに積む。
        """
        logger.debug(f"[AudioInputPipeline] Speech detected. Audio data size: {len(audio_data)} bytes. Transcribing...")
        
        try:
            # STTを実行
            # ※FasterWhisperSpeechRecognizer.transcribe は内部で run_in_executor を使いノンブロッキングに設計されている
            text = await self.stt.transcribe(audio_data)
            text = text.strip()
            
            if not text:
                logger.debug("[AudioInputPipeline] Transcription result is empty. Ignoring.")
                return

            logger.info(f"[AudioInputPipeline] Transcription result: {text}")

            interrupted_action = None

            # sayが再生中なら停止（割り込み処理）
            if self.ctx.say_task and not self.ctx.say_task.done():
                logger.info("[AudioInputPipeline] Cancelling say_task due to voice interrupt by VAD")
                self.ctx.say_task.cancel()
                interrupted_action = "say"

            # キューに積む
            event = QueueEvent(
                priority=PRIORITY_VOICE,
                text=text,
                interrupted_action=interrupted_action,
            )
            await self.ctx.priority_queue.put(event)
            logger.debug(f"[AudioInputPipeline] Queued voice event: {event}")

        except Exception as e:
            logger.error(f"[AudioInputPipeline] Error during transcription or queuing: {e}", exc_info=True)

    async def start_listening(self):
        """
        常時監視の開始。
        VADの音声待機ループを回す。
        """
        logger.info("[AudioInputPipeline] Starting VAD listening loop...")
        try:
            # SileroSpeechDetector の detect() は同期的なジェネレータまたはブロック関数の可能性があるため、
            # run_in_executorを用いて非同期呼び出しする (aiavatarkitの設計に応じて調整)
            # ※ aiavatarkitの実装では async def や async for 等で提供されているか要確認
            await self.vad.start() 
        except Exception as e:
            logger.error(f"[AudioInputPipeline] Listening loop failed: {e}", exc_info=True)


async def setup_audio_listener(ctx: AppContext):
    """メインループから呼ばれるセットアップ関数"""
    pipeline = AudioInputPipeline(ctx)
    await pipeline.start_listening()
