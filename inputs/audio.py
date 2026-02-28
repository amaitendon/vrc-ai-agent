"""
inputs/audio.py

マイク入力をVADで監視し、発話完了後にSTTで文字起こしして
メインの優先度付きキューへエンキューする処理を提供する。
"""

import os
import asyncio
from loguru import logger

from aiavatar.sts.vad.silero import SileroSpeechDetector
from inputs.stt_faster_whisper import FasterWhisperSpeechRecognizer

from core.context import AppContext, QueueEvent, PRIORITY_VOICE
from utils.audio import get_device_index_by_name


class AudioInputPipeline:
    """
    音声入力パイプライン。
    VAD (Silero) -> STT (Faster-Whisper) -> AppContext.priority_queue
    の流れを管理する。
    """

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        
        # 入力デバイスの解決 (ID優先、次に名前)
        device_index_str = os.getenv("AUDIO_INPUT_DEVICE_INDEX", "")
        device_name_str = os.getenv("AUDIO_INPUT_DEVICE_NAME", "")
        
        if device_index_str.isdigit() and int(device_index_str) >= 0:
            self.input_device = int(device_index_str)
        elif device_name_str:
            self.input_device = get_device_index_by_name(device_name_str, is_input=True)
        else:
            self.input_device = -1

        logger.info(f"[AudioInputPipeline] Initializing with input device index: {self.input_device}")

        # VADのセットアップ (aiavatarkit)
        # ※ローカル版の SileroSpeechDetector は device_index を直接受け取らない仕様
        self.vad = SileroSpeechDetector(debug=True)


        # STTのセットアップ (Faster-Whisper)
        # TODO: モデルサイズ等は必要に応じて.env化する
        self.stt = FasterWhisperSpeechRecognizer(
            model_size="base",
            device="cuda" if self._has_cuda() else "cpu", # 可能な限りGPUを使用
            language="ja"
        )

        # 発話検知時のフックを登録 (メソッド呼び出しで登録)
        self.vad.on_speech_detected(self._on_speech_detected)

    def _has_cuda(self) -> bool:
        """簡単なCUDA利用可否チェック（精度は必要に応じて調整）"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def _on_speech_detected(self, audio_data: bytes, text: str = None, metadata: dict = None, recorded_duration: float = 0.0, session_id: str = ""):
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
            # ローカル版 aiavatarkit に合わせて、AudioRecorder でマイクからストリームを取得し
            # VAD の process_stream に渡す
            from aiavatar.device.audio import AudioRecorder, AudioDevice
            
            # デバイスIDを取得 (-1が指定されている場合はAudioDeviceがデフォルトデバイスを解決する)
            resolved_device_index = AudioDevice(input_device=self.input_device).input_device
            logger.info(f"[AudioInputPipeline] Resolved Input Device Index: {resolved_device_index}")

            # AudioRecorder の初期化
            recorder = AudioRecorder(
                sample_rate=self.vad.sample_rate,
                device_index=resolved_device_index
            )
            
            # セッションIDを適当に発行（ユーザーの発話単位）
            session_id = "default_session"
            
            logger.info("[AudioInputPipeline] Recording stream opening...")
            stream = recorder.start_stream()
            
            # VAD にストリームを流し込む
            logger.info("[AudioInputPipeline] Streaming to VAD...")
            await self.vad.process_stream(stream, session_id=session_id)
            
        except Exception as e:
            logger.error(f"[AudioInputPipeline] Listening loop failed: {e}", exc_info=True)


async def setup_audio_listener(ctx: AppContext):
    """メインループから呼ばれるセットアップ関数"""
    pipeline = AudioInputPipeline(ctx)
    await pipeline.start_listening()
