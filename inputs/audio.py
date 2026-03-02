"""
inputs/audio.py

マイク入力をVADで監視し、発話完了後にSTTで文字起こしして
メインの優先度付きキューへエンキューする処理を提供する。
"""

import asyncio
import os
from datetime import datetime
from loguru import logger

from aiavatar.sts.vad.silero import SileroSpeechDetector
from inputs.stt_faster_whisper import FasterWhisperSpeechRecognizer

from core.context import AppContext, QueueEvent, PRIORITY_VOICE
from utils.audio_device import get_device_index_by_name


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

        logger.info(
            f"[AudioInputPipeline] Initializing with input device index: {self.input_device}"
        )

        # VADのセットアップ (aiavatarkit)
        # ※ローカル版の SileroSpeechDetector は device_index を直接受け取らない仕様
        self.vad = SileroSpeechDetector(
            debug=True,
            silence_duration_threshold=0.7,  # 発話終了と判定する無音区間(秒)。。デフォルト0.5
            min_duration=0.3,  # 短い音は発話として扱わない。デフォルト0.2
            # volume_db_threshold=-40.0,     # 指定した音量(dB)より小さい音は無視する。デフォルトNone
            # speech_probability_threshold=0.7 # 声の判定確率のしきい値。デフォルト0.5
        )

        # STTのセットアップ (Faster-Whisper)
        stt_model_size = os.getenv("STT_MODEL_SIZE", "base")
        self.stt = FasterWhisperSpeechRecognizer(
            model_size=stt_model_size,
            device="cuda" if self._has_cuda() else "cpu",  # 可能な限りGPUを使用
            language="ja",
        )

        # 発話検知時のフックを登録 (メソッド呼び出しで登録)
        self.vad.on_speech_detected(self._on_speech_detected)

        # 話者識別ゲートのセットアップ
        self.speaker_registry = None
        if os.getenv("SPEAKER_GATE_ENABLED", "false").lower() == "true":
            from aiavatar.sts.stt.speaker_registry.base import SpeakerRegistry
            from inputs.speaker_store_sqlite import SQLiteSpeakerStore

            db_path = os.getenv("SPEAKER_GATE_DB_PATH", "data/speakers.db")
            threshold = float(os.getenv("SPEAKER_GATE_THRESHOLD", "0.72"))
            self.speaker_registry = SpeakerRegistry(
                match_threshold=threshold,
                store=SQLiteSpeakerStore(db_path),
            )
            logger.info(
                f"[AudioInputPipeline] SpeakerGate enabled. db={db_path}, threshold={threshold}"
            )

    def _has_cuda(self) -> bool:
        """簡単なCUDA利用可否チェック（精度は必要に応じて調整）"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    async def _on_speech_detected(
        self,
        audio_data: bytes,
        text: str = None,
        metadata: dict = None,
        recorded_duration: float = 0.0,
        session_id: str = "",
    ):
        """
        VADが発話を検知し終わった際に呼ばれるコールバック。
        音声データを受け取り、STTで文字起こしを行ってキューに積む。
        """
        logger.debug(
            f"[AudioInputPipeline] Speech detected. Audio data size: {len(audio_data)} bytes. Transcribing..."
        )

        try:
            speaker_prefix = ""
            # 話者識別ゲートによるフィルタリング
            if self.speaker_registry is not None:
                gate_result = await asyncio.to_thread(
                    self.speaker_registry.match_topk_from_pcm,
                    audio_data,
                    self.vad.sample_rate,
                )
                label = gate_result.chosen.metadata.get(
                    "label", gate_result.chosen.speaker_id
                )
                logger.debug(
                    f"[SpeakerGate] speaker={label}, "
                    f"is_new={gate_result.chosen.is_new}, "
                    f"sim={gate_result.chosen.similarity:.3f}"
                )
                if gate_result.chosen.is_new:
                    name = "Unknown"
                    logger.info(
                        f"[SpeakerGate] Unknown speaker detected: {gate_result.chosen.speaker_id}"
                    )
                else:
                    name = gate_result.chosen.metadata.get("label", "Unknown")
                    logger.info(
                        f"[SpeakerGate] Identified: {label} (sim={gate_result.chosen.similarity:.3f})"
                    )
                speaker_prefix = f"[{name} {gate_result.chosen.speaker_id}]: "

            # STTを実行
            # ※FasterWhisperSpeechRecognizer.transcribe は内部で run_in_executor を使いノンブロッキングに設計されている
            text = await self.stt.transcribe(audio_data)
            text = text.strip()

            if not text:
                logger.debug(
                    "[AudioInputPipeline] Transcription result is empty. Ignoring."
                )
                return

            # タイムスタンプ付与 [HH:MM:SS]
            timestamp = datetime.now().strftime("[%H:%M:%S]")
            if speaker_prefix:
                text = f"{timestamp} {speaker_prefix}{text}"
            else:
                text = f"{timestamp}: {text}"

            logger.info(f"[AudioInputPipeline] Transcription result: {text}")

            interrupted_action = None

            # sayが再生中なら停止（割り込み処理）
            if self.ctx.say_task and not self.ctx.say_task.done():
                logger.info(
                    "[AudioInputPipeline] Cancelling say_task due to voice interrupt by VAD"
                )
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
            logger.error(
                f"[AudioInputPipeline] Error during transcription or queuing: {e}",
                exc_info=True,
            )

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
            resolved_device_index = AudioDevice(
                input_device=self.input_device
            ).input_device
            logger.info(
                f"[AudioInputPipeline] Resolved Input Device Index: {resolved_device_index}"
            )

            # AudioRecorder の初期化
            recorder = AudioRecorder(
                sample_rate=self.vad.sample_rate, device_index=resolved_device_index
            )

            # セッションIDを適当に発行（ユーザーの発話単位）
            session_id = "default_session"

            logger.info("[AudioInputPipeline] Recording stream opening...")
            stream = recorder.start_stream()

            # VAD にストリームを流し込む
            logger.info("[AudioInputPipeline] Streaming to VAD...")
            await self.vad.process_stream(stream, session_id=session_id)

        except Exception as e:
            logger.error(
                f"[AudioInputPipeline] Listening loop failed: {e}", exc_info=True
            )


async def setup_audio_listener(ctx: AppContext):
    """メインループから呼ばれるセットアップ関数"""
    pipeline = AudioInputPipeline(ctx)
    await pipeline.start_listening()
