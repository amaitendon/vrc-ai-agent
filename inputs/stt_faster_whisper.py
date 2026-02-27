import asyncio
from typing import List
import numpy as np
from faster_whisper import WhisperModel
from .base import SpeechRecognizer

class FasterWhisperSpeechRecognizer(SpeechRecognizer):
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "default",
        language: str = "ja",
        alternative_languages: List[str] = None,
        *,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        sample_rate: int = 16000,
        debug: bool = False
    ):
        super().__init__(
            language=language,
            alternative_languages=alternative_languages,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            debug=debug
        )
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.sample_rate = sample_rate

    async def transcribe(self, data: bytes) -> str:
        # data is raw 16-bit PCM (from base.py's recognize flow)
        # faster-whisper expects float32 numpy array or path to file
        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Run transcription in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None, 
            lambda: self.model.transcribe(audio_np, language=self.language)
        )
        
        text = "".join([segment.text for segment in segments])
        return text
