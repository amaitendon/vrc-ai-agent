import pytest
import os

# PyTorch (torch.cuda) への依存をモックするために、テストでは必ず "cpu" で初期化するように強制する
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from inputs.stt_faster_whisper import FasterWhisperSpeechRecognizer


@pytest.mark.asyncio
async def test_faster_whisper_transcribe_empty_audio():
    """
    ダミーのPCM(WAV)データを渡し、エラー無くSTT処理(transcribe)を通過できるかをテストする。
    """
    # compute_type="int8" で極力軽量に初期化（本来はモックでモデルロードを回避すべきだが簡易検証とする）
    # 今回のテストで実際に推論させると重いため、テスト規模に応じてモック化の適用が必要
    # ここではインスタンス化できるかと引数処理のみをざっくり確認
    recognizer = FasterWhisperSpeechRecognizer(
        model_size="tiny", device="cpu", compute_type="int8", language="ja"
    )

    # 1秒間・16kHz・16ビットPCM の無音データ
    empty_audio_data = b"\x00" * (16000 * 2)

    text = await recognizer.transcribe(empty_audio_data)

    # 無音データなので文字列は空か、無意味な空白等が返ることを想定
    assert isinstance(text, str)
