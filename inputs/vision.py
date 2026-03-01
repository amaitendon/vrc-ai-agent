import os
import time
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
from langchain_core.tools import tool
from loguru import logger

try:
    import SpoutGL
except ImportError:
    logger.warning("SpoutGL is not installed. Spout vision tool will not work.")
    SpoutGL = None


def _save_debug_image(image_bytes: bytes) -> str | None:
    """
    加工済みの画像（JPEGバイト列）をデバッグ用にファイルとして保存する。
    """
    # SAVE_VISION_LOGS が 0 の場合は保存しない (デフォルトは 1: 有効)
    if os.environ.get("SAVE_VISION_LOGS", "1") == "0":
        return None

    try:
        log_dir = Path(os.environ.get("VISION_LOG_DIR", "logs"))
        log_dir.mkdir(exist_ok=True, parents=True)

        filename = datetime.now().strftime("vision_%Y%m%d_%H%M%S.jpg")
        save_path = log_dir / filename

        with open(save_path, "wb") as f:
            f.write(image_bytes)

        logger.debug(f"[vision] Saved debug image to: {save_path}")
        return str(save_path)
    except Exception as e:
        logger.error(f"[vision] Failed to save debug image: {e}")
        return None


def capture_spout_frame(sender_name: str, timeout_sec: float = 1.0) -> Image.Image | None:
    """
    指定されたSpout送信者からフレームを取得し、Pillow(RGB)の形式で返す。
    取得ごとにSpoutReceiverを生成・解放してシンプルな単発取得を実現する。
    """
    if SpoutGL is None:
        logger.error("SpoutGL module is not available.")
        return None

    try:
        receiver = SpoutGL.SpoutReceiver()
    except Exception as e:
        logger.error(f"Failed to initialize SpoutReceiver: {e}")
        return None

    try:
        receiver.setReceiverName(sender_name)

        # 受信バッファの準備
        info = receiver.getSenderInfo(sender_name)

        # infoがNoneの場合（送信者が見つからない場合）のエラーハンドリング
        if info is None:
            logger.warning(f"Spout sender '{sender_name}' not found (info is None).")
            return None

        width = info.width
        height = info.height

        if width == 0 or height == 0:
            logger.warning(f"Spout sender '{sender_name}' has dimensions 0x0.")
            return None

        # RGBAデータを受け取るNumpy配列
        data = np.zeros((height, width, 4), dtype=np.uint8)

        success = False
        deadline = time.time() + timeout_sec

        while time.time() < deadline:
            receiver.receiveTexture()

            # 新しいフレームが来ているか確認してから画像を受信する
            if receiver.isFrameNew():
                success = receiver.receiveImage(data, SpoutGL.enums.GL_RGBA, True, 0)
                if success and np.any(data > 0):
                    break

            time.sleep(0.05)

        if not success or not np.any(data > 0):
            logger.warning(f"Failed to capture frame from '{sender_name}' within {timeout_sec}s.")
            return None

        # Numpy -> PIL Image (RGBA) -> RGB
        image = Image.fromarray(data, "RGBA")
        return image.convert("RGB")

    except Exception as e:
        logger.error(f"Error capturing Spout frame: {e}")
        return None
    finally:
        # 単発取得のため、必ずリソースを解放する
        receiver.releaseReceiver()


def process_image_for_llm(image: Image.Image, max_width: int = 800) -> str:
    """
    画像をリサイズし、JPEG形式のBase64文字列に変換する。
    LLMへの入力サイズを抑えるための処理。
    """
    # リサイズ（アスペクト比維持）
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_height = int(float(image.height) * float(ratio))
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

    # 環境変数 SPOUT_FLIP_TOP_BOTTOM_IMAGE が 1 の場合のみ上下反転する
    if os.environ.get("SPOUT_FLIP_TOP_BOTTOM_IMAGE") == "1":
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # JPEG形式でメモリバッファに保存
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    jpeg_bytes = buffered.getvalue()

    # デバッグ用に実際にLLMに送られる画像データを保存
    _save_debug_image(jpeg_bytes)

    # Base64エンコード
    img_str = base64.b64encode(jpeg_bytes).decode("utf-8")
    return img_str


@tool
def get_current_view() -> list[dict]:
    """
    Captures the current visual frame from VRChat (or configured Spout sender)
    and returns it as an image for visual analysis.
    Use this when you need to see what is currently happening in the VR environment.
    """
    sender_name = os.environ.get("SPOUT_SENDER_NAME", "VRCSender1")
    logger.debug(f"[vision] Attempting to capture view from Spout sender: {sender_name}")

    image = capture_spout_frame(sender_name)

    if image is None:
        logger.warning(f"[vision] Failed to get view from '{sender_name}'")
        # 失敗時はエラーメッセージをテキストで返却し、エージェントに伝える
        return [
            {
                "type": "text",
                "text": f"Error: Could not capture frame from Spout sender '{sender_name}'. Sender might not be active.",
            }
        ]

    base64_img = process_image_for_llm(image)

    logger.success(f"[vision] Successfully captured and processed view from '{sender_name}'")

    # LiteLLM / Gemini などが解釈できる形式
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
        }
    ]
