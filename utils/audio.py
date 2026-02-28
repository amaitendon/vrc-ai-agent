"""
utils/audio.py

音声・オーディオデバイス関連のユーティリティ関数を提供します。
"""

from loguru import logger
from aiavatar.device.audio import AudioDevice


def get_device_index_by_name(device_name: str, is_input: bool = True) -> int:
    """
    指定されたデバイス名（部分一致）から PyAudio デバイスインデックスを取得します。
    該当するデバイスが見つからない場合は -1 を返します。

    Args:
        device_name (str): 検索するデバイス名の一部（大文字小文字を区別しない）
        is_input (bool): 入力デバイス（マイク等）を探す場合は True、出力デバイス（スピーカー等）を探す場合は False

    Returns:
        int: デバイスが見つかった場合はそのインデックス、見つからなかった場合は -1
    """
    if not device_name:
        return -1

    try:
        # AudioDevice() を使用してシステム上のすべてのオーディオデバイス情報を取得
        audio_dev = AudioDevice()
        devices = audio_dev.get_audio_devices()

        target_name_lower = device_name.lower()

        for d in devices:
            name = d.get("name", "").lower()

            # 入出力の区別
            if is_input:
                if d.get("max_input_channels", 0) == 0:
                    continue
            else:
                if d.get("max_output_channels", 0) == 0:
                    continue

            # 部分一致で検索
            if target_name_lower in name:
                idx = d.get("index", -1)
                logger.info(
                    f"[AudioUtils] Found audio device containing '{device_name}': {d.get('name')} (Index: {idx})"
                )
                return idx

        logger.warning(
            f"[AudioUtils] No audio device found containing '{device_name}' (is_input={is_input})"
        )
        return -1

    except Exception as e:
        logger.error(
            f"[AudioUtils] Error finding audio device by name '{device_name}': {e}"
        )
        return -1


def get_device_name_by_index(device_index: int) -> str:
    """
    指定されたデバイスインデックスからデバイス名を取得します。
    """
    if device_index < 0:
        return "System Default"

    try:
        audio_dev = AudioDevice()
        devices = audio_dev.get_audio_devices()
        for d in devices:
            if d.get("index") == device_index:
                return d.get("name", "Unknown")
    except Exception as e:
        logger.error(
            f"[AudioUtils] Error getting device name for index {device_index}: {e}"
        )

    return "Unknown"
