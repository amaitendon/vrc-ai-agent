"""
actuators/movement.py

VRChatのアバターの移動・視点・ジャンプを制御するツール群。
python-osc を利用し、UDP経由で直接VRChatと通信する。
"""

import asyncio
from langchain_core.tools import tool
from loguru import logger

from core.osc_client import OSCClient


@tool
async def move(direction: str, duration: float = 1.0) -> str:
    """
    指定した方向にアバターを移動させる。

    Args:
        direction (str): 'forward', 'backward', 'left', 'right' のいずれか
        duration (float): 移動する秒数（デフォルト: 1.0）
    """
    valid_directions = {
        "forward": "/input/MoveForward",
        "backward": "/input/MoveBackward",
        "left": "/input/MoveLeft",
        "right": "/input/MoveRight",
    }

    if direction not in valid_directions:
        return f"Error: Invalid direction '{direction}'. Must be one of {list(valid_directions.keys())}."

    if duration <= 0:
        return f"Error: duration must be positive, got {duration}."

    osc_address = valid_directions[direction]
    client = OSCClient.get()

    try:
        logger.info(f"[move] Starting movement: {direction} for {duration} seconds")
        client.send_message(osc_address, 1)

        await asyncio.sleep(duration)

        return f"Successfully moved {direction} for {duration} seconds."
    except Exception as e:
        logger.exception(f"[move] Error during movement: {e}")
        return f"Error occurred while moving {direction}: {e}"
    finally:
        # 必ずリリースする
        try:
            client.send_message(osc_address, 0)
        except Exception:
            logger.warning(f"[move] Failed to release input for {direction}")
        logger.debug(f"[move] Released input for {direction}")


@tool
async def look_direction(direction: str, duration: float = 1.0) -> str:
    """
    指定した方向にアバターの視点を向ける。

    Args:
        direction (str): 'left', 'right' のいずれか
        duration (float): 視点を動かす秒数（デフォルト: 1.0）
    """
    valid_directions = {
        "left": "/input/LookLeft",
        "right": "/input/LookRight",
    }

    if direction not in valid_directions:
        return f"Error: Invalid look direction '{direction}'. Must be one of {list(valid_directions.keys())}."

    if duration <= 0:
        return f"Error: duration must be positive, got {duration}."

    osc_address = valid_directions[direction]
    client = OSCClient.get()

    try:
        logger.info(f"[look_direction] Looking {direction} for {duration} seconds")
        client.send_message(osc_address, 1)

        await asyncio.sleep(duration)

        return f"Successfully looked {direction} for {duration} seconds."
    except Exception as e:
        logger.exception(f"[look_direction] Error looking {direction}: {e}")
        return f"Error occurred while looking {direction}: {e}"
    finally:
        # 必ずリリースする
        try:
            client.send_message(osc_address, 0)
        except Exception:
            logger.warning(f"[look_direction] Failed to release input for {direction}")
        logger.debug(f"[look_direction] Released input for {direction}")


@tool
async def jump() -> str:
    """
    アバターをジャンプさせる。
    """
    osc_address = "/input/Jump"
    client = OSCClient.get()

    try:
        logger.info("[jump] Executing jump")
        client.send_message(osc_address, 1)

        # ボタンを押して離すまでの短いディレイ
        await asyncio.sleep(0.1)

        return "Successfully jumped."
    except Exception as e:
        logger.exception(f"[jump] Error during jump: {e}")
        return f"Error occurred while jumping: {e}"
    finally:
        # 必ずリリースする
        try:
            client.send_message(osc_address, 0)
        except Exception:
            logger.warning("[jump] Failed to release jump input")
        logger.debug("[jump] Released jump input")
