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
    Move in the specified direction for a given duration.

    Args:
        direction (str): One of 'forward', 'backward', 'left', 'right'.
        duration (float): Duration in seconds. Approximately 1-2m per second on flat ground.
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

        return f"Input held: {direction} for {duration}s. Actual displacement is unknown — use get_current_view to verify."
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
async def rotate(direction: str, duration: float = 0.5) -> str:
    """
    Rotate my body in the specified horizontal direction.

    Args:
        direction (str): One of 'left', 'right'.
        duration (float): Rotation duration in seconds. Rotates about 18 degrees per 0.1s.
    """
    valid_directions = {
        "left": "/input/LookLeft",
        "right": "/input/LookRight",
    }

    if direction not in valid_directions:
        return f"Error: Invalid rotation direction '{direction}'. Must be one of {list(valid_directions.keys())}."

    if duration <= 0:
        return f"Error: duration must be positive, got {duration}."

    osc_address = valid_directions[direction]
    client = OSCClient.get()

    try:
        logger.info(f"[rotate] Rotating {direction} for {duration} seconds")
        client.send_message(osc_address, 1)

        await asyncio.sleep(duration)

        return f"Input held: rotate {direction} for {duration}s. Actual rotation is unknown — use get_current_view to verify."
    except Exception as e:
        logger.exception(f"[rotate] Error rotating {direction}: {e}")
        return f"Error occurred while rotating {direction}: {e}"
    finally:
        # 必ずリリースする
        try:
            client.send_message(osc_address, 0)
        except Exception:
            logger.warning(f"[rotate] Failed to release input for {direction}")
        logger.debug(f"[rotate] Released input for {direction}")


@tool
async def jump() -> str:
    """
    Jump.
    """
    osc_address = "/input/Jump"
    client = OSCClient.get()

    try:
        logger.info("[jump] Executing jump")
        client.send_message(osc_address, 1)

        # ボタンを押して離すまでの短いディレイ
        await asyncio.sleep(0.1)

        return "jumped."
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
