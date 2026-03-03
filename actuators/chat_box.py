"""
actuators/chat_box.py

VRChatのチャットボックスを制御するツール。
python-osc を利用し、UDP経由で直接VRChatと通信する。
"""

from langchain_core.tools import tool
from loguru import logger

from core.osc_client import OSCClient


@tool
async def chat_box(message: str) -> str:
    """
    Send a message to the VRChat chat box (visible to nearby users as on-screen text).
    Keep messages within 144 characters. Supports emojis and Japanese text.
    """
    try:
        logger.debug(f"[chat_box] Sending message via OSC: '{message}'")
        client = OSCClient.get()
        # VRChat chatbox standard: [text(string), immediately(bool), play_notification(bool)]
        client.send_message("/chatbox/input", [message, True, True])

        logger.info(f"[chat_box] Message sent successfully: {message}")
        return "Message sent."

    except Exception as e:
        logger.exception(f"[chat_box] Error sending message via OSC: {e}")
        return f"failed to send message via chat box: {e}"
