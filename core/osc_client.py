"""
core/osc_client.py

VRChatと直接通信するためのOSCクライアント。
環境変数から接続先を取得し、UDPでOSCメッセージを送信する。
"""

import os
import threading
from typing import Any

from loguru import logger
from pythonosc.udp_client import SimpleUDPClient


class OSCClient:
    """
    VRChatへOSCメッセージを送信するクライアントのシングルトンラッパー。
    """

    _instance: "OSCClient | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        ip = os.environ.get("VRC_OSC_IP", "127.0.0.1")
        port_str = os.environ.get("VRC_OSC_PORT", "9000")
        try:
            port = int(port_str)
        except ValueError:
            logger.warning(f"Invalid VRC_OSC_PORT '{port_str}', falling back to 9000")
            port = 9000

        self.client = SimpleUDPClient(ip, port)
        self.ip = ip
        self.port = port
        logger.info(f"[OSCClient] Initialized routing to {ip}:{port}")

    @classmethod
    def get(cls) -> "OSCClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = OSCClient()
        return cls._instance

    def send_message(self, address: str, value: Any) -> None:
        """
        指定したアドレスにOSCメッセージを送信する。
        """
        try:
            self.client.send_message(address, value)
            logger.debug(f"[OSCClient] Sent {address} : {value}")
        except Exception as e:
            logger.exception(f"[OSCClient] Failed to send {address}: {e}")
            raise
