"""
actuators/chat_box.py

VRChatのチャットボックスを制御するツール。
Model Context Protocol (MCP) を利用し、vrchat-mcp-osc と通信する。
"""

from langchain_core.tools import tool
from loguru import logger
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@tool
async def chat_box(message: str) -> str:
    """
    VRChatのチャットボックスに指定したメッセージを送信する。
    メッセージは144文字以内を推奨。絵文字・日本語も送信可能。
    """
    cmd = "npx.cmd" if os.name == "nt" else "npx"
    
    server_params = StdioServerParameters(
        command=cmd,
        args=["vrchat-mcp-osc"]
    )

    try:
        logger.debug(f"[chat_box] Starting vrchat-mcp-osc for message: '{message}'")
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "send_message",
                    arguments={"message": message}
                )
                
                if result.isError:
                    error_text = result.content[0].text if result.content else "(no content)"
                    error_msg = f"Failed to send message: {error_text}"
                    logger.warning(f"[chat_box] {error_msg}")
                    return error_msg
                
                response_text = result.content[0].text if result.content else "Message sent to chat box."
                logger.info(f"[chat_box] Message sent successfully: {response_text}")
                return response_text

    except Exception as e:
        logger.exception(f"[chat_box] Error communicating with MCP server: {e}")
        return f"failed to send message via chat box: {e}"
