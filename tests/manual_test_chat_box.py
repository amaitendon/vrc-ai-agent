import asyncio
import os
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from actuators.chat_box import chat_box

async def manual_test():
    """
    手動テスト用のスクリプト。
    VRChatと vrchat-mcp-osc サーバーが連携できるか確認します。
    """
    print("=== ChatBox マニュアルテスト ===")
    print("VRChatが起動されており、OSCが有効になっていることを確認してください。")
    print("送信メッセージ: 'Hello from VRC AI Agent!'")
    print("-----------------------------------")
    
    try:
        # ツールを直接実行
        result = await chat_box.ainvoke({"message": "Hello from VRC AI Agent!"})
        print(f"実行結果:\n{result}")
        print("-----------------------------------")
        print("VRChatのチャットボックスを確認して、メッセージが表示されていれば成功です！")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    asyncio.run(manual_test())
