from aiavatar.device import AudioDevice


def list_devices():
    print("認識されているオーディオデバイス一覧を追加します...")
    try:
        devices = AudioDevice()
        devices.list_audio_devices()
        print("\n上記の一覧から、使用したいマイクの 'index' 番号を探してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("PyAudioが正しくインストールされているか確認してください。")


if __name__ == "__main__":
    list_devices()
