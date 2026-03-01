import os
import sys
import time
import base64
from io import BytesIO
from PIL import Image

# プロジェクトルートディレクトリをパスに追加してモジュールをインポート可能にする
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_vision_test():
    """Test script for the new get_current_view tool."""
    print("=== Testing Spout Vision Tool ===")
    os.environ["SPOUT_SENDER_NAME"] = "VRCSender1"  # Or your testing sender name

    try:
        from inputs.vision import get_current_view

        print("Successfully imported get_current_view tool.")

        # Invoke the tool
        print("Invoking get_current_view()...")
        t0 = time.time()
        result = get_current_view.invoke({})
        dt = time.time() - t0
        print(f"Tool executed in {dt:.3f} seconds.")

        if not isinstance(result, list) or len(result) == 0:
            print("FAILED: Result is not a non-empty list.")
            print(result)
            return

        first_item = result[0]
        if first_item.get("type") == "text":
            print(f"FAILED to capture frame: {first_item.get('text')}")
            print(
                "Make sure VRChat or a Spout sender with the name 'VRChat' is running."
            )
            return

        if first_item.get("type") == "image_url":
            url = first_item["image_url"]["url"]
            print("SUCCESS: Received image URL format")
            print(f"Base64 String Length: {len(url)} characters")

            # Extract header and data
            header, b64_data = url.split(",", 1)
            print(f"Image Header: {header}")

            # Optional: verify image size/format by decoding
            try:
                img_data = base64.b64decode(b64_data)
                img = Image.open(BytesIO(img_data))
                print(f"Decoded Image Dimensions: {img.size}")
                print(f"Decoded Image Format: {img.format}")
                print(f"Decoded Image Mode: {img.mode}")

                # Save the image for visual verification
                output_path = "test_vision_output.jpg"
                img.save(output_path)
                print(f"SUCCESS: Saved captured image to '{output_path}'.")
                print("Please open this file to verify the captured frame.")
                print("Test fully Passed!")
            except Exception as e:
                print(f"FAILED to decode base64 image: {e}")
        else:
            print(f"FAILED: Unknown result type dict: {first_item}")

    except ImportError as e:
        print(f"FAILED: Import Error: {e}")
    except Exception as e:
        print(f"FAILED: Unexpected Exception: {e}")


if __name__ == "__main__":
    run_vision_test()
