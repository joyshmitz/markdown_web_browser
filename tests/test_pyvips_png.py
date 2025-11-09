#!/usr/bin/env python3
"""Test different PNG encoding methods with pyvips to find the issue."""

import pyvips
import sys

def test_png_formats():
    """Test various PNG encoding formats."""

    # Create a simple test image
    image = pyvips.Image.black(100, 100, bands=3)

    print("Testing PNG encoding formats...")

    # Test 1: Using ".png" (current approach)
    try:
        png_bytes = image.write_to_buffer(".png", compression=9, interlace=False)
        print("✅ Format '.png' works - bytes:", len(png_bytes))
    except Exception as e:
        print(f"❌ Format '.png' failed: {e}")

    # Test 2: Using "png" without dot
    try:
        png_bytes = image.write_to_buffer("png", compression=9, interlace=False)
        print("✅ Format 'png' works - bytes:", len(png_bytes))
    except Exception as e:
        print(f"❌ Format 'png' failed: {e}")

    # Test 3: Using "[png]" format
    try:
        png_bytes = image.pngsave_buffer(compression=9, interlace=False)
        print("✅ pngsave_buffer() works - bytes:", len(png_bytes))
    except Exception as e:
        print(f"❌ pngsave_buffer() failed: {e}")

    # Test 4: Check available savers
    print("\nAvailable savers:")
    try:
        # List available foreign save operations
        import subprocess
        result = subprocess.run(["vips", "--list", "classes"], capture_output=True, text=True)
        png_savers = [line for line in result.stdout.split('\n') if 'png' in line.lower()]
        for saver in png_savers[:5]:
            print(f"  - {saver.strip()}")
    except:
        pass

    # Test 5: Test with spng if available
    try:
        png_bytes = image.write_to_buffer(".spng", compression=9)
        print("✅ Format '.spng' works - bytes:", len(png_bytes))
    except Exception as e:
        print(f"❌ Format '.spng' not available: {e}")

    # Test 6: Test reading and re-encoding a PNG
    try:
        # Create a PNG, then read it back
        png_bytes = image.pngsave_buffer(compression=9, interlace=False)
        reloaded = pyvips.Image.new_from_buffer(png_bytes, "")
        re_encoded = reloaded.pngsave_buffer(compression=9, interlace=False)
        print(f"✅ Round-trip PNG encoding works - original: {len(png_bytes)}, re-encoded: {len(re_encoded)}")
    except Exception as e:
        print(f"❌ Round-trip failed: {e}")

if __name__ == "__main__":
    test_png_formats()