#!/usr/bin/env python3
"""Test the full browser capture pipeline now that libvips is installed."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.capture import capture_tiles, CaptureConfig


@pytest.mark.skip(reason="Requires Playwright and libvips - run manually")
async def test_full_capture():
    """Test capturing a website using the actual implementation."""

    print("Testing full browser capture with libvips...")

    # Create a simple test config using current API
    config = CaptureConfig(
        url="https://example.com",
        profile_id="test_profile",
        viewport_width=1280,
        viewport_height=2000,
        device_scale_factor=2,
        color_scheme="light",
    )

    try:
        print(f"Capturing {config.url}...")
        result = await capture_tiles(config)

        print("\n✅ Capture successful!")
        print(f"Tiles generated: {len(result.tiles)}")
        print(f"User agent: {result.user_agent}")
        print(f"DOM snapshot size: {len(result.dom_snapshot) if result.dom_snapshot else 0} bytes")
        print(f"Warnings: {len(result.warnings)}")
        print(f"Capture stats: {result.stats}")

        # List the generated tiles
        if result.tiles:
            print("\nGenerated tiles:")
            for i, tile in enumerate(result.tiles[:5]):  # Show first 5
                print(f"  Tile {i}: {tile.width}x{tile.height}, hash: {tile.sha256[:8]}...")

        return True

    except Exception as e:
        print(f"\n❌ Capture failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_full_capture())
    sys.exit(0 if success else 1)
