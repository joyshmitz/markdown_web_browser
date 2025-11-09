#!/usr/bin/env python3
"""
Script to capture before/after screenshots for README.md

This script:
1. Captures original web pages (before) using Playwright
2. Captures our browser UI showing rendered markdown (after)
3. Saves as PNG format (high quality, good for screenshots)
4. Stores in docs/images/ directory
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright


# Screenshot configuration
SCREENSHOTS_DIR = Path(__file__).parent.parent / "docs" / "images"
VIEWPORT_SIZE = {"width": 1280, "height": 900}

# URLs to capture
URLS = {
    "finviz": "https://finviz.com/screener.ashx?v=111",
    "example": "https://example.com",
    "hackernews": "https://news.ycombinator.com",
}


async def capture_original_page(page, url: str, output_path: Path):
    """Capture a screenshot of the original web page."""
    print(f"📸 Capturing original page: {url}")

    await page.goto(url, wait_until="networkidle", timeout=30000)

    # Wait a bit for any dynamic content to load
    await page.wait_for_timeout(2000)

    # Take screenshot (PNG format for quality)
    await page.screenshot(path=str(output_path), type="png")
    print(f"✅ Saved: {output_path}")


async def capture_browser_ui(page, url: str, output_path: Path, server_url: str = "http://localhost:8000"):
    """Capture a screenshot of our browser UI showing rendered markdown."""
    print(f"📸 Capturing browser UI for: {url}")

    # Navigate to our browser UI
    browser_url = f"{server_url}/browser"
    await page.goto(browser_url, wait_until="networkidle")

    # Wait for page to load
    await page.wait_for_timeout(1000)

    # Enter the URL in the address bar
    url_input = await page.query_selector("#url-input")
    if url_input:
        await url_input.fill(url)
        await url_input.press("Enter")

        # Wait for job to complete (simplified - in reality would need to poll)
        # For now, just wait a reasonable amount of time
        print("⏳ Waiting for markdown capture to complete...")
        await page.wait_for_timeout(60000)  # Wait up to 60 seconds

        # Take screenshot (PNG format for quality)
        await page.screenshot(path=str(output_path), type="png")
        print(f"✅ Saved: {output_path}")
    else:
        print("❌ Could not find URL input field")


async def main():
    """Main entry point."""
    # Create screenshots directory
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Screenshots will be saved to: {SCREENSHOTS_DIR}")

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(channel="chrome")
        context = await browser.new_context(viewport=VIEWPORT_SIZE)
        page = await context.new_page()

        try:
            # Capture example.com (simple page)
            await capture_original_page(
                page,
                URLS["example"],
                SCREENSHOTS_DIR / "example_before.png"
            )

            # Capture Hacker News (moderate complexity)
            await capture_original_page(
                page,
                URLS["hackernews"],
                SCREENSHOTS_DIR / "hackernews_before.png"
            )

            # Capture finviz (complex page)
            await capture_original_page(
                page,
                URLS["finviz"],
                SCREENSHOTS_DIR / "finviz_before.png"
            )

            print("\n" + "="*60)
            print("✅ Original page screenshots complete!")
            print("="*60)
            print("\nTo capture 'after' screenshots of the browser UI:")
            print("1. Start the server: uv run python -m app.cli serve")
            print("2. Run this script with --with-ui flag")
            print("="*60)

        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
