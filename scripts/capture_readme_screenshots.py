#!/usr/bin/env python3
"""
Capture screenshots for README.md demonstrating the finviz.com example.

This script captures:
1. BEFORE: Original finviz.com website
2. AFTER (Raw): Browser UI showing raw markdown with syntax highlighting
3. AFTER (Rendered): Browser UI showing rendered markdown
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright


async def capture_screenshots():
    """Capture all required screenshots for README."""

    output_dir = Path("docs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        # Launch browser with same stealth settings as our system
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--headless=new",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--window-size=1920,1080",
            ]
        )

        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            ),
        )

        page = await context.new_page()

        # Add stealth masking
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            delete navigator.__proto__.webdriver;
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
                    { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: 'Portable Document Format' },
                    { name: 'Native Client', filename: 'internal-nacl-plugin', description: 'Native Client Executable' }
                ]
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            if (!window.chrome) {
                window.chrome = {
                    runtime: {},
                    loadTimes: function() {},
                    csi: function() {},
                    app: {}
                };
            }
        """)

        print("📸 Capturing BEFORE screenshot: finviz.com original...")

        # 1. Capture original finviz.com (BEFORE)
        try:
            await page.goto("https://finviz.com", wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(3000)  # Wait for dynamic content
            await page.screenshot(path=str(output_dir / "finviz_before.png"), full_page=False)
            print("✅ Saved: docs/images/finviz_before.png")
        except Exception as e:
            print(f"⚠️  Warning: Could not capture finviz.com directly: {e}")

        print("\n📸 Capturing AFTER screenshots: Browser UI with markdown...")

        # 2. Load the browser UI with finviz URL
        await page.goto("http://localhost:8000/browser?url=https://finviz.com", wait_until="load", timeout=30000)
        print("   Browser UI loaded, waiting for content processing...")
        
        # Wait for the welcome message to disappear and content to load
        # We'll wait for status bar to show it's done
        await page.wait_for_timeout(5000)
        
        # Wait for rendered content to appear (not the welcome message)
        try:
            await page.wait_for_function(
                """() => {
                    const content = document.querySelector('#rendered-content');
                    if (!content) return false;
                    const text = content.textContent || '';
                    // Check if it has real content, not just the welcome message
                    return text.includes('Finviz') || text.includes('DOW') || text.length > 500;
                }""",
                timeout=120000  # 2 minutes for finviz to be captured and processed
            )
            print("   ✓ Content loaded successfully")
        except Exception as e:
            print(f"   ⚠️  Warning: Timeout waiting for content, proceeding anyway: {e}")
        
        await page.wait_for_timeout(2000)

        # 3. Capture RENDERED view first (it's the default)
        await page.screenshot(path=str(output_dir / "finviz_after_rendered.png"), full_page=True)
        print("✅ Saved: docs/images/finviz_after_rendered.png")

        # 4. Click "Raw" button to show raw markdown
        print("   Switching to raw markdown view...")
        try:
            await page.click("#raw-btn")
            await page.wait_for_timeout(1000)
            
            # Wait for raw content to be visible
            await page.wait_for_selector("#raw-content.active", timeout=5000)
        except Exception as e:
            print(f"   ⚠️  Note: Could not switch to raw view: {e}")

        await page.screenshot(path=str(output_dir / "finviz_after_raw.png"), full_page=True)
        print("✅ Saved: docs/images/finviz_after_raw.png")

        await browser.close()

    print("\n✨ All screenshots captured successfully!")
    print(f"   Location: {output_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(capture_screenshots())
