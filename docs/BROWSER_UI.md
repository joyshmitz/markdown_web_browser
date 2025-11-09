# Markdown Web Browser UI

The **Markdown Web Browser** is a browser-like interface for viewing web pages converted to markdown. It provides a familiar browsing experience with navigation history, while displaying clean, readable markdown instead of HTML.

## Accessing the Browser

Once the server is running, navigate to:

```
http://localhost:8000/browser
```

## Features

### 🌐 Browser-Like Navigation

The interface mimics a traditional web browser with familiar controls:

- **Address Bar**: Enter any URL or search term
- **Back Button**: Navigate to previous pages (Alt+Left)
- **Forward Button**: Navigate to next pages (Alt+Right)
- **Refresh Button**: Reload the current page (Ctrl+R)
- **Navigation History**: Full browsing history with back/forward support

### 📄 Dual View Modes

Toggle between two markdown viewing modes:

1. **Rendered Markdown** - Beautiful, formatted HTML rendering using GitHub markdown styles
2. **Raw Markdown** - Syntax-highlighted source text with Prism.js

Switch views with the toolbar buttons or press **Ctrl+U**.

### 🔍 Smart URL Input

The address bar intelligently handles different input types:

- **Full URLs**: `https://example.com/article`
- **Bare domains**: `example.com` (automatically adds https://)
- **Search queries**: Any text without a URL format triggers Google search
  - Example: `markdown syntax` → searches Google

### ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Alt+Left` | Go back |
| `Alt+Right` | Go forward |
| `Ctrl+R` | Refresh current page |
| `Ctrl+U` | Toggle raw/rendered view |
| `Ctrl+L` | Focus address bar |
| `Enter` | Navigate to URL (when address bar focused) |

### 💾 Smart Caching

- Pages are cached for 1 hour after first load
- Use the refresh button to force reload and bypass cache
- Cache is stored in browser memory (cleared on page reload)

### 📊 Progress Indicators

Real-time feedback during page capture:

1. **Status Bar**: Shows current operation (loading, processing, complete)
2. **Loading Spinner**: Indicates active background processing
3. **Progress Bar**: Displays tile processing progress (e.g., "45/100 tiles")

### ⚠️ Error Handling

If a capture fails, you'll see:
- Clear error message explaining what went wrong
- **Try Again** button to retry the request
- Status bar updated with error details

## How It Works

1. **Enter URL**: Type a URL or search term in the address bar
2. **Job Submission**: Browser submits a capture job to the backend
3. **Page Capture**: Headless Chrome captures the page as tiled screenshots
4. **OCR Processing**: Screenshots are sent to OLMoCR for text extraction
5. **Markdown Generation**: Extracted text is stitched into clean markdown
6. **Display**: Markdown is rendered in your chosen view mode
7. **Caching**: Result is cached for faster subsequent visits

## Technical Details

### Architecture

```
┌─────────────────┐
│   Browser UI    │ (browser.html)
│  - Address bar  │
│  - Navigation   │
│  - View toggle  │
└────────┬────────┘
         │
    HTTP Requests
         │
┌────────▼────────┐
│  FastAPI Server │ (/jobs API)
│  - Job creation │
│  - Status poll  │
│  - Markdown API │
└────────┬────────┘
         │
    Job Queue
         │
┌────────▼────────┐
│ Capture Engine  │
│  - Playwright   │
│  - Tiling       │
│  - OCR client   │
└─────────────────┘
```

### Client-Side Libraries

- **marked.js** (v11.0.0) - Markdown to HTML rendering
- **Prism.js** (v1.29.0) - Syntax highlighting for raw markdown
- **GitHub Markdown CSS** - Styling for rendered markdown view

### Browser State Management

The browser maintains:
- **History Stack**: Array of visited URLs
- **Current Index**: Position in history (for back/forward)
- **Cache Map**: URL → { markdown, jobId, timestamp }
- **View State**: Current view mode (rendered/raw)

### API Endpoints Used

- `POST /jobs` - Submit new capture job
- `GET /jobs/{job_id}` - Poll job status and progress
- `GET /jobs/{job_id}/markdown.md` - Fetch markdown result

## Comparison: Browser UI vs Dashboard UI

| Feature | Browser UI (`/browser`) | Dashboard UI (`/`) |
|---------|------------------------|-------------------|
| **Purpose** | Browse and read pages | Monitor and debug jobs |
| **Navigation** | Back/forward, history | Job ID switching |
| **Views** | Rendered/Raw markdown | 7 tabs (rendered, raw, links, artifacts, etc.) |
| **Focus** | Reading experience | Job telemetry & debugging |
| **Progress** | Simple status bar | Detailed state machine, warnings, validation |
| **Target User** | End users, agents | Developers, operators |

## Use Cases

### For Humans
- Read web articles in clean markdown format
- Browse documentation without JavaScript/ads
- Archive web content as markdown
- Research with distraction-free reading

### For AI Agents
- Navigate web pages without vision models
- Extract structured content from complex layouts
- Follow links and explore websites programmatically
- Build knowledge graphs from web content

## Tips & Tricks

### Capturing Complex Pages
- Some pages with heavy JavaScript may take longer to render
- Progress bar shows tile processing in real-time
- Large pages (>100 tiles) may take several minutes

### Using Search
- Enter natural language: `best markdown editors`
- Browser converts to: `https://www.google.com/search?q=best+markdown+editors`
- Captures and displays the search results page

### Managing Cache
- First visit: Full capture (slow)
- Repeat visit: Instant load from cache
- Forced refresh: Clears cache and re-captures

### Viewing Source
- Toggle to "Raw" view to see the actual markdown
- Syntax highlighting helps identify structure
- Copy raw markdown for use in other tools

## Troubleshooting

### "Job timeout" Error
- Page took >10 minutes to process
- Try a simpler URL or smaller page
- Check server logs for stuck jobs

### "Invalid or revoked API key" Error
- OCR API key is missing or invalid
- Check `.env` file for `OLMOCR_API_KEY`
- Verify key has sufficient quota

### Blank Page After Loading
- Check browser console for JavaScript errors
- Verify markdown was actually generated
- Try toggling to raw view to see if markdown exists

### Slow Performance
- First capture of any URL requires full processing
- Subsequent visits use cache
- Large pages (100+ tiles) naturally take longer
- Consider using simpler pages for testing

## Future Enhancements

Planned features:
- [ ] Bookmarks/favorites
- [ ] Download markdown as .md file
- [ ] Print-friendly view
- [ ] Dark mode theme
- [ ] Reading time estimate
- [ ] Table of contents auto-generation
- [ ] Full-text search within cached pages
- [ ] Export history as JSON
- [ ] Configurable cache TTL
- [ ] Offline mode with IndexedDB

## Development

To modify the browser UI:

1. **HTML**: Edit `web/browser.html`
2. **CSS**: Edit `web/browser.css`
3. **JavaScript**: Edit `web/browser.js`
4. **Reload**: Refresh browser (no server restart needed for static files)

The browser UI is completely client-side - all processing happens in the browser using vanilla JavaScript (no framework required).

## License

Same as the main project - see LICENSE file.
