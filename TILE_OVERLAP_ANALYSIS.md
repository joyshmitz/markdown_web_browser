# Tile Overlap and Markdown Stitching Analysis

**Date**: 2025-11-09
**Question**: How are overlapping tiles handled? Do overlaps cause duplicate markdown? How is stitching done?

---

## TL;DR - Critical Findings

### ✅ What Works:
1. **Tiles DO have overlaps** (~120px by default)
2. **Overlap detection works** via SHA256 hashes
3. **Table headers are deduplicated** when overlaps match
4. **Seam markers are tracked** for overlap verification

### ⚠️ Potential Gap:
**General markdown content in overlaps is NOT deduplicated** - only table headers are trimmed. Regular text, headings, lists, etc. in overlap regions may appear twice in final output.

---

## Architecture Overview

The system has **two levels of overlaps**:

### Level 1: Viewport Overlaps (Scrolling)
**Location**: `app/capture.py:238`
```python
viewport_step = max(1, config.viewport_height - viewport_overlap_px)
# Default: viewport_height=2000, overlap=120px
# Step = 2000 - 120 = 1880px
```

- Page is scrolled in ~1880px increments
- Each viewport screenshot overlaps previous by ~120px
- This ensures content at viewport boundaries is captured twice

### Level 2: Tile Overlaps (Slicing)
**Location**: `app/tiler.py:122`
```python
step = max(1, target_long_side_px - overlap_px)
# Default: target=1288px, overlap=120px
# Step = 1288 - 120 = 1168px
```

- Each viewport screenshot is sliced into tiles
- Tiles overlap by ~120px vertically
- This ensures content at tile boundaries is captured twice

---

## How Overlaps Are Detected

### Overlap Hashing (app/tiler.py:171-187)

Each tile tracks overlap region hashes:

```python
@dataclass
class TileSlice:
    top_overlap_sha256: Optional[str]     # Hash of top 120px
    bottom_overlap_sha256: Optional[str]  # Hash of bottom 120px
    seam_top_hash: Optional[str]          # Seam marker at top
    seam_bottom_hash: Optional[str]       # Seam marker at bottom
```

**Hash Calculation**:
```python
def _overlap_sha(image, position: str, overlap_px: int) -> str:
    sample_height = min(overlap_px, image.height)

    if position == "top":
        strip = image.crop(0, 0, image.width, sample_height)
    else:
        y = max(0, image.height - sample_height)
        strip = image.crop(0, y, image.width, sample_height)

    png_bytes = strip.pngsave_buffer(**_PNG_ENCODE_ARGS)
    return hashlib.sha256(png_bytes).hexdigest()
```

### Overlap Matching (app/stitch.py:290-299)

```python
def _tiles_share_overlap(prev_tile: TileSlice | None, tile: TileSlice | None):
    """Check if two consecutive tiles share overlapping content."""

    # Method 1: Direct pixel hash comparison
    if prev_tile.bottom_overlap_sha256 and tile.top_overlap_sha256:
        if prev_tile.bottom_overlap_sha256 == tile.top_overlap_sha256:
            return True, False  # Match found, no seam marker

    # Method 2: Seam marker hash comparison (more robust)
    if prev_tile.seam_bottom_hash and tile.seam_top_hash:
        if prev_tile.seam_bottom_hash == tile.seam_top_hash:
            return True, True   # Match found, using seam marker

    return False, False
```

---

## What Gets Deduplicated

### ✅ Table Headers (app/stitch.py:213-247)

**ONLY table headers are deduplicated when overlaps match:**

```python
def _trim_duplicate_table_header(
    lines: list[str],
    last_signature: str | None,
    prev_tile: TileSlice | None,
    tile: TileSlice | None,
):
    """Drop repeated Markdown table header rows emitted across tiles."""

    extraction = _extract_table_header_signature(lines)
    if not extraction:
        return lines, last_signature, None  # Not a table

    header_signature, header_start = extraction
    overlap_match, _ = _tiles_share_overlap(prev_tile, tile)

    # Only trim if overlaps match AND headers are identical/similar
    if overlap_match:
        identical = header_signature == last_signature
        if identical:
            # Remove duplicate table header
            return trimmed, header_signature, TrimmedHeaderInfo(reason="identical")

        similarity = _header_similarity(header_signature, last_signature)
        if similarity >= 0.92:
            # Remove similar table header
            return trimmed, header_signature, TrimmedHeaderInfo(reason="similar")

    return lines, header_signature, None
```

**Detection Logic**:
1. Finds Markdown table pattern: `| Header |` followed by `| --- |`
2. Creates signature from header + separator
3. Compares with previous tile's table signature
4. Removes if identical OR >92% similar

### ❌ General Content NOT Deduplicated

**The following content types in overlap regions are NOT deduplicated:**

- Regular paragraphs
- Headings (`# Heading`)
- Lists (ordered/unordered)
- Blockquotes (`> quote`)
- Code blocks
- Links, images
- **Any non-table content**

**Example of Potential Duplication**:
```markdown
<!-- Tile 0 bottom overlap contains: -->
This is a paragraph that appears in the overlap region.
It will be captured in both tiles.

<!-- Tile 1 top overlap contains: -->
This is a paragraph that appears in the overlap region.
It will be captured in both tiles.
```

Both instances would appear in final markdown unless OCR produces identical output.

---

## How Stitching Works

### Stitching Flow (app/stitch.py:97-169)

```python
def stitch_markdown(chunks, tiles, dom_headings, dom_overlays, job_id):
    """Join OCR-derived Markdown segments with provenance."""

    processed: list[str] = []
    previous_tile: TileSlice | None = None

    for idx, chunk in enumerate(chunks):
        lines = _split_lines(chunk)
        tile = tiles[idx]

        # 1. Normalize headings
        lines = _normalize_headings(lines, ...)

        # 2. Trim duplicate table headers (ONLY dedup logic)
        lines = _trim_duplicate_table_header(lines, last_signature, previous_tile, tile)

        # 3. Apply DOM overlays (fix OCR errors)
        lines = _apply_dom_overlays(lines, overlay_index, ...)

        # 4. Add seam marker if overlaps match
        if tile and previous_tile:
            overlap_match, used_seam_marker = _tiles_share_overlap(previous_tile, tile)
            if overlap_match:
                # Add HTML comment marking the seam
                processed.append(_format_seam_marker(previous_tile, tile))

        # 5. Add provenance comment
        processed.append(_format_provenance(tile, job_id=job_id))

        # 6. Add the tile's markdown content
        body = "\n".join(lines).strip("\n")
        if body:
            processed.append(body)

        previous_tile = tile

    return "\n\n".join(processed)
```

**Output Structure**:
```markdown
<!-- source: tile_0000, y=0, height=1288, ... -->
[Tile 0 content]

<!-- seam-marker: prev=tile_0000 curr=tile_0001 overlap_hash=abc123... -->
<!-- source: tile_0001, y=1168, height=1288, ... -->
[Tile 1 content including overlap region]

<!-- seam-marker: prev=tile_0001 curr=tile_0002 overlap_hash=def456... -->
<!-- source: tile_0002, y=2336, height=1288, ... -->
[Tile 2 content including overlap region]
```

---

## Gap Analysis: Why Duplicates Might Occur

### Scenario 1: Text in Overlap Region

**Given**:
- Tile 0 ends at y=1288
- Tile 1 starts at y=1168 (120px overlap)
- Overlap region (y=1168 to y=1288) contains paragraph text

**Problem**:
```
Tile 0 OCR output:
"...end of content. This paragraph is in the overlap region and will appear twice."

Tile 1 OCR output:
"This paragraph is in the overlap region and will appear twice. Start of new content..."
```

**Stitched Result**:
```markdown
<!-- source: tile_0000 -->
...end of content. This paragraph is in the overlap region and will appear twice.

<!-- seam-marker: prev=tile_0000 curr=tile_0001 -->
<!-- source: tile_0001 -->
This paragraph is in the overlap region and will appear twice. Start of new content...
```

**DUPLICATE**: The overlap paragraph appears twice!

### Scenario 2: Heading in Overlap Region

**Given**:
- Overlap contains `## Important Heading`

**Problem**:
```
Tile 0 OCR: "...content\n\n## Important Heading\n"
Tile 1 OCR: "## Important Heading\n\nMore content..."
```

**Stitched Result**:
```markdown
<!-- source: tile_0000 -->
...content

## Important Heading

<!-- seam-marker: prev=tile_0000 curr=tile_0001 -->
<!-- source: tile_0001 -->
## Important Heading

More content...
```

**DUPLICATE**: Heading appears twice!

### Scenario 3: List Items in Overlap

**Problem**: Same duplication issue for list items.

---

## Why Table Headers Are Special

**Tables get special treatment because**:
1. Table headers are **structurally recognizable** via Markdown syntax
2. Duplicated table headers are **visually obvious and problematic**
3. Tables often span multiple tiles in tall documents
4. Easy to detect: `| Header |` followed by `| --- |`

**Why general content isn't deduplicated**:
1. Hard to identify overlap boundaries in plain text/markdown
2. OCR might produce slightly different output for same region
3. No clear heuristic for "is this line in overlap or not?"
4. Risk of incorrectly removing unique content

---

## Current Deduplication Strategy

### What System Does Now:

1. **Detect overlaps**: ✅ Via SHA256 hashes
2. **Mark overlaps**: ✅ Via `<!-- seam-marker -->` comments
3. **Deduplicate tables**: ✅ Via `_trim_duplicate_table_header()`
4. **Deduplicate general content**: ❌ NOT IMPLEMENTED

### What Happens in Practice:

**Best Case** (OCR produces different output for overlaps):
- Overlap content appears once in tile 0
- Overlap content appears differently in tile 1
- No visible duplication (OCR differences hide it)

**Worst Case** (OCR produces identical output):
- Overlap content appears identically in both tiles
- Visible duplication in final markdown
- User sees repeated paragraphs/headings/lists

**Table Case**:
- Table headers deduplicated automatically ✅
- Table rows might duplicate ❌

---

## Potential Solutions

### Option 1: Smart Content Deduplication

Similar to table header logic, detect and remove duplicates:

```python
def _trim_overlap_content(
    current_lines: list[str],
    previous_lines: list[str],
    overlap_match: bool,
) -> list[str]:
    """Remove duplicate content from overlap regions."""

    if not overlap_match:
        return current_lines  # No overlap, keep everything

    # Find common prefix between previous tail and current head
    prev_tail = previous_lines[-20:]  # Last ~20 lines of prev tile
    curr_head = current_lines[:20]     # First ~20 lines of curr tile

    # Use difflib to find matching sequences
    matcher = difflib.SequenceMatcher(a=prev_tail, b=curr_head)
    matching_blocks = matcher.get_matching_blocks()

    # Remove matching prefix from current tile
    if matching_blocks:
        largest_match = max(matching_blocks, key=lambda b: b.size)
        if largest_match.size > 2:  # At least 3 lines match
            # Trim the duplicate lines from current tile
            return current_lines[largest_match.b + largest_match.size:]

    return current_lines
```

**Pros**:
- Removes all duplicate content, not just tables
- Uses same overlap detection mechanism

**Cons**:
- Might incorrectly remove legitimate repeated content
- OCR differences could prevent matches
- Complex heuristics needed

### Option 2: Reduce/Eliminate Overlaps

```python
# In settings:
viewport_overlap_px: int = 0   # No overlap
tile_overlap_px: int = 0       # No overlap
```

**Pros**:
- No duplicates possible
- Simpler logic

**Cons**:
- Risk of cutting content at boundaries
- Loss of overlap validation mechanism
- Can't verify sweep stability

### Option 3: Better OCR Boundary Detection

Train OCR model to recognize tile boundaries and avoid duplicating:

**Cons**:
- Requires OCR model changes
- Complex implementation

### Option 4: Post-processing Deduplication

Add a final pass that detects and removes duplicated blocks:

```python
def deduplicate_markdown(markdown: str) -> str:
    """Remove duplicate blocks that appear consecutively."""

    lines = markdown.splitlines()
    deduplicated = []
    window_size = 10

    i = 0
    while i < len(lines):
        # Look for repeated sequences
        current_window = lines[i:i+window_size]
        next_window = lines[i+window_size:i+2*window_size]

        if current_window == next_window:
            # Found duplicate - skip second occurrence
            deduplicated.extend(current_window)
            i += 2 * window_size
        else:
            deduplicated.append(lines[i])
            i += 1

    return "\n".join(deduplicated)
```

---

## Recommendations

### Immediate:
1. **Document the behavior**: Users should know overlaps exist and might duplicate
2. **Monitor metrics**: Track how often duplicates occur in practice
3. **Test with real data**: Run analysis on actual OCR outputs to quantify issue

### Short-term:
1. **Extend table deduplication**: Add similar logic for lists, blockquotes
2. **Use seam markers**: Leverage existing `<!-- seam-marker -->` comments to identify boundaries
3. **Add configuration**: Let users choose overlap strategy

### Long-term:
1. **Smart deduplication**: Implement difflib-based content matching
2. **OCR-aware boundaries**: Teach OCR model about tile boundaries
3. **Hybrid approach**: Small overlaps + smart deduplication

---

## Testing Recommendations

### Test Case 1: Paragraph in Overlap
```python
def test_paragraph_deduplication():
    # Create two tiles with overlapping paragraph
    tile0_markdown = "Previous content.\n\nOverlap paragraph text.\n"
    tile1_markdown = "Overlap paragraph text.\n\nNext content."

    result = stitch_markdown([tile0_markdown, tile1_markdown], tiles, ...)

    # Should contain "Overlap paragraph text" only ONCE
    assert result.count("Overlap paragraph text") == 1
```

### Test Case 2: Heading in Overlap
```python
def test_heading_deduplication():
    tile0_markdown = "Content before.\n\n## Overlap Heading\n"
    tile1_markdown = "## Overlap Heading\n\nContent after."

    result = stitch_markdown([tile0_markdown, tile1_markdown], tiles, ...)

    assert result.count("## Overlap Heading") == 1
```

### Test Case 3: Table Already Works
```python
def test_table_header_deduplication():
    # This should already pass due to existing logic
    tile0_markdown = "| Col1 | Col2 |\n| --- | --- |\n| A | B |\n"
    tile1_markdown = "| Col1 | Col2 |\n| --- | --- |\n| C | D |\n"

    result = stitch_markdown([tile0_markdown, tile1_markdown], tiles, ...)

    # Table header appears only once
    assert result.count("| Col1 | Col2 |") == 1
```

---

## Conclusion

**Current State**:
- ✅ Overlaps exist and are tracked via SHA256 hashes
- ✅ Table headers are deduplicated
- ❌ General content (paragraphs, headings, lists) is NOT deduplicated
- ⚠️ Potential for visible duplicates in final markdown

**Impact**:
- **Low** if OCR produces different output for overlaps
- **High** if OCR produces identical output for overlaps
- **Critical** for tables (already handled)

**Next Steps**:
1. Measure how often duplicates occur in production
2. Decide if deduplication is worth the complexity
3. Implement smart deduplication if needed
4. Add tests to prevent regressions
