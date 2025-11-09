"""Tests for tile overlap deduplication functionality."""

from __future__ import annotations

import pytest

from app.dedup import (
    DeduplicationResult,
    deduplicate_tile_overlap,
    _find_exact_boundary_match,
    _find_overlap_fuzzy,
    _find_overlap_sequence_matching,
    _estimate_overlap_lines,
)
from app.tiler import TileSlice


def create_test_tile(
    index: int,
    overlap_px: int = 120,
    bottom_hash: str | None = "abc123",
    top_hash: str | None = "abc123",
) -> TileSlice:
    """Helper to create test tiles."""
    return TileSlice(
        index=index,
        png_bytes=b"fake",
        sha256="fake_sha",
        width=1280,
        height=1288,
        scale=1.0,
        source_y_offset=index * 1168,
        viewport_y_offset=0,
        overlap_px=overlap_px,
        top_overlap_sha256=top_hash,
        bottom_overlap_sha256=bottom_hash,
    )


class TestExactBoundaryMatch:
    """Tests for exact line matching algorithm."""

    def test_perfect_match(self):
        """Test exact matching with identical lines."""
        prev_tail = ["Line 1", "Line 2", "Overlap A", "Overlap B"]
        curr_head = ["Overlap A", "Overlap B", "Line 3", "Line 4"]

        match_count = _find_exact_boundary_match(prev_tail, curr_head)

        assert match_count == 2

    def test_no_match(self):
        """Test no match when lines differ."""
        prev_tail = ["Line 1", "Line 2", "Different A"]
        curr_head = ["Different B", "Line 3", "Line 4"]

        match_count = _find_exact_boundary_match(prev_tail, curr_head)

        assert match_count == 0

    def test_partial_match(self):
        """Test partial match at boundary."""
        # Boundary match: end of prev must match start of curr
        prev_tail = ["Line 1", "Line 2", "Overlap A", "Overlap B"]
        curr_head = ["Overlap A", "Overlap B", "Line 3", "Line 4"]

        match_count = _find_exact_boundary_match(prev_tail, curr_head)

        # Should match last 2 lines of prev with first 2 lines of curr
        assert match_count == 2

    def test_trailing_whitespace_ignored(self):
        """Test that trailing whitespace is ignored."""
        prev_tail = ["Line with spaces   ", "Another line  "]
        curr_head = ["Line with spaces", "Another line", "Next"]

        match_count = _find_exact_boundary_match(prev_tail, curr_head)

        assert match_count == 2

    def test_empty_lists(self):
        """Test empty input lists."""
        assert _find_exact_boundary_match([], []) == 0
        assert _find_exact_boundary_match(["Line"], []) == 0
        assert _find_exact_boundary_match([], ["Line"]) == 0


class TestSequenceMatching:
    """Tests for sequence matching algorithm."""

    def test_requires_minimum_similarity(self):
        """Test that low similarity blocks don't match."""
        prev_tail = ["Completely different", "text here"]
        curr_head = ["Totally other", "content there", "more"]

        result = _find_overlap_sequence_matching(
            prev_tail,
            curr_head,
            min_match_lines=2,
            min_similarity=0.90,
        )

        assert result is None

    def test_requires_boundary_alignment(self):
        """Test that matches must be at boundary (end of prev, start of curr)."""
        prev_tail = ["Start", "Middle match", "End"]
        curr_head = ["Different", "Middle match", "Other"]

        result = _find_overlap_sequence_matching(
            prev_tail,
            curr_head,
            min_match_lines=1,
            min_similarity=0.90,
        )

        # "Middle match" appears in both but not at boundary
        assert result is None


class TestFuzzyMatching:
    """Tests for fuzzy line-by-line matching."""

    def test_fuzzy_match_with_punctuation_differences(self):
        """Test fuzzy matching handles punctuation variations."""
        prev_tail = ["Hello World!", "How are you?", "Fine thanks"]
        curr_head = ["Hello World", "How are you", "Fine thanks", "Next"]

        match_count = _find_overlap_fuzzy(
            prev_tail,
            curr_head,
            line_threshold=0.85,
            min_match_lines=2,
        )

        assert match_count == 3

    def test_stops_at_significant_difference(self):
        """Test fuzzy matching finds longest valid match."""
        # For boundary matches, we look for end of prev matching start of curr
        prev_tail = ["Line 1", "Match 1", "Match 2"]
        curr_head = ["Match 1", "Match 2", "Next"]

        match_count = _find_overlap_fuzzy(
            prev_tail,
            curr_head,
            line_threshold=0.85,
            min_match_lines=2,
        )

        # Should match last 2 lines of prev with first 2 of curr
        assert match_count == 2

    def test_empty_lines_counted_as_match(self):
        """Test that matching empty lines count as matches."""
        prev_tail = ["Text", "", "More"]
        curr_head = ["", "More", "Next"]

        match_count = _find_overlap_fuzzy(
            prev_tail,
            curr_head,
            line_threshold=0.85,
            min_match_lines=1,
        )

        # Should match empty line and "More"
        assert match_count == 2


class TestEstimateOverlapLines:
    """Tests for overlap line estimation."""

    def test_estimate_from_pixel_overlap(self):
        """Test line estimation from pixel overlap."""
        tile1 = create_test_tile(0, overlap_px=120)
        tile2 = create_test_tile(1, overlap_px=120)

        estimated = _estimate_overlap_lines(tile1, tile2)

        # 120px / 25px per line ~= 4-5 lines
        assert estimated >= 3
        assert estimated <= 6

    def test_minimum_estimate(self):
        """Test minimum estimate is enforced."""
        tile1 = create_test_tile(0, overlap_px=10)  # Very small overlap
        tile2 = create_test_tile(1, overlap_px=10)

        estimated = _estimate_overlap_lines(tile1, tile2)

        # Should return minimum of 3
        assert estimated == 3


class TestDeduplicateTileOverlap:
    """Integration tests for complete deduplication flow."""

    def test_exact_match_deduplication(self):
        """Test exact line matching removes duplicates."""
        prev_lines = ["Line 1", "Line 2", "Overlap A", "Overlap B"]
        curr_lines = ["Overlap A", "Overlap B", "Line 3", "Line 4"]
        prev_tile = create_test_tile(0, bottom_hash="match123")
        curr_tile = create_test_tile(1, top_hash="match123")

        result_lines, info = deduplicate_tile_overlap(
            prev_lines, curr_lines, prev_tile, curr_tile
        )

        assert info.lines_removed == 2
        assert info.method == "exact"
        assert info.similarity == 1.0
        assert result_lines == ["Line 3", "Line 4"]

    def test_sequence_match_with_ocr_errors(self):
        """Test sequence matching handles OCR variations."""
        prev_lines = ["Hello World!", "Second line."]
        curr_lines = ["Hello World", "Second line", "New content"]
        prev_tile = create_test_tile(0, bottom_hash="match123")
        curr_tile = create_test_tile(1, top_hash="match123")

        result_lines, info = deduplicate_tile_overlap(
            prev_lines, curr_lines, prev_tile, curr_tile
        )

        assert info.lines_removed == 2
        assert info.method in ["exact", "sequence", "fuzzy"]  # Any tier can match
        assert result_lines == ["New content"]

    def test_no_dedup_without_pixel_overlap(self):
        """Test no deduplication when pixel hashes don't match."""
        prev_lines = ["Line 1", "Line 2", "Overlap"]
        curr_lines = ["Overlap", "Line 3", "Line 4"]
        prev_tile = create_test_tile(0, bottom_hash="abc123")
        curr_tile = create_test_tile(1, top_hash="def456")  # Different!

        result_lines, info = deduplicate_tile_overlap(
            prev_lines, curr_lines, prev_tile, curr_tile
        )

        assert info.lines_removed == 0
        assert info.method == "no_overlap"
        assert result_lines == curr_lines  # Unchanged

    def test_disabled_deduplication(self):
        """Test deduplication can be disabled."""
        prev_lines = ["Line 1", "Overlap"]
        curr_lines = ["Overlap", "Line 2"]
        prev_tile = create_test_tile(0, bottom_hash="match123")
        curr_tile = create_test_tile(1, top_hash="match123")

        result_lines, info = deduplicate_tile_overlap(
            prev_lines,
            curr_lines,
            prev_tile,
            curr_tile,
            enabled=False,  # Disabled
        )

        assert info.lines_removed == 0
        assert info.method == "disabled"
        assert result_lines == curr_lines  # Unchanged

    def test_insufficient_content(self):
        """Test no dedup when content is too short."""
        prev_lines = ["Single line"]
        curr_lines = ["Single line", "Next"]
        prev_tile = create_test_tile(0, bottom_hash="match123")
        curr_tile = create_test_tile(1, top_hash="match123")

        result_lines, info = deduplicate_tile_overlap(
            prev_lines,
            curr_lines,
            prev_tile,
            curr_tile,
            min_overlap_lines=2,  # Require at least 2 lines
        )

        assert info.lines_removed == 0
        assert info.method == "insufficient_content"

    def test_safety_limit_prevents_over_deletion(self):
        """Test safety mechanism prevents removing too many lines."""
        # Create scenario where match is larger than 3x estimated overlap
        prev_lines = ["Line" + str(i) for i in range(50)]
        curr_lines = ["Line" + str(i) for i in range(50)] + ["New"]
        prev_tile = create_test_tile(0, overlap_px=10, bottom_hash="match123")  # Small overlap
        curr_tile = create_test_tile(1, overlap_px=10, top_hash="match123")

        result_lines, info = deduplicate_tile_overlap(
            prev_lines, curr_lines, prev_tile, curr_tile
        )

        # Estimated overlap from 10px = 3 lines minimum
        # Max removable = 3 * 3 = 9 lines
        # But exact match found 50 lines, which exceeds safety limit
        # Should NOT remove all 50 (safety mechanism kicks in)
        if info.lines_removed > 0:
            assert info.lines_removed <= 9, "Should not exceed safety limit"

    def test_markdown_heading_in_overlap(self):
        """Test deduplication works with markdown headings."""
        prev_lines = ["# Heading", "Paragraph text"]
        curr_lines = ["# Heading", "Paragraph text", "More content"]
        prev_tile = create_test_tile(0, bottom_hash="match123")
        curr_tile = create_test_tile(1, top_hash="match123")

        result_lines, info = deduplicate_tile_overlap(
            prev_lines, curr_lines, prev_tile, curr_tile
        )

        assert info.lines_removed == 2
        assert "# Heading" not in result_lines
        assert result_lines == ["More content"]

    def test_markdown_list_in_overlap(self):
        """Test deduplication works with markdown lists."""
        prev_lines = ["- List item 1", "- List item 2"]
        curr_lines = ["- List item 1", "- List item 2", "- List item 3"]
        prev_tile = create_test_tile(0, bottom_hash="match123")
        curr_tile = create_test_tile(1, top_hash="match123")

        result_lines, info = deduplicate_tile_overlap(
            prev_lines, curr_lines, prev_tile, curr_tile
        )

        assert info.lines_removed == 2
        assert result_lines == ["- List item 3"]


class TestTierFallback:
    """Test that deduplication falls through tiers correctly."""

    def test_fallback_to_sequence_when_exact_fails(self):
        """Test fallback to sequence matching when exact matching fails."""
        # Similar but not exact (punctuation differences)
        prev_lines = ["Line 1!", "Line 2?"]
        curr_lines = ["Line 1", "Line 2", "Line 3"]
        prev_tile = create_test_tile(0, bottom_hash="match123")
        curr_tile = create_test_tile(1, top_hash="match123")

        result_lines, info = deduplicate_tile_overlap(
            prev_lines, curr_lines, prev_tile, curr_tile
        )

        # Should match via sequence or fuzzy, not exact
        assert info.lines_removed == 2
        assert info.method in ["sequence", "fuzzy"]
        assert result_lines == ["Line 3"]

    def test_no_match_when_all_tiers_fail(self):
        """Test no deduplication when all matching tiers fail."""
        prev_lines = ["Completely", "Different"]
        curr_lines = ["Totally", "Other", "Content"]
        prev_tile = create_test_tile(0, bottom_hash="match123")
        curr_tile = create_test_tile(1, top_hash="match123")

        result_lines, info = deduplicate_tile_overlap(
            prev_lines, curr_lines, prev_tile, curr_tile
        )

        # No confident match found
        assert info.lines_removed == 0
        assert info.method == "no_match"
        assert result_lines == curr_lines  # Unchanged
