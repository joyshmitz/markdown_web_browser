"""Tests for tiler helpers."""

from __future__ import annotations

import hashlib

import pytest

try:  # pragma: no cover - exercised only when libvips missing
    import pyvips as _pyvips
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"pyvips unavailable: {exc}", allow_module_level=True)
else:
    if not hasattr(_pyvips, "Image"):
        pytest.skip("pyvips.Image not available", allow_module_level=True)
    pyvips = _pyvips  # type: Any
    if not hasattr(getattr(pyvips, "Image", None), "black"):
        pytest.skip("pyvips stub lacks Image.black; skipping tiler tests", allow_module_level=True)

from app.tiler import TileSlice, slice_into_tiles, validate_tiles


def _png_bytes(width: int = 10, height: int = 10) -> bytes:
    image = pyvips.Image.black(width, height).add(1)  # type: ignore[attr-defined]
    return image.write_to_buffer(".png")


def test_validate_tiles_passes_for_valid_png() -> None:
    png = _png_bytes()
    tile = TileSlice(
        index=0,
        png_bytes=png,
        sha256=hashlib.sha256(png).hexdigest(),
        width=10,
        height=10,
        scale=1.0,
        source_y_offset=0,
        viewport_y_offset=0,
        overlap_px=0,
        top_overlap_sha256=None,
        bottom_overlap_sha256=None,
    )

    validate_tiles([tile])  # should not raise


def test_validate_tiles_raises_on_checksum_mismatch() -> None:
    png = _png_bytes()
    tile = TileSlice(
        index=5,
        png_bytes=png,
        sha256="deadbeef",
        width=10,
        height=10,
        scale=1.0,
        source_y_offset=0,
        viewport_y_offset=0,
        overlap_px=0,
        top_overlap_sha256=None,
        bottom_overlap_sha256=None,
    )

    try:
        validate_tiles([tile])
    except ValueError as exc:
        assert "checksum" in str(exc)
    else:  # pragma: no cover - safety net
        raise AssertionError("Expected checksum mismatch ValueError")


@pytest.mark.asyncio()
async def test_slice_into_tiles_downscales_wide_capture() -> None:
    png = _png_bytes(width=2000, height=600)

    tiles = await slice_into_tiles(png, target_long_side_px=1000)

    assert len(tiles) == 1  # height fits in a single tile
    tile = tiles[0]
    assert tile.width == 1000  # downscaled from 2000
    assert tile.scale == pytest.approx(0.5)
    assert tile.height == 600  # untouched because height < target


@pytest.mark.asyncio()
async def test_slice_into_tiles_records_overlap_hashes() -> None:
    png = _png_bytes(width=800, height=2800)

    tiles = await slice_into_tiles(png, overlap_px=120, target_long_side_px=1000)

    assert len(tiles) >= 3  # tall image yields multiple tiles
    for tile in tiles:
        assert tile.top_overlap_sha256 is not None
        assert tile.bottom_overlap_sha256 is not None
    for previous, current in zip(tiles, tiles[1:]):
        assert previous.bottom_overlap_sha256 == current.top_overlap_sha256
