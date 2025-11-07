"""Tile slicing helpers backed by pyvips."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import List

import pyvips

# PNG encode defaults that match PLAN §19.3 guidance.
_PNG_ENCODE_ARGS = {
    "compression": 9,
    "interlace": False,
}


@dataclass(slots=True)
class TileSlice:
    """A single OCR-ready tile plus provenance metadata."""

    index: int
    png_bytes: bytes
    sha256: str
    width: int
    height: int
    scale: float
    source_y_offset: int
    viewport_y_offset: int
    overlap_px: int


async def slice_into_tiles(
    image_bytes: bytes,
    *,
    overlap_px: int = 120,
    tile_index_offset: int = 0,
    viewport_y_offset: int = 0,
    target_long_side_px: int = 1288,
) -> List[TileSlice]:
    """Slice a viewport-sized screenshot into <=1288 px-long tiles.

    The longest dimension for each resulting tile is capped at ``target_long_side_px``
    while keeping ≈``overlap_px`` pixels of vertical overlap so OCR stitching can
    deduplicate seams deterministically.
    """

    return await asyncio.to_thread(
        _slice_sync,
        image_bytes,
        overlap_px,
        tile_index_offset,
        viewport_y_offset,
        target_long_side_px,
    )


def _slice_sync(
    image_bytes: bytes,
    overlap_px: int,
    tile_index_offset: int,
    viewport_y_offset: int,
    target_long_side_px: int,
) -> List[TileSlice]:
    image = pyvips.Image.new_from_buffer(image_bytes, "", access="sequential")

    # Downscale only when the width exceeds the 1288 px guidance; height is handled via tiling.
    scale = 1.0
    if image.width > target_long_side_px:
        scale = target_long_side_px / image.width
        image = image.resize(scale)

    tiles: List[TileSlice] = []
    height = image.height
    width = image.width

    if height <= target_long_side_px:
        png_bytes = image.write_to_buffer(".png", **_PNG_ENCODE_ARGS)
        tiles.append(
            TileSlice(
                index=tile_index_offset,
                png_bytes=png_bytes,
                sha256=hashlib.sha256(png_bytes).hexdigest(),
                width=width,
                height=height,
                scale=scale,
                source_y_offset=int(viewport_y_offset),
                viewport_y_offset=viewport_y_offset,
                overlap_px=overlap_px,
            )
        )
        return tiles

    step = max(1, target_long_side_px - overlap_px)
    cursor = 0
    tile_idx = 0

    while cursor < height:
        remaining = height - cursor
        tile_height = target_long_side_px if remaining > target_long_side_px else remaining

        # Ensure the final tile ends exactly at the bottom to avoid tiny slivers.
        if remaining < target_long_side_px and cursor > 0:
            cursor = max(0, height - target_long_side_px)
            tile_height = height - cursor

        cropped = image.crop(0, cursor, width, tile_height)
        png_bytes = cropped.write_to_buffer(".png", **_PNG_ENCODE_ARGS)
        tiles.append(
            TileSlice(
                index=tile_index_offset + tile_idx,
                png_bytes=png_bytes,
                sha256=hashlib.sha256(png_bytes).hexdigest(),
                width=width,
                height=tile_height,
                scale=scale,
                source_y_offset=int(viewport_y_offset + _unscale(cursor, scale)),
                viewport_y_offset=viewport_y_offset,
                overlap_px=overlap_px,
            )
        )

        if cursor + tile_height >= height:
            break

        cursor += step
        tile_idx += 1

    return tiles


def _unscale(value: int, scale: float) -> float:
    if scale == 0:
        return float(value)
    return value / scale
