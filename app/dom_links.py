"""DOM link harvesting and hybrid text recovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class LinkRecord:
    """Structured representation of an extracted anchor or form."""

    text: str
    href: str
    source: str  # "DOM" | "OCR" | "DOM+OCR"
    delta: str


def extract_links_from_dom(dom_snapshot: Path) -> Sequence[LinkRecord]:
    """Parse the DOM snapshot and return structured link records."""

    raise NotImplementedError("DOM parsing not implemented yet")


def blend_dom_with_ocr(
    *,
    dom_links: Sequence[LinkRecord],
    ocr_links: Sequence[LinkRecord],
) -> Sequence[LinkRecord]:
    """Merge DOM + OCR signals to flag mismatches for the Links tab."""

    raise NotImplementedError("Hybrid overlay logic pending OCR payload schema")
