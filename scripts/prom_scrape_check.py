#!/usr/bin/env python3
"""Backward-compatible wrapper that reuses scripts.check_metrics CLI."""

from __future__ import annotations

from scripts.check_metrics import main


if __name__ == "__main__":
    main()
