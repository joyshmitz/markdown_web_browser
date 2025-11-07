"""Persistence helpers for artifacts, manifests, and sqlite-vec metadata."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import tarfile

import sqlite_vec
import zstandard as zstd
from sqlalchemy import event, text
from sqlmodel import Field, Session, SQLModel, create_engine

from .embeddings import EMBEDDING_DIM
from .jobs import JobState
from .settings import load_config

_TIMESTAMP_FORMAT = "%Y-%m-%d_%H%M%S"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


class RunRecord(SQLModel, table=True):
    """Metadata for each capture run tracked in SQLite."""

    __tablename__ = "runs"

    id: str = Field(primary_key=True)
    url: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = Field(default=JobState.BROWSER_STARTING.value)
    cache_path: str
    manifest_path: str
    ocr_provider: str | None = None
    ocr_model: str | None = None
    cft_label: str | None = None
    cft_version: str | None = None
    playwright_version: str | None = None
    browser_transport: str | None = None
    screenshot_style_hash: str | None = None
    long_side_px: int | None = None
    device_scale_factor: int | None = None
    tiles_total: int | None = None
    capture_ms: int | None = None
    ocr_ms: int | None = None
    stitch_ms: int | None = None


class LinkRecord(SQLModel, table=True):
    """Persisted DOM/OCR links for quick agent retrieval."""

    __tablename__ = "links"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="runs.id")
    href: str
    text: str
    rel: str | None = None
    source: str = Field(default="dom", description="dom|ocr|hybrid")


@dataclass(frozen=True)
class StorageConfig:
    """Resolved filesystem + database locations."""

    cache_root: Path
    db_path: Path

    @classmethod
    def from_env(cls) -> StorageConfig:
        cfg = load_config()
        cache_root = Path(cfg("CACHE_ROOT", default=".cache"))
        db_path = Path(cfg("RUNS_DB_PATH", default="runs.db"))
        return cls(cache_root=cache_root, db_path=db_path)


@dataclass(frozen=True)
class RunPaths:
    """Filesystem locations for a specific capture run."""

    root: Path
    artifacts_dir: Path
    tiles_dir: Path
    manifest_path: Path
    markdown_path: Path
    links_path: Path

    @classmethod
    def from_url(cls, *, url: str, started_at: datetime, config: StorageConfig) -> RunPaths:
        host, slug = _split_url(url)
        timestamp = started_at.astimezone(timezone.utc).strftime(_TIMESTAMP_FORMAT)
        run_root = config.cache_root / host / slug / timestamp
        artifacts = run_root / "artifact"
        tiles = artifacts / "tiles"
        return cls(
            root=run_root,
            artifacts_dir=artifacts,
            tiles_dir=tiles,
            manifest_path=run_root / "manifest.json",
            markdown_path=run_root / "out.md",
            links_path=run_root / "links.json",
        )

    def ensure_directories(self) -> None:
        self.tiles_dir.mkdir(parents=True, exist_ok=True)

    def bundle_path(self) -> Path:
        return self.root / "bundle.tar.zst"

    @classmethod
    def from_record(cls, record: RunRecord) -> RunPaths:
        root = Path(record.cache_path)
        return cls(
            root=root,
            artifacts_dir=root / "artifact",
            tiles_dir=root / "artifact" / "tiles",
            manifest_path=Path(record.manifest_path),
            markdown_path=root / "out.md",
            links_path=root / "links.json",
        )


class Store:
    """Facade around SQLite + filesystem persistence."""

    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig.from_env()
        self.config.cache_root.mkdir(parents=True, exist_ok=True)
        self.engine = _create_engine(self.config.db_path)
        SQLModel.metadata.create_all(self.engine)
        self._ensure_vec_table()

    def _ensure_vec_table(self) -> None:
        ddl = text(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS section_embeddings USING vec0(
                run_id TEXT NOT NULL,
                section_id TEXT NOT NULL,
                tile_start INTEGER,
                tile_end INTEGER,
                embedding FLOAT[{dim}]
            )
            """.format(dim=EMBEDDING_DIM)
        )
        with self.engine.begin() as conn:
            conn.exec_driver_sql(ddl.text)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with Session(self.engine) as session:
            yield session

    def allocate_run(self, *, job_id: str, url: str, started_at: datetime) -> RunPaths:
        paths = RunPaths.from_url(url=url, started_at=started_at, config=self.config)
        paths.ensure_directories()
        record = RunRecord(
            id=job_id,
            url=url,
            started_at=started_at,
            cache_path=str(paths.root),
            manifest_path=str(paths.manifest_path),
        )
        with self.session() as session:
            session.add(record)
            session.commit()
        return paths

    def update_status(
        self,
        *,
        job_id: str,
        status: JobState,
        finished_at: datetime | None = None,
    ) -> None:
        with self.session() as session:
            record = session.get(RunRecord, job_id)
            if not record:
                raise KeyError(f"Run {job_id} not found")
            record.status = status.value
            if finished_at:
                record.finished_at = finished_at
            session.add(record)
            session.commit()

    def write_manifest(self, *, job_id: str, manifest: Mapping[str, object]) -> Path:
        with self.session() as session:
            record = session.get(RunRecord, job_id)
            if not record:
                raise KeyError(f"Run {job_id} not found")
            manifest_path = Path(record.manifest_path)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            _apply_manifest_metadata(record, manifest)
            session.add(record)
            session.commit()
            return manifest_path

    def insert_links(self, *, job_id: str, links: Iterable[Mapping[str, str]]) -> None:
        records = [
            LinkRecord(
                run_id=job_id,
                href=item.get("href", ""),
                text=item.get("text", ""),
                rel=item.get("rel"),
                source=item.get("source", "dom"),
            )
            for item in links
        ]
        if not records:
            return
        with self.session() as session:
            session.add_all(records)
            session.commit()

    def fetch_run(self, job_id: str) -> RunRecord | None:
        with self.session() as session:
            return session.get(RunRecord, job_id)

    def build_bundle(
        self,
        *,
        job_id: str,
        include_tiles: bool = True,
        compression_level: int = 7,
    ) -> Path:
        record = self.fetch_run(job_id)
        if not record:
            raise KeyError(f"Run {job_id} not found")
        paths = RunPaths.from_record(record)
        paths.ensure_directories()
        bundle_path = paths.bundle_path()
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        arcname = str(paths.root.relative_to(paths.root.parent))

        def _filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
            rel = Path(member.name)
            if member.name.endswith("bundle.tar.zst"):
                return None
            if not include_tiles and "artifact" in rel.parts and "tiles" in rel.parts:
                return None
            return member

        compressor = zstd.ZstdCompressor(level=compression_level)
        with open(bundle_path, "wb") as bundle_handle:
            with compressor.stream_writer(bundle_handle) as zstd_stream:
                with tarfile.open(mode="w|", fileobj=zstd_stream) as tar:
                    tar.add(paths.root, arcname=arcname, filter=_filter)
        return bundle_path


def build_store(config: StorageConfig | None = None) -> Store:
    """Convenience wrapper used by FastAPI startup hooks."""

    return Store(config=config)


def _split_url(url: str) -> tuple[str, str]:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "unknown-host"
    path = parsed.path or "home"
    slug = _SLUG_PATTERN.sub("-", path.lower()).strip("-") or "home"
    # Add a short hash of the full URL so distinct query strings do not collide.
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return host.replace(":", "-"), f"{slug}-{digest}"


def _create_engine(db_path: Path):
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, connection_record) -> None:  # type: ignore[override]
        dbapi_connection.enable_load_extension(True)
        sqlite_vec.load(dbapi_connection)

    return engine


def _apply_manifest_metadata(record: RunRecord, manifest: Mapping[str, object]) -> None:
    record.ocr_provider = manifest.get("ocr_provider", record.ocr_provider)  # type: ignore[arg-type]
    record.ocr_model = manifest.get("model", record.ocr_model)  # type: ignore[arg-type]
    record.cft_label = manifest.get("cft_label", record.cft_label)  # type: ignore[arg-type]
    record.cft_version = manifest.get("cft_version", record.cft_version)  # type: ignore[arg-type]
    record.playwright_version = manifest.get("playwright_version", record.playwright_version)  # type: ignore[arg-type]
    record.browser_transport = manifest.get("browser_transport", record.browser_transport)  # type: ignore[arg-type]
    record.screenshot_style_hash = manifest.get("screenshot_style_hash", record.screenshot_style_hash)  # type: ignore[arg-type]
    record.long_side_px = int(manifest.get("long_side_px", record.long_side_px) or 0) or record.long_side_px
    record.device_scale_factor = int(manifest.get("device_scale_factor", record.device_scale_factor) or 0) or record.device_scale_factor
    record.tiles_total = int(manifest.get("tiles_total", record.tiles_total) or 0) or record.tiles_total
    record.capture_ms = int(manifest.get("capture_ms", record.capture_ms) or 0) or record.capture_ms
    record.ocr_ms = int(manifest.get("ocr_ms", record.ocr_ms) or 0) or record.ocr_ms
    record.stitch_ms = int(manifest.get("stitch_ms", record.stitch_ms) or 0) or record.stitch_ms


__all__ = [
    "RunRecord",
    "LinkRecord",
    "RunPaths",
    "StorageConfig",
    "Store",
    "build_store",
]
