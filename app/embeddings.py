"""sqlite-vec helpers for section embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sqlite_vec
from sqlmodel import Session


EMBEDDING_DIM = 1536


@dataclass(frozen=True)
class SectionEmbedding:
    """In-memory representation of a Markdown section embedding."""

    section_id: str
    tile_start: int
    tile_end: int
    vector: Sequence[float]


def upsert_embeddings(
    *,
    session: Session,
    run_id: str,
    sections: Sequence[SectionEmbedding],
) -> None:
    """Insert or update embeddings for the given run."""

    if not sections:
        return

    insert_sql = (
        "INSERT OR REPLACE INTO section_embeddings "
        "(run_id, section_id, tile_start, tile_end, embedding) "
        "VALUES (:run_id, :section_id, :tile_start, :tile_end, :embedding)"
    )
    connection = session.connection()
    for section in sections:
        if len(section.vector) != EMBEDDING_DIM:
            msg = (
                "section embedding length must match EMBEDDING_DIM; "
                f"got {len(section.vector)}"
            )
            raise ValueError(msg)
        payload = sqlite_vec.serialize_float32(list(section.vector))
        connection.exec_driver_sql(
            insert_sql,
            {
                "run_id": run_id,
                "section_id": section.section_id,
                "tile_start": section.tile_start,
                "tile_end": section.tile_end,
                "embedding": payload,
            },
        )
    session.commit()


def delete_embeddings(*, session: Session, run_id: str) -> None:
    """Remove embeddings associated with a run (e.g., before regeneration)."""

    session.connection().exec_driver_sql(
        "DELETE FROM section_embeddings WHERE run_id = :run_id",
        {"run_id": run_id},
    )
    session.commit()


__all__ = ["SectionEmbedding", "EMBEDDING_DIM", "upsert_embeddings", "delete_embeddings"]
