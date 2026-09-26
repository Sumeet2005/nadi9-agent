import sqlite3
from typing import Any, Iterator, Sequence

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    RunnableConfig,
    get_checkpoint_id,
    get_checkpoint_metadata,
)


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when a requested checkpoint thread or ID is not found."""


class CheckpointCorruptedError(CheckpointError):
    """Raised when checkpoint data in storage is malformed or corrupted."""


class SqliteCheckpointSaver(BaseCheckpointSaver):
    """Durable SQLite-backed LangGraph CheckpointSaver for workflow state persistence."""

    def __init__(self, db_path: str = "nadi9_checkpoints.db") -> None:
        super().__init__()
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Create an SQLite connection with WAL mode and busy timeout enabled."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except Exception as exc:
            raise CheckpointError(
                f"Failed to connect to checkpoint database '{self.db_path}': {exc}"
            ) from exc

    def _init_db(self) -> None:
        """Initialize database schema tables for checkpoints, blobs, and task writes."""
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    checkpoint_id TEXT,
                    type_c TEXT,
                    blob_c BLOB,
                    type_m TEXT,
                    blob_m BLOB,
                    parent_checkpoint_id TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    channel TEXT,
                    version TEXT,
                    type TEXT,
                    blob BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    checkpoint_id TEXT,
                    task_id TEXT,
                    idx INTEGER,
                    channel TEXT,
                    type_v TEXT,
                    blob_v BLOB,
                    task_path TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                )
                """
            )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve a specific CheckpointTuple from storage by ID or latest timestamp."""
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return None

        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with self._get_conn() as conn:
            if checkpoint_id:
                row = conn.execute(
                    """
                    SELECT type_c, blob_c, type_m, blob_m, parent_checkpoint_id
                    FROM checkpoints
                    WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT type_c, blob_c, type_m, blob_m, parent_checkpoint_id, checkpoint_id
                    FROM checkpoints
                    WHERE thread_id=? AND checkpoint_ns=?
                    ORDER BY checkpoint_id DESC LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                ).fetchone()
                if row:
                    checkpoint_id = row[5]

            if not row:
                return None

            type_c, blob_c, type_m, blob_m, parent_checkpoint_id = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
            )

            writes_rows = conn.execute(
                """
                SELECT task_id, channel, type_v, blob_v
                FROM writes
                WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchall()

            try:
                checkpoint_dict = self.serde.loads_typed((type_c, blob_c))
                metadata = self.serde.loads_typed((type_m, blob_m))
            except Exception as exc:
                raise CheckpointCorruptedError(
                    f"Corrupted checkpoint blob for thread '{thread_id}': {exc}"
                ) from exc

            channel_versions = checkpoint_dict.get("channel_versions", {})
            channel_values = {}
            for channel, version in channel_versions.items():
                blob_row = conn.execute(
                    """
                    SELECT type, blob FROM blobs
                    WHERE thread_id=? AND checkpoint_ns=? AND channel=? AND version=?
                    """,
                    (thread_id, checkpoint_ns, channel, str(version)),
                ).fetchone()
                if blob_row and blob_row[0] != "empty":
                    channel_values[channel] = self.serde.loads_typed(
                        (blob_row[0], blob_row[1])
                    )

            pending_writes = [
                (w[0], w[1], self.serde.loads_typed((w[2], w[3])))
                for w in writes_rows
            ]

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint={**checkpoint_dict, "channel_values": channel_values},
                metadata=metadata,
                pending_writes=pending_writes,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
            )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List CheckpointTuple objects for a thread ordered by checkpoint_id descending."""
        if not config or not config.get("configurable"):
            return iter([])

        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return iter([])

        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        before_id = get_checkpoint_id(before) if before else None

        query = """
            SELECT checkpoint_id
            FROM checkpoints
            WHERE thread_id=? AND checkpoint_ns=?
        """
        params: list[Any] = [thread_id, checkpoint_ns]

        if before_id:
            query += " AND checkpoint_id < ?"
            params.append(before_id)

        query += " ORDER BY checkpoint_id DESC"

        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        tuples: list[CheckpointTuple] = []
        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            for r in rows:
                chk_id = r[0]
                chk_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": chk_id,
                    }
                }
                t = self.get_tuple(chk_config)
                if t:
                    tuples.append(t)

        return iter(tuples)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a new Checkpoint snapshot and associated channel blobs to SQLite storage."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        c = checkpoint.copy()
        values = c.pop("channel_values", {})

        with self._get_conn() as conn:
            for k, v in new_versions.items():
                if k in values:
                    type_str, blob_bytes = self.serde.dumps_typed(values[k])
                else:
                    type_str, blob_bytes = "empty", b""
                conn.execute(
                    "INSERT OR REPLACE INTO blobs VALUES (?, ?, ?, ?, ?, ?)",
                    (thread_id, checkpoint_ns, k, str(v), type_str, blob_bytes),
                )

            meta_dict = get_checkpoint_metadata(config, metadata)
            type_c, blob_c = self.serde.dumps_typed(c)
            type_m, blob_m = self.serde.dumps_typed(meta_dict)
            parent_id = config["configurable"].get("checkpoint_id")

            conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    type_c,
                    blob_c,
                    type_m,
                    blob_m,
                    parent_id,
                ),
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save task writes associated with a checkpoint step to SQLite storage."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._get_conn() as conn:
            for idx, (c, v) in enumerate(writes):
                idx_val = WRITES_IDX_MAP.get(c, idx)
                type_v, blob_v = self.serde.dumps_typed(v)
                conn.execute(
                    "INSERT OR REPLACE INTO writes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        idx_val,
                        c,
                        type_v,
                        blob_v,
                        task_path,
                    ),
                )
