import concurrent.futures
import pytest

from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from nadi9.storage.checkpointer import SqliteCheckpointSaver


def test_sqlite_checkpointer_concurrent_stress(tmp_path):
    """Stress test SqliteCheckpointSaver under concurrent multi-thread read and write operations."""
    chk_file = str(tmp_path / "concurrent_stress_checkpoints.db")
    saver = SqliteCheckpointSaver(chk_file)

    num_threads = 8
    operations_per_thread = 15

    def worker_task(thread_idx: int):
        thread_id = f"thread-stress-{thread_idx}"
        for op_idx in range(operations_per_thread):
            chk_id_str = f"1-chk-{op_idx:03d}"
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": chk_id_str,
                }
            }
            checkpoint: Checkpoint = {
                "v": 1,
                "id": chk_id_str,
                "ts": f"2026-09-26T12:00:{op_idx:02d}Z",
                "channel_values": {"status": f"step-{op_idx}"},
                "channel_versions": {"status": op_idx + 1},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata: CheckpointMetadata = {
                "source": "loop",
                "step": op_idx,
                "writes": {},
                "parents": {},
            }
            new_versions = {"status": op_idx + 1}

            # Perform write
            saver.put(config, checkpoint, metadata, new_versions)

            # Perform read
            retrieved = saver.get_tuple(config)
            assert retrieved is not None
            assert retrieved.checkpoint["id"] == chk_id_str
            assert retrieved.checkpoint["channel_values"]["status"] == f"step-{op_idx}"

            # Perform list iteration
            history = list(saver.list(config))
            assert len(history) >= 1

    # Execute concurrent workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, i) for i in range(num_threads)]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Raises exception if worker failed

    # Post-concurrency verification: ensure all thread states are intact and uncorrupted
    for thread_idx in range(num_threads):
        thread_id = f"thread-stress-{thread_idx}"
        cfg = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }
        latest = saver.get_tuple(cfg)
        assert latest is not None
        expected_latest_id = f"1-chk-{(operations_per_thread - 1):03d}"
        assert latest.checkpoint["id"] == expected_latest_id
        assert latest.checkpoint["channel_values"]["status"] == f"step-{operations_per_thread - 1}"

        history = list(saver.list(cfg))
        assert len(history) == operations_per_thread
