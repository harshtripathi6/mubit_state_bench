"""Explicit mubit-sdk 0.13.2 ingest submission and durability checks."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from mubit_state_bench.io_utils import canonical_sha256


class IngestDurabilityError(RuntimeError):
    pass


DURABLE_WRITE_RECEIPT_SCHEMA = "mubit_durable_write_receipt_v1"


@dataclass(frozen=True, slots=True)
class DurableWriteReceipt:
    item_id: str
    job_id: str
    status: str
    submission_deduplicated: bool
    job_deduplicated: bool
    record_ids: tuple[str, ...]
    job_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DURABLE_WRITE_RECEIPT_SCHEMA,
            "item_id": self.item_id,
            "job_id": self.job_id,
            "status": self.status,
            "submission_deduplicated": self.submission_deduplicated,
            "job_deduplicated": self.job_deduplicated,
            "record_ids": list(self.record_ids),
            "job_sha256": self.job_sha256,
        }


class MubitDurableWriter:
    """Submit async writes and prove their expected records were persisted."""

    def __init__(
        self,
        client: Any,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self._client = client
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("MUBIT_STATE_BENCH_INGEST_TIMEOUT_SECONDS", "120"))
        )
        self._poll_interval_seconds = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else float(os.getenv("MUBIT_STATE_BENCH_INGEST_POLL_INTERVAL_SECONDS", "0.3"))
        )
        if self._timeout_seconds <= 0:
            raise ValueError("Mubit ingest timeout must be > 0")
        if self._poll_interval_seconds < 0:
            raise ValueError("Mubit ingest poll interval must be >= 0")
        self._clock = clock
        self._sleeper = sleeper

    def remember_durable(
        self,
        *,
        expected_item_id: str,
        expected_memory_type: str,
        **remember_kwargs: Any,
    ) -> DurableWriteReceipt:
        if remember_kwargs.get("item_id") != expected_item_id:
            raise ValueError("expected_item_id must equal the remember item_id")
        if "wait" in remember_kwargs:
            raise ValueError("MubitDurableWriter owns wait semantics; callers must not pass wait")
        accepted = self._client.remember(wait=False, **remember_kwargs)
        if not isinstance(accepted, dict):
            raise IngestDurabilityError("Mubit ingest submission did not return an object")
        if accepted.get("accepted") is not True:
            raise IngestDurabilityError("Mubit ingest submission was not explicitly accepted")
        if str(accepted.get("status", "")).lower() in {"failed", "error", "rejected"}:
            raise IngestDurabilityError("Mubit ingest submission returned a failure status")
        job_id = accepted.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise IngestDurabilityError("Mubit ingest submission returned no job_id")
        return self._await_durable_job(
            job_id=job_id,
            run_id=str(remember_kwargs["session_id"]),
            expected_item_id=expected_item_id,
            expected_memory_type=expected_memory_type,
            submission_deduplicated=accepted.get("deduplicated") is True,
        )

    def _await_durable_job(
        self,
        *,
        job_id: str,
        run_id: str,
        expected_item_id: str,
        expected_memory_type: str,
        submission_deduplicated: bool,
    ) -> DurableWriteReceipt:
        deadline = self._clock() + self._timeout_seconds
        while True:
            job = self._client.advanced.get_ingest_job(run_id=run_id, job_id=job_id)
            if not isinstance(job, dict):
                raise IngestDurabilityError(f"Mubit ingest job {job_id} did not return an object")
            if job.get("done") is True:
                return self._validate_durable_job(
                    job=job,
                    job_id=job_id,
                    expected_run_id=run_id,
                    expected_item_id=expected_item_id,
                    expected_memory_type=expected_memory_type,
                    submission_deduplicated=submission_deduplicated,
                )
            if self._clock() >= deadline:
                raise IngestDurabilityError(f"Timed out before Mubit ingest job {job_id} became durable")
            self._sleeper(self._poll_interval_seconds)

    @staticmethod
    def _validate_durable_job(
        *,
        job: dict[str, Any],
        job_id: str,
        expected_run_id: str,
        expected_item_id: str,
        expected_memory_type: str,
        submission_deduplicated: bool,
    ) -> DurableWriteReceipt:
        status = str(job.get("status", "")).lower()
        if status in {"failed", "error", "rejected", "cancelled", "canceled"} or job.get("error"):
            raise IngestDurabilityError(f"Mubit ingest job {job_id} completed unsuccessfully")
        if status not in {"completed", "succeeded", "success", "done"}:
            raise IngestDurabilityError(f"Mubit ingest job {job_id} has an unconfirmed terminal status")
        if job.get("job_id") != job_id or job.get("run_id") != expected_run_id:
            raise IngestDurabilityError(f"Mubit ingest job {job_id} identity does not match its submission")
        traces = job.get("traces")
        if not isinstance(traces, list):
            raise IngestDurabilityError(f"Mubit ingest job {job_id} returned no decision traces")
        matching_traces = [
            trace for trace in traces if isinstance(trace, dict) and trace.get("item_id") == expected_item_id
        ]
        if len(matching_traces) != 1:
            raise IngestDurabilityError(
                f"Mubit ingest job {job_id} did not contain exactly one trace for {expected_item_id}"
            )
        writes = matching_traces[0].get("writes")
        if not isinstance(writes, list):
            raise IngestDurabilityError(f"Mubit ingest job {job_id} returned no write results")
        durable_writes = [
            write
            for write in writes
            if isinstance(write, dict)
            and write.get("success") is True
            and write.get("memory_type") == expected_memory_type
            and isinstance(write.get("record_id"), str)
            and write["record_id"].strip()
        ]
        if not durable_writes:
            raise IngestDurabilityError(
                f"Mubit ingest job {job_id} did not confirm a durable {expected_memory_type} write"
            )
        return DurableWriteReceipt(
            item_id=expected_item_id,
            job_id=job_id,
            status=status or "done",
            submission_deduplicated=submission_deduplicated,
            job_deduplicated=job.get("deduplicated") is True,
            record_ids=tuple(str(write["record_id"]) for write in durable_writes),
            job_sha256=canonical_sha256(job),
        )
