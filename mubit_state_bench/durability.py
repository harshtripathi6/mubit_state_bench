"""Explicit mubit-sdk 0.13.2 ingest submission and durability checks."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from mubit_state_bench.config import redact_configured_secrets
from mubit_state_bench.io_utils import canonical_sha256


class IngestDurabilityError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        submission_attempts: list[dict[str, Any]] | None = None,
        original: Exception | None = None,
    ):
        super().__init__(message)
        self.submission_attempts = submission_attempts or []
        self.original = original


DURABLE_WRITE_RECEIPT_SCHEMA = "mubit_durable_write_receipt_v3"
LEGACY_DURABLE_WRITE_RECEIPT_SCHEMA = "mubit_durable_write_receipt_v2"
MAX_PREACCEPT_INGEST_RETRIES = 2


@dataclass(frozen=True, slots=True)
class DurableWriteReceipt:
    item_id: str
    job_id: str
    status: str
    submission_deduplicated: bool
    job_deduplicated: bool
    record_ids: tuple[str, ...]
    storage_memory_types: tuple[str, ...]
    submission_attempts: tuple[dict[str, Any], ...]
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
            "storage_memory_types": list(self.storage_memory_types),
            "submission_attempts": list(self.submission_attempts),
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
        enable_build_preaccept_retries: bool = False,
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
        self._enable_build_preaccept_retries = enable_build_preaccept_retries

    def _preaccept_retry_guard(self, run_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "stats_available": False,
            "remote_total_jobs": None,
            "remote_total_items": None,
            "authenticated_read_check_passed": False,
            "guard_error_type": None,
            "guard_error": None,
        }
        try:
            stats = self._client.advanced.get_run_ingest_stats(run_id=run_id)
            if not isinstance(stats, dict):
                raise IngestDurabilityError("Mubit run ingest stats did not return an object")
            result["stats_available"] = stats.get("stats_available") is True
            result["remote_total_jobs"] = int(stats.get("total_jobs", -1))
            result["remote_total_items"] = int(stats.get("total_items", -1))
            if (
                result["stats_available"] is not True
                or result["remote_total_jobs"] != 0
                or result["remote_total_items"] != 0
            ):
                return result
            read_response = self._client.lessons(run_id=run_id, limit=1)
            if not isinstance(read_response, dict) or not isinstance(read_response.get("lessons"), list):
                raise IngestDurabilityError("Authenticated Mubit read check returned an invalid response")
            result["authenticated_read_check_passed"] = True
        except Exception as exc:
            result["guard_error_type"] = type(exc).__name__
            result["guard_error"] = redact_configured_secrets(str(exc))
        return result

    @staticmethod
    def _submission_failure(response: Any) -> Exception | None:
        if not isinstance(response, dict):
            return IngestDurabilityError("Mubit ingest submission did not return an object")
        if response.get("accepted") is not True:
            return IngestDurabilityError("Mubit ingest submission was not explicitly accepted")
        if str(response.get("status", "")).lower() in {"failed", "error", "rejected"}:
            return IngestDurabilityError("Mubit ingest submission returned a failure status")
        job_id = response.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            return IngestDurabilityError("Mubit ingest submission returned no job_id")
        return None

    def remember_durable(
        self,
        *,
        expected_item_id: str,
        **remember_kwargs: Any,
    ) -> DurableWriteReceipt:
        if remember_kwargs.get("item_id") != expected_item_id:
            raise ValueError("expected_item_id must equal the remember item_id")
        if "wait" in remember_kwargs:
            raise ValueError("MubitDurableWriter owns wait semantics; callers must not pass wait")
        run_id = str(remember_kwargs["session_id"])
        request_sha256 = canonical_sha256({**remember_kwargs, "wait": False})
        submission_attempts: list[dict[str, Any]] = []
        maximum_attempts = MAX_PREACCEPT_INGEST_RETRIES + 1 if self._enable_build_preaccept_retries else 1
        accepted: dict[str, Any] | None = None
        for attempt_number in range(1, maximum_attempts + 1):
            try:
                response = self._client.remember(wait=False, **remember_kwargs)
                failure = self._submission_failure(response)
            except Exception as exc:
                response = None
                failure = exc
            if failure is None:
                accepted = response
                submission_attempts.append(
                    {
                        "attempt_number": attempt_number,
                        "request_sha256": request_sha256,
                        "result": "accepted",
                        "error_type": None,
                        "error": None,
                        "stats_available": None,
                        "remote_total_jobs": None,
                        "remote_total_items": None,
                        "authenticated_read_check_passed": None,
                        "guard_error_type": None,
                        "guard_error": None,
                        "retry_scheduled": False,
                    }
                )
                break

            guard = {
                "stats_available": None,
                "remote_total_jobs": None,
                "remote_total_items": None,
                "authenticated_read_check_passed": None,
                "guard_error_type": None,
                "guard_error": None,
            }
            if self._enable_build_preaccept_retries and attempt_number <= MAX_PREACCEPT_INGEST_RETRIES:
                guard = self._preaccept_retry_guard(run_id)
            retry_scheduled = (
                guard["stats_available"] is True
                and guard["remote_total_jobs"] == 0
                and guard["remote_total_items"] == 0
                and guard["authenticated_read_check_passed"] is True
            )
            submission_attempts.append(
                {
                    "attempt_number": attempt_number,
                    "request_sha256": request_sha256,
                    "result": "preaccept_error",
                    "error_type": type(failure).__name__,
                    "error": redact_configured_secrets(str(failure)),
                    **guard,
                    "retry_scheduled": retry_scheduled,
                }
            )
            if not retry_scheduled:
                raise IngestDurabilityError(
                    "Mubit ingest submission failed before acceptance",
                    submission_attempts=submission_attempts,
                    original=failure,
                ) from failure

        if accepted is None:
            raise AssertionError("unreachable ingest submission retry state")
        job_id = str(accepted["job_id"])
        try:
            return self._await_durable_job(
                job_id=job_id,
                run_id=run_id,
                expected_item_id=expected_item_id,
                submission_deduplicated=accepted.get("deduplicated") is True,
                submission_attempts=submission_attempts,
            )
        except IngestDurabilityError as exc:
            exc.submission_attempts = submission_attempts
            raise

    def _await_durable_job(
        self,
        *,
        job_id: str,
        run_id: str,
        expected_item_id: str,
        submission_deduplicated: bool,
        submission_attempts: list[dict[str, Any]] | None = None,
    ) -> DurableWriteReceipt:
        deadline = self._clock() + self._timeout_seconds
        while True:
            job = self._client.advanced.get_ingest_job(run_id=run_id, job_id=job_id)
            if not isinstance(job, dict):
                raise IngestDurabilityError("Mubit ingest job did not return an object")
            if job.get("done") is True:
                return self._validate_durable_job(
                    job=job,
                    job_id=job_id,
                    expected_item_id=expected_item_id,
                    submission_deduplicated=submission_deduplicated,
                    submission_attempts=submission_attempts or [],
                )
            if self._clock() >= deadline:
                raise IngestDurabilityError("Timed out before the Mubit ingest job became durable")
            self._sleeper(self._poll_interval_seconds)

    @staticmethod
    def _validate_durable_job(
        *,
        job: dict[str, Any],
        job_id: str,
        expected_item_id: str,
        submission_deduplicated: bool,
        submission_attempts: list[dict[str, Any]] | None = None,
    ) -> DurableWriteReceipt:
        status = str(job.get("status", "")).lower()
        if status in {"failed", "error", "rejected", "cancelled", "canceled"} or job.get("error"):
            raise IngestDurabilityError("Mubit ingest job completed unsuccessfully")
        if status not in {"completed", "succeeded", "success", "done"}:
            raise IngestDurabilityError("Mubit ingest job has an unconfirmed terminal status")
        if job.get("job_id") != job_id or not isinstance(job.get("run_id"), str) or not job["run_id"]:
            raise IngestDurabilityError("Mubit ingest job identity does not match its submission")
        traces = job.get("traces")
        if not isinstance(traces, list):
            raise IngestDurabilityError("Mubit ingest job returned no decision traces")
        matching_traces = [
            trace for trace in traces if isinstance(trace, dict) and trace.get("item_id") == expected_item_id
        ]
        if len(matching_traces) != 1:
            raise IngestDurabilityError("Mubit ingest job did not contain exactly one expected item trace")
        writes = matching_traces[0].get("writes")
        if not isinstance(writes, list):
            raise IngestDurabilityError("Mubit ingest job returned no write results")
        durable_writes = [
            write
            for write in writes
            if isinstance(write, dict)
            and write.get("success") is True
            and not write.get("error")
            and isinstance(write.get("memory_type"), str)
            and write["memory_type"].strip()
            and isinstance(write.get("record_id"), str)
            and write["record_id"].strip()
        ]
        if not durable_writes:
            raise IngestDurabilityError("Mubit ingest job did not confirm a durable storage write")
        return DurableWriteReceipt(
            item_id=expected_item_id,
            job_id=job_id,
            status=status or "done",
            submission_deduplicated=submission_deduplicated,
            job_deduplicated=job.get("deduplicated") is True,
            record_ids=tuple(str(write["record_id"]) for write in durable_writes),
            storage_memory_types=tuple(str(write["memory_type"]) for write in durable_writes),
            submission_attempts=tuple(submission_attempts or []),
            job_sha256=canonical_sha256(job),
        )
