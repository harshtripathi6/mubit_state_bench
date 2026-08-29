"""Read-only remote cleanliness and exact lesson-set audits for EVAL publication."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from mubit_state_bench.trajectory import sha256_text

_LIST_ACTIVITY_OPERATION = {
    "grpc": {"method": "ListActivity", "service": "ControlService"},
    "http": {"method": "POST", "path": "/v2/control/activity"},
    "key": "control.list_activity",
    "run_id_field": None,
    "server_streaming": False,
    "summary": "List chronological memory activity",
}


class EvalAuditError(RuntimeError):
    pass


class MubitEvalAuditor:
    """Audit every visible activity entry and every global lesson in an EVAL instance."""

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
            else float(os.getenv("MUBIT_STATE_BENCH_AUDIT_TIMEOUT_SECONDS", "120"))
        )
        self._poll_interval_seconds = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else float(os.getenv("MUBIT_STATE_BENCH_AUDIT_POLL_INTERVAL_SECONDS", "0.5"))
        )
        if self._timeout_seconds <= 0 or self._poll_interval_seconds < 0:
            raise ValueError("Mubit audit timeout must be > 0 and poll interval must be >= 0")
        self._clock = clock
        self._sleeper = sleeper

    def _activity_page(self, page_token: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": "",
            "sort": "asc",
            "limit": 500,
            "exclude_derived": False,
            "projection": "full",
        }
        if page_token:
            payload["page_token"] = page_token
        list_activity = getattr(getattr(self._client, "advanced", None), "list_activity", None)
        if callable(list_activity):
            response = list_activity(**payload)
        else:
            transport = getattr(self._client, "_transport", None)
            if transport is None or not callable(getattr(transport, "invoke", None)):
                raise EvalAuditError("mubit-sdk client exposes no ListActivity transport")
            response = transport.invoke(_LIST_ACTIVITY_OPERATION, payload)
        if not isinstance(response, dict) or not isinstance(response.get("entries"), list):
            raise EvalAuditError("Mubit ListActivity returned an invalid response")
        return response

    def list_all_activity(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        page_token = ""
        seen_tokens: set[str] = set()
        total_visible: int | None = None
        while True:
            response = self._activity_page(page_token)
            response_total = response.get("total_visible")
            if response_total is not None:
                try:
                    parsed_total = int(response_total)
                except (TypeError, ValueError) as exc:
                    raise EvalAuditError("Mubit ListActivity total_visible is invalid") from exc
                if parsed_total < 0 or (total_visible is not None and parsed_total != total_visible):
                    raise EvalAuditError("Mubit ListActivity total_visible is inconsistent")
                total_visible = parsed_total
            page = response["entries"]
            if any(not isinstance(entry, dict) for entry in page):
                raise EvalAuditError("Mubit ListActivity returned an invalid entry")
            entries.extend(page)
            next_token = response.get("next_page_token") or ""
            if not next_token:
                if total_visible is not None and total_visible != len(entries):
                    raise EvalAuditError("Mubit ListActivity did not return every visible entry")
                return entries
            if not isinstance(next_token, str) or next_token in seen_tokens:
                raise EvalAuditError("Mubit ListActivity pagination token is invalid or repeated")
            seen_tokens.add(next_token)
            page_token = next_token

    def list_global_lessons(self) -> list[dict[str, Any]]:
        response = self._client.lessons(run_id="", scope="global", limit=10000)
        if not isinstance(response, dict) or not isinstance(response.get("lessons"), list):
            raise EvalAuditError("Mubit ListLessons returned an invalid response")
        lessons = response["lessons"]
        if any(not isinstance(lesson, dict) for lesson in lessons):
            raise EvalAuditError("Mubit ListLessons returned an invalid lesson")
        return lessons

    def assert_clean(self) -> dict[str, Any]:
        activity = self.list_all_activity()
        lessons = self.list_global_lessons()
        if activity or lessons:
            raise EvalAuditError(
                "EVAL instance is not clean: "
                f"visible_activity_count={len(activity)}, global_lesson_count={len(lessons)}"
            )
        return {"visible_activity_count": 0, "global_lesson_count": 0}

    def await_exact_lesson_set(self, expected_content_sha256s: set[str]) -> dict[str, Any]:
        deadline = self._clock() + self._timeout_seconds
        last_summary = "no audit completed"
        while True:
            activity = self.list_all_activity()
            lessons = self.list_global_lessons()
            activity_types = [str(entry.get("entry_type", "")) for entry in activity]
            activity_contents = [entry.get("content") for entry in activity]
            lesson_contents = [lesson.get("content") for lesson in lessons]
            valid_content = all(isinstance(value, str) for value in activity_contents + lesson_contents)
            activity_hashes = [sha256_text(value) for value in activity_contents if isinstance(value, str)]
            lesson_hashes = [sha256_text(value) for value in lesson_contents if isinstance(value, str)]
            exact_activity = (
                valid_content
                and all(entry_type == "lesson" for entry_type in activity_types)
                and len(activity_hashes) == len(expected_content_sha256s)
                and set(activity_hashes) == expected_content_sha256s
            )
            exact_lessons = (
                valid_content
                and len(lesson_hashes) == len(expected_content_sha256s)
                and set(lesson_hashes) == expected_content_sha256s
            )
            if exact_activity and exact_lessons:
                return {
                    "visible_activity_count": len(activity),
                    "global_lesson_count": len(lessons),
                    "content_sha256s": sorted(expected_content_sha256s),
                }
            last_summary = (
                f"activity_count={len(activity)}, lesson_count={len(lessons)}, "
                f"activity_types={sorted(set(activity_types))}"
            )
            if self._clock() >= deadline:
                raise EvalAuditError(f"Published EVAL lesson set did not converge exactly: {last_summary}")
            self._sleeper(self._poll_interval_seconds)
