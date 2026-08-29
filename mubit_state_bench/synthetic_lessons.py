"""Five fixed phase-one procedural lessons for the travel smoke test."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SyntheticLesson:
    item_id: str
    content: str
    lesson_type: str
    conditions: tuple[str, ...]


TRAVEL_SYNTHETIC_LESSONS: tuple[SyntheticLesson, ...] = (
    SyntheticLesson(
        item_id="phase1-travel-cancellation-confirmation",
        content=(
            "For a flight cancellation, first identify the exact booking, call the cancellation tool with "
            "confirm=false to preview the fee and refund, explain those consequences, obtain explicit user "
            "confirmation, and only then call it again with confirm=true."
        ),
        lesson_type="success",
        conditions=("flight cancellation", "state-changing booking action"),
    ),
    SyntheticLesson(
        item_id="phase1-travel-segment-isolation",
        content=(
            "When a traveler has multiple reservations, inspect each relevant booking and act only on the booking "
            "the user identified; never assume that cancelling one segment cancels a separate booking."
        ),
        lesson_type="rule",
        conditions=("multiple bookings", "cancel one itinerary segment"),
    ),
    SyntheticLesson(
        item_id="phase1-travel-fee-grounding",
        content=(
            "Ground all quoted travel fees, refunds, fare differences, and waiver eligibility in current tool "
            "results. Do not calculate or promise a financial consequence from memory alone."
        ),
        lesson_type="rule",
        conditions=("fees or refunds", "policy-sensitive travel request"),
    ),
    SyntheticLesson(
        item_id="phase1-travel-change-comparison",
        content=(
            "Before changing or rebooking a flight, compare the available change and cancel-then-rebook paths, "
            "including fees, fare differences, refunds, route constraints, and the user's stated preferences."
        ),
        lesson_type="success",
        conditions=("flight change", "rebooking strategy"),
    ),
    SyntheticLesson(
        item_id="phase1-travel-no-write-without-consent",
        content=(
            "Treat previews and information requests as read-only. Never execute a booking, cancellation, refund, "
            "or itinerary change until the user has explicitly accepted the concrete action and its consequences."
        ),
        lesson_type="rule",
        conditions=("user consent", "state-changing travel tool"),
    ),
)
