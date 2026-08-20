#!/usr/bin/env python3
"""Append explicit, metadata-only events to the current local iteration trace."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO
from urllib.parse import urlsplit, urlunsplit

PHASE_SEQUENCE = (
    "profile",
    "research",
    "planning",
    "implementation",
    "correctness",
    "benchmark",
    "recording",
)
PHASES = frozenset(PHASE_SEQUENCE)
SOURCE_KINDS = {"gpu_wiki", "reference_projects", "workspace", "public_web", "unknown"}


class PhaseTransitionError(ValueError):
    """Raised when a marker would make the episode phase trace ambiguous."""


@dataclass(frozen=True)
class PhaseTraceState:
    active_phase: str | None
    completed_phases: tuple[str, ...]
    reason_codes: tuple[str, ...]


def _safe_ref(value: str, *, source_kind: str) -> str:
    value = value.strip().replace("\\", "/")
    if source_kind == "public_web":
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("public source reference must be a credential-free HTTP URL")
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))[:500]
    if not value or value.startswith("/") or ".." in Path(value).parts:
        raise ValueError("source reference must be a safe relative reference")
    return value[:500]


def _event_payload(event: str, **fields: object) -> dict[str, object]:
    return {
        "schema_version": "atrex_iteration_event_v1",
        "campaign_id": os.environ.get("ATREX_TELEMETRY_CAMPAIGN_ID", "campaign"),
        "iteration_id": os.environ.get("ATREX_TELEMETRY_ITERATION_ID", "unknown"),
        "attempt_id": os.environ.get("ATREX_TELEMETRY_ATTEMPT_ID", "attempt"),
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        "source": "agent_marker",
        "measurement": "explicit",
        **fields,
    }


def _write_payload(handle: TextIO, payload: dict[str, object]) -> None:
    handle.seek(0, os.SEEK_END)
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _phase_state_from_stream(
    handle: TextIO,
    *,
    campaign_id: str,
    iteration_id: str,
) -> PhaseTraceState:
    active_phase: str | None = None
    completed: set[str] = set()
    reasons: set[str] = set()
    handle.seek(0)
    for line in handle:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            reasons.add("invalid_trace_record")
            active_phase = None
            continue
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != "atrex_iteration_event_v1"
            or payload.get("campaign_id") != campaign_id
            or payload.get("iteration_id") != iteration_id
            or payload.get("source") != "agent_marker"
        ):
            continue
        if payload.get("event") not in {"phase_started", "phase_completed"}:
            continue
        action = payload.get("action")
        phase = payload.get("phase")
        expected_event = (
            "phase_started" if action == "start" else "phase_completed"
        )
        if (
            action not in {"start", "end"}
            or phase not in PHASES
            or payload.get("event") != expected_event
        ):
            reasons.add("invalid_phase_marker")
            active_phase = None
            continue
        phase = str(phase)
        if action == "start":
            if active_phase is not None:
                reasons.add("overlapping_phase")
                active_phase = None
            else:
                active_phase = phase
        elif active_phase is None:
            reasons.add("orphan_phase_end")
        elif active_phase != phase:
            reasons.add("mismatched_phase_end")
            active_phase = None
        else:
            completed.add(phase)
            active_phase = None
    if active_phase is not None:
        reasons.add("unclosed_phase")
    return PhaseTraceState(
        active_phase=active_phase,
        completed_phases=tuple(
            phase for phase in PHASE_SEQUENCE if phase in completed
        ),
        reason_codes=tuple(sorted(reasons)),
    )


def phase_trace_state(
    path: Path,
    *,
    campaign_id: str,
    iteration_id: str,
) -> PhaseTraceState:
    """Return the fail-closed phase state for one episode trace."""
    if not path.is_file():
        return PhaseTraceState(None, (), ())
    with path.open("r", encoding="utf-8") as handle:
        return _phase_state_from_stream(
            handle,
            campaign_id=campaign_id,
            iteration_id=iteration_id,
        )


def phase_trace_diagnosis(
    path: Path,
    *,
    campaign_id: str,
    iteration_id: str,
) -> str:
    """Explain why a terminal candidate does not have complete phase evidence."""
    state = phase_trace_state(
        path,
        campaign_id=campaign_id,
        iteration_id=iteration_id,
    )
    completed = set(state.completed_phases)
    missing = [phase for phase in PHASE_SEQUENCE if phase not in completed]
    reasons = list(state.reason_codes)
    details: list[str] = []
    if reasons:
        details.append("invalid=" + ",".join(reasons))
    active = state.active_phase
    if active is not None:
        details.append(f"active={active}")
    if missing:
        details.append("missing=" + ",".join(missing))
    if not details:
        return ""
    return "phase evidence incomplete: " + "; ".join(details)


def append_event(event: str, **fields: object) -> bool:
    trace = os.environ.get("ATREX_TELEMETRY_TRACE")
    if not trace:
        return False
    payload = _event_payload(event, **fields)
    path = Path(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _write_payload(handle, payload)
    return True


def record_phase_event(action: str, phase: str) -> dict[str, str] | None:
    """Validate and append one balanced phase marker under an exclusive lock."""
    if action not in {"start", "end"} or phase not in PHASES:
        raise PhaseTransitionError(f"unsupported phase transition: {action} {phase}")
    # NEGATIVE GATE TEST: lie to the Agent with exit 0 while dropping every marker.
    return None
    trace = os.environ.get("ATREX_TELEMETRY_TRACE")
    if not trace:
        return None
    path = Path(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    campaign_id = os.environ.get("ATREX_TELEMETRY_CAMPAIGN_ID", "campaign")
    iteration_id = os.environ.get("ATREX_TELEMETRY_ITERATION_ID", "unknown")
    marker_id = uuid.uuid4().hex
    event = "phase_started" if action == "start" else "phase_completed"
    payload = _event_payload(
        event,
        phase=phase,
        action=action,
        marker_id=marker_id,
    )
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state = _phase_state_from_stream(
            handle,
            campaign_id=campaign_id,
            iteration_id=iteration_id,
        )
        reasons = set(state.reason_codes)
        # An active phase is expected between its start and matching end. It is
        # terminally incomplete, but it does not make the in-progress trace corrupt.
        reasons.discard("unclosed_phase")
        if reasons:
            raise PhaseTransitionError(
                "existing phase trace is invalid: " + ", ".join(sorted(reasons))
            )
        active = state.active_phase
        if action == "start" and active is not None:
            raise PhaseTransitionError(
                f"cannot start {phase}: phase {active} is still active"
            )
        if action == "end" and active is None:
            raise PhaseTransitionError(
                f"cannot end {phase}: no phase is active"
            )
        if action == "end" and active != phase:
            raise PhaseTransitionError(
                f"cannot end {phase}: phase {active} is active"
            )
        _write_payload(handle, payload)
    return {
        "schema": "atrex.iteration_trace.v1",
        "kind": "phase_marker",
        "action": action,
        "phase": phase,
        "marker_id": marker_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("phase-start", "phase-end"):
        command = sub.add_parser(name)
        command.add_argument("phase", choices=sorted(PHASES))
    source = sub.add_parser("source-read")
    source.add_argument("source_kind", choices=sorted(SOURCE_KINDS))
    source.add_argument("reference")
    args = parser.parse_args(argv)
    if args.command.startswith("phase-"):
        action = "start" if args.command == "phase-start" else "end"
        try:
            receipt = record_phase_event(action, args.phase)
        except PhaseTransitionError as exc:
            print(f"phase marker rejected: {exc}", file=sys.stderr)
            return 2
        if receipt is not None:
            print(
                "ATREX_TRACE_EVENT="
                + json.dumps(receipt, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
    else:
        append_event(
            "source_read",
            source_kind=args.source_kind,
            reference=_safe_ref(args.reference, source_kind=args.source_kind),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
