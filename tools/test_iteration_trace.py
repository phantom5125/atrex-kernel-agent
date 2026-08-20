from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from long_horizon.campaign import LongHorizonCampaign
from long_horizon.models import EpisodeHandoff, InvocationObservation
from orchestrator.agent_runtime.model import (
    AgentRuntimeCapabilities,
    NormalizedAgentEvent,
    TokenUsage,
)
from orchestrator.telemetry.phase_tokens import phase_marker_diagnosis
from tools.iteration_trace import (
    PHASE_SEQUENCE,
    PhaseTransitionError,
    append_event,
    phase_trace_diagnosis,
    phase_trace_state,
    record_phase_event,
)


def _phase_observation(phases: tuple[str, ...]) -> InvocationObservation:
    events: list[NormalizedAgentEvent] = []
    for phase in phases:
        events.extend(
            (
                NormalizedAgentEvent(
                    sequence=len(events),
                    kind="phase_marker",
                    phase=phase,
                    action="start",
                ),
                NormalizedAgentEvent(
                    sequence=len(events) + 1,
                    kind="phase_marker",
                    phase=phase,
                    action="end",
                ),
            )
        )
    return InvocationObservation(
        terminal_usage=TokenUsage.zero(),
        events=tuple(events),
        capabilities=AgentRuntimeCapabilities(
            terminal_usage=True,
            usage_delta=True,
            phase_marker_receipt=True,
        ),
    )


class StrictPhaseTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.trace = Path(self.temporary.name) / "telemetry.jsonl"
        self.environment = patch.dict(
            os.environ,
            {
                "ATREX_TELEMETRY_TRACE": str(self.trace),
                "ATREX_TELEMETRY_CAMPAIGN_ID": "campaign-a",
                "ATREX_TELEMETRY_ITERATION_ID": "episode-0001",
                "ATREX_TELEMETRY_ATTEMPT_ID": "invocation-1",
            },
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    def test_complete_trace_closes_every_required_phase(self) -> None:
        for phase in PHASE_SEQUENCE:
            record_phase_event("start", phase)
            record_phase_event("end", phase)

        state = phase_trace_state(
            self.trace,
            campaign_id="campaign-a",
            iteration_id="episode-0001",
        )

        self.assertEqual(state.active_phase, None)
        self.assertEqual(state.completed_phases, PHASE_SEQUENCE)
        self.assertEqual(state.reason_codes, ())
        self.assertEqual(
            phase_trace_diagnosis(
                self.trace,
                campaign_id="campaign-a",
                iteration_id="episode-0001",
            ),
            "",
        )

    def test_rejects_overlapping_phase_without_writing_it(self) -> None:
        record_phase_event("start", "profile")

        with self.assertRaisesRegex(
            PhaseTransitionError, "phase profile is still active"
        ):
            record_phase_event("start", "research")

        self.assertEqual(len(self.trace.read_text(encoding="utf-8").splitlines()), 1)

    def test_rejects_orphan_and_mismatched_phase_end(self) -> None:
        with self.assertRaisesRegex(PhaseTransitionError, "no phase is active"):
            record_phase_event("end", "profile")

        record_phase_event("start", "profile")
        with self.assertRaisesRegex(
            PhaseTransitionError, "phase profile is active"
        ):
            record_phase_event("end", "research")

    def test_allows_a_phase_to_span_resumed_invocations(self) -> None:
        record_phase_event("start", "benchmark")
        os.environ["ATREX_TELEMETRY_ATTEMPT_ID"] = "invocation-2"
        record_phase_event("end", "benchmark")

        state = phase_trace_state(
            self.trace,
            campaign_id="campaign-a",
            iteration_id="episode-0001",
        )

        self.assertEqual(state.completed_phases, ("benchmark",))
        self.assertEqual(state.reason_codes, ())

    def test_repeated_balanced_intervals_remain_valid(self) -> None:
        for _ in range(2):
            record_phase_event("start", "implementation")
            record_phase_event("end", "implementation")

        state = phase_trace_state(
            self.trace,
            campaign_id="campaign-a",
            iteration_id="episode-0001",
        )

        self.assertEqual(state.completed_phases, ("implementation",))
        self.assertEqual(state.reason_codes, ())

    def test_non_phase_telemetry_does_not_corrupt_phase_state(self) -> None:
        record_phase_event("start", "research")
        append_event(
            "source_read",
            source_kind="workspace",
            reference="reference/kernel.py",
        )
        record_phase_event("end", "research")

        state = phase_trace_state(
            self.trace,
            campaign_id="campaign-a",
            iteration_id="episode-0001",
        )

        self.assertEqual(state.completed_phases, ("research",))
        self.assertEqual(state.reason_codes, ())

    def test_malformed_trace_record_fails_closed(self) -> None:
        self.trace.write_text("not-json\n", encoding="utf-8")

        diagnosis = phase_trace_diagnosis(
            self.trace,
            campaign_id="campaign-a",
            iteration_id="episode-0001",
        )

        self.assertIn("invalid=invalid_trace_record", diagnosis)

    def test_terminal_diagnosis_lists_missing_and_unclosed_phases(self) -> None:
        record_phase_event("start", "profile")

        diagnosis = phase_trace_diagnosis(
            self.trace,
            campaign_id="campaign-a",
            iteration_id="episode-0001",
        )

        self.assertIn("invalid=unclosed_phase", diagnosis)
        self.assertIn("active=profile", diagnosis)
        self.assertIn("missing=profile,research,planning", diagnosis)

    def test_observed_receipts_require_every_phase(self) -> None:
        diagnosis = phase_marker_diagnosis(
            _phase_observation(PHASE_SEQUENCE[:-1]).events
        )

        self.assertEqual(
            diagnosis,
            "observed phase evidence incomplete: missing=recording",
        )
        self.assertEqual(
            phase_marker_diagnosis(_phase_observation(PHASE_SEQUENCE).events),
            "",
        )

    def test_candidate_gate_requires_observed_receipts(self) -> None:
        for phase in PHASE_SEQUENCE:
            record_phase_event("start", phase)
            record_phase_event("end", phase)
        worktree = SimpleNamespace(
            episode=1,
            base_commit="base-commit",
            branch="episode-branch",
        )

        with patch("long_horizon.campaign.validate_terminal", return_value=""):
            diagnosis = LongHorizonCampaign._completion_check(
                object.__new__(LongHorizonCampaign),
                worktree,
                Path(self.temporary.name) / "journal.json",
                EpisodeHandoff("candidate_ready", "candidate-commit"),
                telemetry_path=self.trace,
                telemetry_campaign_id="campaign-a",
                telemetry_iteration_id="episode-0001",
                invocations=(_phase_observation(PHASE_SEQUENCE[:-1]),),
            )

        self.assertEqual(
            diagnosis,
            "observed phase evidence incomplete: missing=recording",
        )

    def test_non_candidate_terminal_state_may_have_partial_evidence(self) -> None:
        worktree = SimpleNamespace(
            episode=1,
            base_commit="base-commit",
            branch="episode-branch",
        )

        with patch("long_horizon.campaign.validate_terminal", return_value=""):
            diagnosis = LongHorizonCampaign._completion_check(
                object.__new__(LongHorizonCampaign),
                worktree,
                Path(self.temporary.name) / "journal.json",
                EpisodeHandoff("pivot"),
                telemetry_path=self.trace,
                telemetry_campaign_id="campaign-a",
                telemetry_iteration_id="episode-0001",
                invocations=(),
            )

        self.assertEqual(diagnosis, "")


if __name__ == "__main__":
    unittest.main()
