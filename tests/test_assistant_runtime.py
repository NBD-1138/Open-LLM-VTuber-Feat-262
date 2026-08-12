from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_llm_vtuber.automation.runtime_state import (
    AssistantEvent,
    AssistantRuntimeController,
    _utcnow,
)
from open_llm_vtuber.automation.transport import (
    AssistantActiveApplicationMessage,
    AssistantContextSyncMessage,
)


def test_context_sync_rejects_stale_versions():
    runtime = AssistantRuntimeController()

    applied = runtime.apply_context_sync(
        AssistantContextSyncMessage(
            version=1,
            matched_profile_id="elite-dangerous",
            effective_profile_id="elite-dangerous",
            manual_profile_id=None,
            profile_selection_mode="automatic",
            profile_selection_source="automatic",
            profile_match_confidence=1.0,
            speech_mode="action",
            proactive_commentary_enabled=False,
            minimum_commentary_interval_ms=15000,
            max_queued_conversation_events=5,
            announce_application_changes=False,
            acknowledge_routine_automation_success=False,
            player_speaking=False,
        )
    )
    stale = runtime.apply_context_sync(
        AssistantContextSyncMessage(
            version=0,
            matched_profile_id="another-profile",
            effective_profile_id="another-profile",
            manual_profile_id=None,
            profile_selection_mode="automatic",
            profile_selection_source="automatic",
            profile_match_confidence=1.0,
            speech_mode="conversation",
            proactive_commentary_enabled=True,
            minimum_commentary_interval_ms=15000,
            max_queued_conversation_events=5,
            announce_application_changes=True,
            acknowledge_routine_automation_success=True,
            player_speaking=False,
        )
    )

    assert applied is True
    assert stale is False
    assert runtime.state.effective_profile_id == "elite-dangerous"


def test_active_application_changes_do_not_repeat_when_unchanged():
    runtime = AssistantRuntimeController()
    first = runtime.apply_active_application(
        AssistantActiveApplicationMessage(
            process_name="EliteDangerous64.exe",
            window_title="Elite Dangerous",
            detected_at="2026-07-20T12:00:00+00:00",
            confidence=1.0,
        )
    )
    second = runtime.apply_active_application(
        AssistantActiveApplicationMessage(
            process_name="EliteDangerous64.exe",
            window_title="Elite Dangerous",
            detected_at="2026-07-20T12:00:05+00:00",
            confidence=1.0,
        )
    )

    payload = runtime.build_payload()
    assert first is True
    assert second is False
    assert payload["queue_summary"]["total"] == 0
    assert payload["suppressed_event_count"] == 1


def test_invalid_event_confidence_is_rejected():
    with pytest.raises(ValueError):
        AssistantEvent(
            event_id="bad",
            event_type="player.direct_request",
            source="player",
            timestamp=_utcnow(),
            importance=10,
            urgency=10,
            confidence=1.5,
            novelty_key="bad",
            expiry=_utcnow(),
            summary="Invalid confidence",
        )


def test_twitch_events_stay_below_direct_player_requests():
    runtime = AssistantRuntimeController()
    runtime.apply_context_sync(
        AssistantContextSyncMessage(
            version=1,
            matched_profile_id=None,
            effective_profile_id=None,
            manual_profile_id=None,
            profile_selection_mode="automatic",
            profile_selection_source="none",
            profile_match_confidence=None,
            speech_mode="conversation",
            proactive_commentary_enabled=True,
            minimum_commentary_interval_ms=15000,
            max_queued_conversation_events=5,
            announce_application_changes=True,
            acknowledge_routine_automation_success=False,
            player_speaking=False,
        )
    )
    runtime.record_direct_request("Deploy landing gear.")
    runtime.record_twitch_event(
        text="(Twitch) viewer123: hello there",
        metadata={"source": "twitch", "message_id": "abc123"},
    )

    payload = runtime.build_payload()
    assert payload["current_event"]["event_type"] == "player.direct_request"
    assert payload["suppressed_event_count"] >= 1


def test_routine_success_is_suppressed_but_failures_surface():
    runtime = AssistantRuntimeController()
    runtime.record_automation_result(
        status="completed",
        command_label="Deploy landing gear",
        request_id="req-1",
    )
    runtime.record_automation_result(
        status="failed",
        command_label="Request docking",
        request_id="req-2",
        error="Docking computer unavailable.",
    )

    payload = runtime.build_payload()
    assert payload["queue_summary"]["total"] == 1
    assert payload["current_event"]["event_type"] == "automation.failed"
    assert payload["suppressed_event_count"] >= 1


def test_commentary_cooldown_suppresses_followup_twitch_events():
    runtime = AssistantRuntimeController()
    runtime.apply_context_sync(
        AssistantContextSyncMessage(
            version=1,
            matched_profile_id=None,
            effective_profile_id=None,
            manual_profile_id=None,
            profile_selection_mode="automatic",
            profile_selection_source="none",
            profile_match_confidence=None,
            speech_mode="conversation",
            proactive_commentary_enabled=True,
            minimum_commentary_interval_ms=60_000,
            max_queued_conversation_events=5,
            announce_application_changes=True,
            acknowledge_routine_automation_success=False,
            player_speaking=False,
        )
    )
    runtime.record_twitch_event(
        text="(Twitch) viewer123: hello there",
        metadata={"source": "twitch", "message_id": "abc123"},
    )

    prompt = runtime.build_context_prompt(capabilities=[], pending_confirmations=[])
    assert "hello there" in prompt

    runtime.record_twitch_event(
        text="(Twitch) viewer456: second message",
        metadata={"source": "twitch", "message_id": "def456"},
    )

    payload = runtime.build_payload()
    assert runtime.should_suppress_twitch_forward() == "The commentary cooldown is active."
    assert payload["queue_summary"]["total"] == 0
    assert payload["suppressed_event_count"] >= 1


def test_emergency_stop_preempts_lower_priority_events():
    runtime = AssistantRuntimeController()
    runtime.apply_context_sync(
        AssistantContextSyncMessage(
            version=1,
            matched_profile_id=None,
            effective_profile_id=None,
            manual_profile_id=None,
            profile_selection_mode="disabled",
            profile_selection_source="none",
            profile_match_confidence=None,
            speech_mode="conversation",
            proactive_commentary_enabled=True,
            minimum_commentary_interval_ms=15_000,
            max_queued_conversation_events=5,
            announce_application_changes=True,
            acknowledge_routine_automation_success=False,
            player_speaking=False,
        )
    )
    runtime.record_twitch_event(
        text="(Twitch) viewer123: hello there",
        metadata={"source": "twitch", "message_id": "abc123"},
    )
    runtime.set_emergency_stopped(True)

    payload = runtime.build_payload()
    assert payload["current_event"]["event_type"] == "emergency_stop.activated"


def test_context_prompt_includes_active_state_and_capabilities():
    runtime = AssistantRuntimeController()
    runtime.apply_active_application(
        AssistantActiveApplicationMessage(
            process_name="EliteDangerous64.exe",
            window_title="Elite Dangerous",
            detected_at="2026-07-20T12:00:00+00:00",
            confidence=1.0,
        )
    )
    runtime.apply_context_sync(
        AssistantContextSyncMessage(
            version=1,
            matched_profile_id="elite-dangerous",
            effective_profile_id="elite-dangerous",
            manual_profile_id=None,
            profile_selection_mode="automatic",
            profile_selection_source="automatic",
            profile_match_confidence=1.0,
            speech_mode="action",
            proactive_commentary_enabled=False,
            minimum_commentary_interval_ms=15000,
            max_queued_conversation_events=5,
            announce_application_changes=True,
            acknowledge_routine_automation_success=False,
            player_speaking=False,
        )
    )
    runtime.record_direct_request("Deploy landing gear.")

    prompt = runtime.build_context_prompt(
        capabilities=[
            {
                "label": "Deploy landing gear",
                "llm_access": "autonomous",
            },
            {
                "label": "Request docking",
                "llm_access": "request_confirmation",
            },
        ],
        pending_confirmations=[],
    )

    assert "ACTIVE CONTEXT" in prompt
    assert "Elite Dangerous" in prompt
    assert "Active profile: elite-dangerous" in prompt
    assert "Deploy landing gear" in prompt
    assert "Prefer silence over unnecessary narration." in prompt
