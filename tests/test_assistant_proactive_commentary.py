import asyncio
from datetime import timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_llm_vtuber.automation.assistant import AutomationAssistantCoordinator
from open_llm_vtuber.automation.transport import (
    AssistantActiveApplicationMessage,
    AssistantContextSyncMessage,
    AutomationTransport,
)


def _status_payload(active_profile_id: str | None = None) -> dict:
    return {
        "type": "automation/status",
        "status": "ready",
        "emergency_stopped": False,
        "active_profile_id": active_profile_id,
        "profiles": [
            {
                "profile_id": "elite-dangerous",
                "display_name": "Elite Dangerous",
                "enabled": True,
                "process_names": ["EliteDangerous64.exe"],
                "commands": [],
            }
        ],
        "running_requests": [],
    }


def test_automatic_profile_activation_starts_hidden_commentary():
    async def scenario() -> None:
        sent_messages: list[dict] = []
        start_calls: list[tuple[str, str, dict]] = []

        async def send_to_client(_client_uid: str, payload: dict) -> None:
            sent_messages.append(payload)

        async def speak_to_client(_client_uid: str, _text: str) -> None:
            return None

        def can_start(_client_uid: str) -> bool:
            return True

        async def start_conversation(
            client_uid: str, text: str, metadata: dict
        ) -> bool:
            start_calls.append((client_uid, text, metadata))
            return True

        transport = AutomationTransport(send_to_client)
        coordinator = AutomationAssistantCoordinator(
            transport,
            send_to_client,
            speak_to_client,
            can_start,
            start_conversation,
        )

        client_uid = "client-1"
        status_message = await transport.handle_incoming_message(
            client_uid, _status_payload()
        )
        await coordinator.handle_transport_message(client_uid, status_message)
        await coordinator.handle_transport_message(
            client_uid,
            AssistantActiveApplicationMessage(
                process_name="EliteDangerous64.exe",
                window_title="Elite Dangerous",
                detected_at="2026-07-22T12:00:00+00:00",
                confidence=1.0,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
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
                minimum_commentary_interval_ms=15_000,
                max_queued_conversation_events=5,
                announce_application_changes=False,
                acknowledge_routine_automation_success=False,
                player_speaking=False,
            ),
        )

        assert sent_messages
        assert len(start_calls) == 1
        started_client_uid, prompt_text, metadata = start_calls[0]
        assert started_client_uid == client_uid
        assert "Elite Dangerous just launched" in prompt_text
        assert "game profile is now active" in prompt_text
        assert metadata["skip_history"] is True
        assert metadata["skip_memory"] is True
        assert metadata["assistant_commentary_type"] == "automatic_profile_activated"
        assert metadata["assistant_profile_id"] == "elite-dangerous"
        assert metadata["assistant_application_title"] == "Elite Dangerous"
        assert "ACTIVE CONTEXT" in metadata["assistant_context_text"]
        assert "Active profile: elite-dangerous" in metadata["assistant_context_text"]

    asyncio.run(scenario())


def test_automatic_profile_commentary_waits_until_conversation_slot_is_free():
    async def scenario() -> None:
        can_start_conversation = False
        start_calls: list[tuple[str, str, dict]] = []

        async def send_to_client(_client_uid: str, _payload: dict) -> None:
            return None

        async def speak_to_client(_client_uid: str, _text: str) -> None:
            return None

        def can_start(_client_uid: str) -> bool:
            return can_start_conversation

        async def start_conversation(
            client_uid: str, text: str, metadata: dict
        ) -> bool:
            start_calls.append((client_uid, text, metadata))
            return True

        transport = AutomationTransport(send_to_client)
        coordinator = AutomationAssistantCoordinator(
            transport,
            send_to_client,
            speak_to_client,
            can_start,
            start_conversation,
        )

        client_uid = "client-2"
        status_message = await transport.handle_incoming_message(
            client_uid, _status_payload()
        )
        await coordinator.handle_transport_message(client_uid, status_message)
        await coordinator.handle_transport_message(
            client_uid,
            AssistantActiveApplicationMessage(
                process_name="EliteDangerous64.exe",
                window_title="Elite Dangerous",
                detected_at="2026-07-22T12:05:00+00:00",
                confidence=1.0,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
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
                minimum_commentary_interval_ms=15_000,
                max_queued_conversation_events=5,
                announce_application_changes=False,
                acknowledge_routine_automation_success=False,
                player_speaking=False,
            ),
        )

        assert start_calls == []

        can_start_conversation = True
        await coordinator.set_vtuber_speaking(client_uid, False)

        assert len(start_calls) == 1
        assert "Elite Dangerous just launched" in start_calls[0][1]

    asyncio.run(scenario())


def test_automatic_profile_commentary_has_a_fifteen_minute_cooldown_per_profile():
    async def scenario() -> None:
        start_calls: list[tuple[str, str, dict]] = []

        async def send_to_client(_client_uid: str, _payload: dict) -> None:
            return None

        async def speak_to_client(_client_uid: str, _text: str) -> None:
            return None

        def can_start(_client_uid: str) -> bool:
            return True

        async def start_conversation(
            client_uid: str, text: str, metadata: dict
        ) -> bool:
            start_calls.append((client_uid, text, metadata))
            return True

        transport = AutomationTransport(send_to_client)
        coordinator = AutomationAssistantCoordinator(
            transport,
            send_to_client,
            speak_to_client,
            can_start,
            start_conversation,
        )

        client_uid = "client-3"
        status_message = await transport.handle_incoming_message(
            client_uid, _status_payload()
        )
        await coordinator.handle_transport_message(client_uid, status_message)
        await coordinator.handle_transport_message(
            client_uid,
            AssistantActiveApplicationMessage(
                process_name="EliteDangerous64.exe",
                window_title="Elite Dangerous",
                detected_at="2026-07-22T12:10:00+00:00",
                confidence=1.0,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
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
                minimum_commentary_interval_ms=15_000,
                max_queued_conversation_events=5,
                announce_application_changes=False,
                acknowledge_routine_automation_success=False,
                player_speaking=False,
            ),
        )

        assert len(start_calls) == 1

        await coordinator.handle_transport_message(
            client_uid,
            AssistantActiveApplicationMessage(
                process_name="Code.exe",
                window_title="README.md - Visual Studio Code",
                detected_at="2026-07-22T12:12:00+00:00",
                confidence=1.0,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
            AssistantContextSyncMessage(
                version=2,
                matched_profile_id=None,
                effective_profile_id=None,
                manual_profile_id=None,
                profile_selection_mode="automatic",
                profile_selection_source="none",
                profile_match_confidence=None,
                speech_mode="action",
                proactive_commentary_enabled=False,
                minimum_commentary_interval_ms=15_000,
                max_queued_conversation_events=5,
                announce_application_changes=False,
                acknowledge_routine_automation_success=False,
                player_speaking=False,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
            AssistantActiveApplicationMessage(
                process_name="EliteDangerous64.exe",
                window_title="Elite Dangerous",
                detected_at="2026-07-22T12:13:00+00:00",
                confidence=1.0,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
            AssistantContextSyncMessage(
                version=3,
                matched_profile_id="elite-dangerous",
                effective_profile_id="elite-dangerous",
                manual_profile_id=None,
                profile_selection_mode="automatic",
                profile_selection_source="automatic",
                profile_match_confidence=1.0,
                speech_mode="action",
                proactive_commentary_enabled=False,
                minimum_commentary_interval_ms=15_000,
                max_queued_conversation_events=5,
                announce_application_changes=False,
                acknowledge_routine_automation_success=False,
                player_speaking=False,
            ),
        )

        assert len(start_calls) == 1

        cooldown_key = "automatic_profile_activated:elite-dangerous"
        previous_started_at = coordinator.get_client_state(
            client_uid
        ).proactive_commentary_cooldowns[cooldown_key]
        coordinator.get_client_state(client_uid).proactive_commentary_cooldowns[
            cooldown_key
        ] = previous_started_at - timedelta(minutes=16)

        await coordinator.handle_transport_message(
            client_uid,
            AssistantActiveApplicationMessage(
                process_name="Code.exe",
                window_title="README.md - Visual Studio Code",
                detected_at="2026-07-22T12:30:00+00:00",
                confidence=1.0,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
            AssistantContextSyncMessage(
                version=4,
                matched_profile_id=None,
                effective_profile_id=None,
                manual_profile_id=None,
                profile_selection_mode="automatic",
                profile_selection_source="none",
                profile_match_confidence=None,
                speech_mode="action",
                proactive_commentary_enabled=False,
                minimum_commentary_interval_ms=15_000,
                max_queued_conversation_events=5,
                announce_application_changes=False,
                acknowledge_routine_automation_success=False,
                player_speaking=False,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
            AssistantActiveApplicationMessage(
                process_name="EliteDangerous64.exe",
                window_title="Elite Dangerous",
                detected_at="2026-07-22T12:31:00+00:00",
                confidence=1.0,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
            AssistantContextSyncMessage(
                version=5,
                matched_profile_id="elite-dangerous",
                effective_profile_id="elite-dangerous",
                manual_profile_id=None,
                profile_selection_mode="automatic",
                profile_selection_source="automatic",
                profile_match_confidence=1.0,
                speech_mode="action",
                proactive_commentary_enabled=False,
                minimum_commentary_interval_ms=15_000,
                max_queued_conversation_events=5,
                announce_application_changes=False,
                acknowledge_routine_automation_success=False,
                player_speaking=False,
            ),
        )

        assert len(start_calls) == 2

    asyncio.run(scenario())


def test_periodic_game_factoid_commentary_starts_for_matching_active_game():
    async def scenario() -> None:
        start_calls: list[tuple[str, str, dict]] = []

        async def send_to_client(_client_uid: str, _payload: dict) -> None:
            return None

        async def speak_to_client(_client_uid: str, _text: str) -> None:
            return None

        def can_start(_client_uid: str) -> bool:
            return True

        async def start_conversation(
            client_uid: str, text: str, metadata: dict
        ) -> bool:
            start_calls.append((client_uid, text, metadata))
            return True

        transport = AutomationTransport(send_to_client)
        coordinator = AutomationAssistantCoordinator(
            transport,
            send_to_client,
            speak_to_client,
            can_start,
            start_conversation,
        )

        client_uid = "client-4"
        status_message = await transport.handle_incoming_message(
            client_uid, _status_payload("elite-dangerous")
        )
        await coordinator.handle_transport_message(client_uid, status_message)
        await coordinator.handle_transport_message(
            client_uid,
            AssistantActiveApplicationMessage(
                process_name="EliteDangerous64.exe",
                window_title="Elite Dangerous",
                detected_at="2026-07-22T11:50:00+00:00",
                confidence=1.0,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
            AssistantContextSyncMessage(
                version=1,
                matched_profile_id="elite-dangerous",
                effective_profile_id="elite-dangerous",
                manual_profile_id="elite-dangerous",
                profile_selection_mode="manual",
                profile_selection_source="manual",
                profile_match_confidence=1.0,
                speech_mode="conversation",
                proactive_commentary_enabled=True,
                minimum_commentary_interval_ms=15_000,
                max_queued_conversation_events=5,
                announce_application_changes=False,
                acknowledge_routine_automation_success=False,
                player_speaking=False,
            ),
        )

        state = coordinator.get_client_state(client_uid)
        queued = coordinator._queue_periodic_game_factoid_commentary(
            client_uid, state
        )
        started = await coordinator._maybe_trigger_pending_proactive_commentary(
            client_uid,
            state,
        )

        assert queued is True
        assert started is True
        assert len(start_calls) == 1
        _, prompt_text, metadata = start_calls[0]
        assert "spoiler-safe comment" in prompt_text
        assert metadata["assistant_commentary_type"] == "game_factoid_commentary"
        assert metadata["assistant_profile_id"] == "elite-dangerous"
        assert "GAME FACTOID COMMENTARY" in metadata["assistant_context_text"]
        assert "get_game_factoid_context" in metadata["assistant_context_text"]
        assert "remember_game_factoid" in metadata["assistant_context_text"]

    asyncio.run(scenario())


def test_periodic_game_factoid_commentary_ignores_non_game_foreground_app():
    async def scenario() -> None:
        async def send_to_client(_client_uid: str, _payload: dict) -> None:
            return None

        async def speak_to_client(_client_uid: str, _text: str) -> None:
            return None

        def can_start(_client_uid: str) -> bool:
            return True

        async def start_conversation(
            _client_uid: str, _text: str, _metadata: dict
        ) -> bool:
            return True

        transport = AutomationTransport(send_to_client)
        coordinator = AutomationAssistantCoordinator(
            transport,
            send_to_client,
            speak_to_client,
            can_start,
            start_conversation,
        )

        client_uid = "client-5"
        status_message = await transport.handle_incoming_message(
            client_uid, _status_payload("elite-dangerous")
        )
        await coordinator.handle_transport_message(client_uid, status_message)
        await coordinator.handle_transport_message(
            client_uid,
            AssistantActiveApplicationMessage(
                process_name="Code.exe",
                window_title="README.md - Visual Studio Code",
                detected_at="2026-07-22T11:50:00+00:00",
                confidence=1.0,
            ),
        )
        await coordinator.handle_transport_message(
            client_uid,
            AssistantContextSyncMessage(
                version=1,
                matched_profile_id="elite-dangerous",
                effective_profile_id="elite-dangerous",
                manual_profile_id="elite-dangerous",
                profile_selection_mode="manual",
                profile_selection_source="manual",
                profile_match_confidence=1.0,
                speech_mode="conversation",
                proactive_commentary_enabled=True,
                minimum_commentary_interval_ms=15_000,
                max_queued_conversation_events=5,
                announce_application_changes=False,
                acknowledge_routine_automation_success=False,
                player_speaking=False,
            ),
        )

        state = coordinator.get_client_state(client_uid)
        queued = coordinator._queue_periodic_game_factoid_commentary(
            client_uid, state
        )

        assert queued is False
        assert state.pending_proactive_commentary is None

    asyncio.run(scenario())
