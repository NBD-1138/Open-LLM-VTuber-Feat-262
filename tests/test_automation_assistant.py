import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from open_llm_vtuber.automation.assistant import AutomationAssistantCoordinator
from open_llm_vtuber.mcpp.tool_executor import (
    MAX_WEB_FETCH_RESULT_LENGTH,
    MAX_WEB_STATUS_RESULT_LENGTH,
    ToolExecutor,
)
from open_llm_vtuber.mcpp.tool_manager import ToolManager
from open_llm_vtuber.mcpp.types import (
    FormattedTool,
    ToolCallFunctionObject,
    ToolCallObject,
)
from open_llm_vtuber.automation.transport import (
    AutomationConfirmationResponseMessage,
    AutomationTransport,
    parse_automation_message,
)


def _run(coro):
    return asyncio.run(coro)


async def _collect(async_iter):
    items = []
    async for item in async_iter:
        items.append(item)
    return items


def _default_settings(**overrides):
    payload = {
        "type": "automation/assistant-settings",
        "llm_automation_enabled": True,
        "allow_autonomous_harmless_commands": False,
        "allow_autonomous_low_risk_commands": False,
        "confirmation_timeout_ms": 10_000,
        "max_pending_confirmations": 1,
        "announce_blocked_command_requests": True,
        "include_command_suggestions_in_speech": True,
        "result_acknowledgements_enabled": False,
        "confirmation_phrases": ["confirm", "do it", "yes, run it"],
        "cancellation_phrases": ["cancel", "never mind", "stop"],
        "voice_command_activation_mode": "wake_phrase",
        "voice_command_wake_phrases": ["assistant", "computer"],
        "voice_command_push_to_command_hotkey": "CommandOrControl+Alt+Shift+F11",
        "voice_command_ambiguity_timeout_ms": 15_000,
        "voice_command_acknowledgement_mode": "fixed_original",
    }
    payload.update(overrides)
    return payload


def _default_capability(**overrides):
    payload = {
        "profile_id": "assistant-test-profile",
        "command_id": "assistant_test_sequence",
        "label": "Assistant Test Sequence",
        "description": "Safe assistant test.",
        "aliases": ["assistant test", "test sequence"],
        "category": "testing",
        "enabled": True,
        "available": True,
        "risk": "harmless",
        "autonomy_policy": "confirmation_required",
        "allowed_trigger_sources": ["manual", "vtuber"],
        "cooldown_remaining_ms": 0,
    }
    payload.update(overrides)
    return payload


class AssistantHarness:
    def __init__(self):
        self.sent_messages = []
        self.spoken_messages = []
        self.client_uid = "client-1"

        async def send_to_client(client_uid, message):
            self.sent_messages.append((client_uid, message))

        async def speak_to_client(client_uid, text):
            self.spoken_messages.append((client_uid, text))

        def can_start_conversation(client_uid):
            del client_uid
            return True

        async def start_conversation(client_uid, text, metadata):
            del client_uid, text, metadata
            return False

        self.transport = AutomationTransport(send_to_client)
        self.transport.register_client(self.client_uid)
        self.coordinator = AutomationAssistantCoordinator(
            self.transport,
            send_to_client,
            speak_to_client,
            can_start_conversation,
            start_conversation,
        )
        self.coordinator.register_client(self.client_uid)

    def deliver(self, payload):
        message = _run(self.transport.handle_incoming_message(self.client_uid, payload))
        _run(self.coordinator.handle_transport_message(self.client_uid, message))
        return message

    def seed(
        self,
        *,
        profile_enabled=True,
        command_enabled=True,
        capability_enabled=True,
        capability_available=True,
        autonomy_policy="confirmation_required",
        risk="harmless",
        llm_enabled=True,
        allow_autonomous_harmless=False,
        allow_autonomous_low_risk=False,
        result_acknowledgements_enabled=False,
        allowed_trigger_sources=None,
    ):
        self.deliver(
            {
                "type": "automation/status",
                "status": "ready",
                "emergency_stopped": False,
                "active_profile_id": "assistant-test-profile",
                "profiles": [
                    {
                        "profile_id": "assistant-test-profile",
                        "display_name": "Assistant Test Profile",
                        "enabled": profile_enabled,
                        "commands": [
                            {
                                "command_id": "assistant_test_sequence",
                                "label": "Assistant Test Sequence",
                                "enabled": command_enabled,
                            }
                        ],
                    }
                ],
                "running_requests": [],
                "last_error": None,
            }
        )
        self.deliver(
            {
                "type": "automation/capabilities",
                "active_profile_id": "assistant-test-profile",
                "revision": 1,
                "capabilities": [
                    _default_capability(
                        enabled=capability_enabled,
                        available=capability_available,
                        autonomy_policy=autonomy_policy,
                        risk=risk,
                        allowed_trigger_sources=(
                            allowed_trigger_sources
                            if allowed_trigger_sources is not None
                            else ["manual", "vtuber"]
                        ),
                    )
                ],
            }
        )
        self.deliver(
            _default_settings(
                llm_automation_enabled=llm_enabled,
                allow_autonomous_harmless_commands=allow_autonomous_harmless,
                allow_autonomous_low_risk_commands=allow_autonomous_low_risk,
                result_acknowledgements_enabled=result_acknowledgements_enabled,
            )
        )

    def run_tool(self, tool_name, tool_input=None):
        tool_input = tool_input or {}
        tool = self.coordinator.build_local_tools(self.client_uid)[tool_name]
        result = _run(tool.handler(tool_input))
        return result["metadata"]

    def last_message_of_type(self, message_type):
        for _client_uid, message in reversed(self.sent_messages):
            if message.get("type") == message_type:
                return message
        return None


class FakeStreamingLLM:
    def __init__(self, event_batches):
        self._event_batches = list(event_batches)
        self.call_count = 0

    async def chat_completion(self, messages, system=None, tools=None):
        del messages, system, tools
        if self.call_count >= len(self._event_batches):
            batch = []
        else:
            batch = self._event_batches[self.call_count]
        self.call_count += 1
        for item in batch:
            yield item


def test_capability_message_rejects_duplicate_command_ids():
    with pytest.raises(ValidationError):
        parse_automation_message(
            {
                "type": "automation/capabilities",
                "active_profile_id": "assistant-test-profile",
                "revision": 1,
                "capabilities": [
                    _default_capability(command_id="alpha"),
                    _default_capability(command_id="alpha"),
                ],
            }
        )


def test_capability_revision_replaces_catalogue_state():
    harness = AssistantHarness()
    harness.deliver(
        {
            "type": "automation/capabilities",
            "active_profile_id": "assistant-test-profile",
            "revision": 1,
            "capabilities": [_default_capability(command_id="alpha", label="Alpha")],
        }
    )
    harness.deliver(
        {
            "type": "automation/capabilities",
            "active_profile_id": "assistant-test-profile",
            "revision": 2,
            "capabilities": [_default_capability(command_id="beta", label="Beta")],
        }
    )

    state = harness.coordinator.get_client_state(harness.client_uid)
    assert state.capability_revision == 2
    assert list(state.capabilities.keys()) == ["beta"]


def test_hallucinated_command_ids_are_blocked():
    harness = AssistantHarness()
    harness.seed()

    result = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "imaginary_command",
            "concise_reason": "Testing a missing command.",
        },
    )

    assert result["status"] == "blocked"
    assert "capability catalogue" in result["blocked_reason"]


def test_suggest_tool_uses_active_profile_and_resolves_aliases():
    harness = AssistantHarness()
    harness.deliver(
        {
            "type": "automation/status",
            "status": "ready",
            "emergency_stopped": False,
            "active_profile_id": "warframe",
            "profiles": [
                {
                    "profile_id": "warframe",
                    "display_name": "Warframe",
                    "enabled": True,
                    "commands": [
                        {
                            "command_id": "bullet_jump",
                            "label": "Bullet Jump",
                            "enabled": True,
                        }
                    ],
                }
            ],
            "running_requests": [],
            "last_error": None,
        }
    )
    harness.deliver(
        {
            "type": "automation/capabilities",
            "active_profile_id": "warframe",
            "revision": 1,
            "capabilities": [
                _default_capability(
                    profile_id="warframe",
                    command_id="bullet_jump",
                    label="Bullet Jump",
                    aliases=["bullet jump", "jump"],
                )
            ],
        }
    )
    harness.deliver(_default_settings())

    result = harness.run_tool(
        "suggest_automation_command",
        {
            "command_id": "bullet jump",
            "concise_reason": "Player requested a bullet jump.",
        },
    )

    assert result["status"] == "suggested"
    assert result["profile_id"] == "warframe"
    assert result["command_id"] == "bullet_jump"
    state = harness.coordinator.get_client_state(harness.client_uid)
    assert state.recent_decisions[0].command_id == "bullet_jump"


def test_request_tool_uses_active_profile_and_resolves_camel_case_names():
    harness = AssistantHarness()
    harness.deliver(
        {
            "type": "automation/status",
            "status": "ready",
            "emergency_stopped": False,
            "active_profile_id": "warframe",
            "profiles": [
                {
                    "profile_id": "warframe",
                    "display_name": "Warframe",
                    "enabled": True,
                    "commands": [
                        {
                            "command_id": "bullet_jump",
                            "label": "Bullet Jump",
                            "enabled": True,
                        }
                    ],
                }
            ],
            "running_requests": [],
            "last_error": None,
        }
    )
    harness.deliver(
        {
            "type": "automation/capabilities",
            "active_profile_id": "warframe",
            "revision": 1,
            "capabilities": [
                _default_capability(
                    profile_id="warframe",
                    command_id="bullet_jump",
                    label="Bullet Jump",
                    aliases=["bullet jump", "jump"],
                )
            ],
        }
    )
    harness.deliver(_default_settings())

    result = harness.run_tool(
        "request_automation_command",
        {
            "command_id": "BulletJump",
            "concise_reason": "Player requested a bullet jump.",
        },
    )

    assert result["status"] == "pending_confirmation"
    assert result["profile_id"] == "warframe"
    assert result["command_id"] == "bullet_jump"
    state = harness.coordinator.get_client_state(harness.client_uid)
    pending = next(iter(state.pending_confirmations.values()))
    assert pending.profile_id == "warframe"
    assert pending.command_id == "bullet_jump"


def test_llm_disabled_blocks_requests():
    harness = AssistantHarness()
    harness.seed(llm_enabled=False)

    result = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "Please run the assistant test.",
        },
    )

    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "LLM automation is disabled."


def test_player_voice_trigger_source_allows_assistant_execution():
    harness = AssistantHarness()
    harness.seed(
        allowed_trigger_sources=["manual", "player_voice"],
    )

    result = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "The player asked for the assistant test by voice.",
        },
    )

    assert result["status"] == "pending_confirmation"
    assert result["command_id"] == "assistant_test_sequence"
    assert harness.last_message_of_type("automation/execute") is None


def test_exact_typed_alias_triggers_pending_confirmation_without_llm():
    harness = AssistantHarness()
    harness.seed(
        allowed_trigger_sources=["manual", "player_voice"],
    )
    harness.deliver(
        _default_settings(
            voice_command_activation_mode="always_listening",
        )
    )

    suppress, metadata = _run(
        harness.coordinator.prepare_conversation_metadata(
            harness.client_uid,
            "test sequence",
            {"source": "player"},
            source="player",
        )
    )

    del metadata
    state = harness.coordinator.get_client_state(harness.client_uid)
    assert suppress is True
    assert len(state.pending_confirmations) == 1
    assert any(
        "Say confirm to continue." in text
        for _client_uid, text in harness.spoken_messages
    )


def test_exact_voice_label_triggers_pending_confirmation_without_llm():
    harness = AssistantHarness()
    harness.seed(
        allowed_trigger_sources=["manual", "player_voice"],
    )

    suppress, metadata = _run(
        harness.coordinator.prepare_conversation_metadata(
            harness.client_uid,
            "Computer, Assistant Test Sequence.",
            {"source": "player_voice"},
            source="player_voice",
        )
    )

    del metadata
    state = harness.coordinator.get_client_state(harness.client_uid)
    assert suppress is True
    assert len(state.pending_confirmations) == 1
    assert any(
        "Say confirm to continue." in text
        for _client_uid, text in harness.spoken_messages
    )


def test_incidental_wake_word_in_the_middle_does_not_activate_voice_commands():
    harness = AssistantHarness()
    harness.seed(
        allowed_trigger_sources=["manual", "player_voice"],
    )

    suppress, metadata = _run(
        harness.coordinator.prepare_conversation_metadata(
            harness.client_uid,
            "Could you ask the computer about test sequence?",
            {"source": "player_voice"},
            source="player_voice",
        )
    )

    assert suppress is False
    assert metadata.get("assistant_voice_command_requested") is None


def test_explicit_wake_phrase_no_match_falls_back_to_conversation_with_safe_context():
    harness = AssistantHarness()
    harness.seed(
        allowed_trigger_sources=["manual", "player_voice"],
    )

    suppress, metadata = _run(
        harness.coordinator.prepare_conversation_metadata(
            harness.client_uid,
            "Computer, do the flashy thing",
            {"source": "player_voice"},
            source="player_voice",
        )
    )

    assert suppress is False
    assert metadata.get("assistant_voice_command_requested") is True
    assistant_context_text = metadata.get("assistant_context_text") or ""
    assert "no deterministic alias matched" in assistant_context_text
    assert "macro steps" in assistant_context_text


def test_confirmation_request_requires_explicit_confirmation_before_execution():
    harness = AssistantHarness()
    harness.seed()

    pending = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "The player asked for the assistant test.",
        },
    )

    assert pending["status"] == "pending_confirmation"
    assert harness.last_message_of_type("automation/execute") is None

    handled = _run(
        harness.coordinator.maybe_handle_confirmation_phrase(
            harness.client_uid, "confirm"
        )
    )
    assert handled is True

    execute_message = harness.last_message_of_type("automation/execute")
    assert execute_message is not None
    assert execute_message["command_id"] == "assistant_test_sequence"


def test_rejecting_pending_confirmation_clears_request():
    harness = AssistantHarness()
    harness.seed()

    pending = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "The player asked for the assistant test.",
        },
    )

    handled = _run(
        harness.coordinator.handle_confirmation_response(
            harness.client_uid,
            AutomationConfirmationResponseMessage(
                request_id=pending["request_id"],
                action="reject",
            ),
        )
    )
    assert handled is True
    state = harness.coordinator.get_client_state(harness.client_uid)
    assert pending["request_id"] not in state.pending_confirmations
    assert harness.last_message_of_type("automation/execute") is None


def test_emergency_stop_cancels_pending_confirmations():
    harness = AssistantHarness()
    harness.seed()

    pending = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "The player asked for the assistant test.",
        },
    )
    assert pending["status"] == "pending_confirmation"

    harness.deliver(
        {
            "type": "automation/emergency_stop",
            "emergency_stopped": True,
            "reason": "manual stop",
        }
    )

    state = harness.coordinator.get_client_state(harness.client_uid)
    assert not state.pending_confirmations
    assert state.last_blocked_reason == "Emergency stop is active."


def test_duplicate_requests_are_blocked_while_pending():
    harness = AssistantHarness()
    harness.seed()

    first = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "The player asked for the assistant test.",
        },
    )
    second = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "Trying to request it twice.",
        },
    )

    assert first["status"] == "pending_confirmation"
    assert second["status"] == "blocked"
    assert "already has a pending" in second["blocked_reason"]


def test_completed_requests_are_blocked_within_same_player_turn():
    harness = AssistantHarness()
    harness.seed(
        autonomy_policy="autonomous",
        allow_autonomous_harmless=True,
    )

    state = harness.coordinator.get_client_state(harness.client_uid)
    state.runtime.record_direct_request("Please do the assistant test.")

    first = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "The player asked for the assistant test.",
        },
    )
    assert first["status"] == "started"

    harness.deliver(
        {
            "type": "automation/result",
            "request_id": first["request_id"],
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "status": "completed",
            "duration_ms": 250,
        }
    )

    second = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "Trying to request it again in the same turn.",
        },
    )

    assert second["status"] == "blocked"
    assert "already handled for the current player request" in second["blocked_reason"]


def test_new_player_request_allows_repeating_a_completed_command():
    harness = AssistantHarness()
    harness.seed(
        autonomy_policy="autonomous",
        allow_autonomous_harmless=True,
    )

    state = harness.coordinator.get_client_state(harness.client_uid)
    state.runtime.record_direct_request("Please do the assistant test.")

    first = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "The player asked for the assistant test.",
        },
    )
    assert first["status"] == "started"

    harness.deliver(
        {
            "type": "automation/result",
            "request_id": first["request_id"],
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "status": "completed",
            "duration_ms": 250,
        }
    )

    state.runtime.record_direct_request("Do the assistant test again.")
    second = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "The player asked to repeat the assistant test.",
        },
    )

    assert second["status"] == "started"


def test_low_risk_autonomy_requires_override_but_can_run_when_enabled():
    harness = AssistantHarness()
    harness.seed(autonomy_policy="autonomous", risk="low")

    blocked = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "Try low-risk automation without override.",
        },
    )
    assert blocked["status"] == "pending_confirmation"

    harness = AssistantHarness()
    harness.seed(
        autonomy_policy="autonomous",
        risk="low",
        allow_autonomous_low_risk=True,
    )
    started = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "Try low-risk automation with override.",
        },
    )
    assert started["status"] == "started"
    assert harness.last_message_of_type("automation/execute") is not None


def test_failed_results_are_sanitized_and_optionally_acknowledged():
    harness = AssistantHarness()
    harness.seed(
        autonomy_policy="autonomous",
        allow_autonomous_harmless=True,
        result_acknowledgements_enabled=True,
    )

    started = harness.run_tool(
        "request_automation_command",
        {
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "concise_reason": "Run and then simulate a failure.",
        },
    )
    assert started["status"] == "started"

    harness.deliver(
        {
            "type": "automation/result",
            "request_id": started["request_id"],
            "profile_id": "assistant-test-profile",
            "command_id": "assistant_test_sequence",
            "status": "failed",
            "duration_ms": 250,
            "error": r"C:\secret\path\macro.json exploded badly",
        }
    )

    state = harness.coordinator.get_client_state(harness.client_uid)
    assert state.last_blocked_reason is not None
    assert "[path]" in state.last_blocked_reason
    assert any(
        "I couldn't run Assistant Test Sequence." in text
        for _client_uid, text in harness.spoken_messages
    )


def test_remain_silent_records_a_decision():
    harness = AssistantHarness()
    harness.seed()
    state = harness.coordinator.get_client_state(harness.client_uid)
    state.allow_remain_silent_tool = True

    result = harness.run_tool("remain_silent")

    assert result["status"] == "silent"
    assert result["stop_after_tool"] is True
    assert state.recent_decisions[0].decision_type == "silent"


def test_remain_silent_is_blocked_for_direct_turns():
    harness = AssistantHarness()
    harness.seed()

    _run(
        harness.coordinator.prepare_conversation_metadata(
            harness.client_uid,
            "Can you help me with this?",
            {"source": "player"},
            source="player",
        )
    )
    result = harness.run_tool("remain_silent")
    state = harness.coordinator.get_client_state(harness.client_uid)

    assert result["status"] == "blocked"
    assert "only allowed during proactive commentary turns" in result["blocked_reason"]
    assert state.last_blocked_reason == result["blocked_reason"]


def test_remain_silent_stops_openai_tool_loop():
    async def remain_silent_handler(tool_input):
        del tool_input
        return {
            "metadata": {"status": "silent", "stop_after_tool": True},
            "content_items": [{"type": "text", "text": '{"status": "silent"}'}],
        }

    tool_manager = ToolManager(
        initial_tools_dict={
            "remain_silent": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                related_server="assistant_automation",
                description="Stop talking.",
                handler=remain_silent_handler,
            )
        }
    )
    tool_executor = ToolExecutor(None, tool_manager)
    llm = FakeStreamingLLM(
        [
            [
                [
                    ToolCallObject(
                        id="call-1",
                        index=0,
                        function=ToolCallFunctionObject(
                            name="remain_silent",
                            arguments="{}",
                        ),
                    )
                ]
            ]
        ]
    )
    agent = BasicMemoryAgent(
        llm=llm,
        system="You are a helpful assistant.",
        live2d_model=None,
        use_mcpp=True,
        tool_manager=tool_manager,
        tool_executor=tool_executor,
    )

    outputs = _run(
        _collect(
            agent._openai_tool_interaction_loop(
                [{"role": "user", "content": "Remain silent."}],
                [],
            )
        )
    )

    assert llm.call_count == 1
    assert [item["type"] for item in outputs] == [
        "tool_call_status",
        "tool_call_status",
    ]
    assert outputs[0]["status"] == "running"
    assert outputs[1]["status"] == "completed"


def test_terminal_tool_stops_remaining_tool_execution():
    calls = []

    async def remain_silent_handler(tool_input):
        del tool_input
        calls.append("remain_silent")
        return {
            "metadata": {"status": "silent", "stop_after_tool": True},
            "content_items": [{"type": "text", "text": '{"status": "silent"}'}],
        }

    async def second_handler(tool_input):
        del tool_input
        calls.append("second")
        return {
            "metadata": {"status": "ok"},
            "content_items": [{"type": "text", "text": '{"status": "ok"}'}],
        }

    tool_manager = ToolManager(
        initial_tools_dict={
            "remain_silent": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                related_server="assistant_automation",
                description="Stop talking.",
                handler=remain_silent_handler,
            ),
            "second_tool": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                related_server="assistant_automation",
                description="A second tool that should not run.",
                handler=second_handler,
            ),
        }
    )
    tool_executor = ToolExecutor(None, tool_manager)

    updates = _run(
        _collect(
            tool_executor.execute_tools(
                tool_calls=[
                    {"id": "call-1", "name": "remain_silent", "args": {}},
                    {"id": "call-2", "name": "second_tool", "args": {}},
                ],
                caller_mode="OpenAI",
            )
        )
    )

    assert calls == ["remain_silent"]
    assert updates[-1]["type"] == "final_tool_results"
    assert updates[-1]["stop_after_tools"] is True
    assert updates[-1]["results"] == []


def test_fetch_content_results_are_truncated_for_llm_and_status():
    long_text = "Scorpion synthesis target location details. " * 300

    async def fetch_content_handler(tool_input):
        del tool_input
        return {
            "metadata": {"status": "ok"},
            "content_items": [{"type": "text", "text": long_text}],
        }

    tool_manager = ToolManager(
        initial_tools_dict={
            "fetch_content": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                related_server="ddg-search",
                description="Fetch a web page.",
                handler=fetch_content_handler,
            )
        }
    )
    tool_executor = ToolExecutor(None, tool_manager)

    updates = _run(
        _collect(
            tool_executor.execute_tools(
                tool_calls=[
                    {"id": "call-1", "name": "fetch_content", "args": {}},
                ],
                caller_mode="OpenAI",
            )
        )
    )

    status_update = updates[1]
    final_update = updates[-1]
    final_content = final_update["results"][0]["content"]

    assert status_update["type"] == "tool_call_status"
    assert len(status_update["content"]) <= MAX_WEB_STATUS_RESULT_LENGTH + 3
    assert final_update["type"] == "final_tool_results"
    assert len(final_content) <= MAX_WEB_FETCH_RESULT_LENGTH + 120
    assert "truncated" in final_content
    assert final_content != long_text


def test_twitch_chat_messages_bypass_commentary_policy_gate():
    harness = AssistantHarness()

    suppress, metadata = _run(
        harness.coordinator.prepare_conversation_metadata(
            harness.client_uid,
            "(Twitch) viewer123: hello there",
            {
                "source": "twitch",
                "message_id": "abc123",
                "message_kind": "chat",
                "user": "viewer123",
            },
            source="twitch",
        )
    )

    state = harness.coordinator.get_client_state(harness.client_uid)

    assert suppress is False
    assert "assistant_context_text" in metadata
    assert state.runtime.state.recent_twitch_events[0]["type"] == "twitch.chat"


def test_twitch_notifications_still_respect_commentary_policy_gate():
    harness = AssistantHarness()

    suppress, metadata = _run(
        harness.coordinator.prepare_conversation_metadata(
            harness.client_uid,
            "(Twitch) viewer123 joined the stream.",
            {
                "source": "twitch",
                "category": "system",
                "message_kind": "notification",
                "notification": True,
                "assistant_recorded": True,
                "user": "viewer123",
            },
            source="twitch",
        )
    )

    state = harness.coordinator.get_client_state(harness.client_uid)

    assert suppress is True
    assert "assistant_context_text" not in metadata
    assert state.runtime.build_payload()["activity"][0]["policy_reason"] == (
        "Twitch events are limited to conversation mode."
    )


def test_game_factoid_tools_store_and_reuse_safe_trivia():
    harness = AssistantHarness()
    harness.deliver(
        {
            "type": "automation/status",
            "status": "ready",
            "emergency_stopped": False,
            "active_profile_id": "warframe",
            "profiles": [
                {
                    "profile_id": "warframe",
                    "display_name": "Warframe",
                    "enabled": True,
                    "process_names": ["Warframe.x64.exe"],
                    "commands": [],
                }
            ],
            "running_requests": [],
            "last_error": None,
        }
    )

    stored = harness.run_tool(
        "remember_game_factoid",
        {
            "profile_id": "warframe",
            "fact_text": "Warframe's bullet jump became one of the game's signature movement features.",
            "source_name": "Developer interview",
            "category": "feature",
            "tags": ["movement", "trivia"],
        },
    )
    duplicate = harness.run_tool(
        "remember_game_factoid",
        {
            "profile_id": "warframe",
            "fact_text": "Warframe's bullet jump became one of the game's signature movement features.",
            "source_name": "Developer interview",
        },
    )
    context = harness.run_tool(
        "get_game_factoid_context",
        {"profile_id": "warframe"},
    )
    listed = harness.run_tool(
        "list_game_factoids",
        {"profile_id": "warframe"},
    )

    assert stored["status"] == "stored"
    assert duplicate["status"] == "existing"
    assert context["status"] == "ok"
    assert context["game_name"] == "Warframe"
    assert context["suggested_factoid"]["fact_text"].startswith(
        "Warframe's bullet jump"
    )
    assert listed["cached_factoid_count"] == 1


def test_game_factoid_tools_reject_spoilery_entries():
    harness = AssistantHarness()
    harness.deliver(
        {
            "type": "automation/status",
            "status": "ready",
            "emergency_stopped": False,
            "active_profile_id": "warframe",
            "profiles": [
                {
                    "profile_id": "warframe",
                    "display_name": "Warframe",
                    "enabled": True,
                    "process_names": ["Warframe.x64.exe"],
                    "commands": [],
                }
            ],
            "running_requests": [],
            "last_error": None,
        }
    )

    rejected = harness.run_tool(
        "remember_game_factoid",
        {
            "profile_id": "warframe",
            "fact_text": "The secret ending reveals the final boss after the late game twist.",
        },
    )

    assert rejected["status"] == "rejected"
