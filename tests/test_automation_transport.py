import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_llm_vtuber.automation.transport import (
    AutomationTransport,
    parse_automation_message,
)


def _run(coro):
    return asyncio.run(coro)


def test_automation_message_validation_rejects_unknown_trigger_source():
    with pytest.raises(ValidationError):
        parse_automation_message(
            {
                "type": "automation/execute",
                "request_id": "req-1",
                "profile_id": "profile-1",
                "command_id": "command-1",
                "source": "llm",
                "variables": {},
            }
        )


def test_automation_message_validation_rejects_malformed_request():
    with pytest.raises(ValidationError):
        parse_automation_message(
            {
                "type": "automation/result",
                "request_id": "",
                "profile_id": "profile-1",
                "command_id": "command-1",
                "status": "completed",
            }
        )


def test_request_execution_rejects_unknown_profile():
    sent_messages = []

    async def send_to_client(_client_uid, message):
        sent_messages.append(message)

    transport = AutomationTransport(send_to_client)
    transport.register_client("client-1")

    with pytest.raises(ValueError, match="Unknown automation profile"):
        _run(
            transport.request_execution(
                "client-1", "missing-profile", "command-1", source="manual"
            )
        )

    assert sent_messages == []


def test_request_execution_rejects_unknown_command():
    async def send_to_client(_client_uid, _message):
        return None

    transport = AutomationTransport(send_to_client)
    transport.register_client("client-1")
    _run(
        transport.handle_incoming_message(
            "client-1",
            {
                "type": "automation/status",
                "status": "ready",
                "profiles": [
                    {
                        "profile_id": "profile-1",
                        "display_name": "Profile 1",
                        "commands": [],
                    }
                ],
                "running_requests": [],
            },
        )
    )

    with pytest.raises(ValueError, match="Unknown automation command"):
        _run(
            transport.request_execution(
                "client-1", "profile-1", "missing-command", source="manual"
            )
        )


def test_result_handling_clears_running_request():
    sent_messages = []

    async def send_to_client(_client_uid, message):
        sent_messages.append(message)

    transport = AutomationTransport(send_to_client)
    transport.register_client("client-1")
    _run(
        transport.handle_incoming_message(
            "client-1",
            {
                "type": "automation/status",
                "status": "ready",
                "profiles": [
                    {
                        "profile_id": "profile-1",
                        "display_name": "Profile 1",
                        "commands": [
                            {
                                "command_id": "command-1",
                                "label": "Command 1",
                                "enabled": True,
                            }
                        ],
                    }
                ],
                "running_requests": [],
            },
        )
    )

    message = _run(
        transport.request_execution(
            "client-1",
            "profile-1",
            "command-1",
            source="manual",
            request_id="req-1",
        )
    )
    assert message.request_id == "req-1"
    assert sent_messages[0]["type"] == "automation/execute"

    _run(
        transport.handle_incoming_message(
            "client-1",
            {
                "type": "automation/result",
                "request_id": "req-1",
                "profile_id": "profile-1",
                "command_id": "command-1",
                "status": "completed",
                "duration_ms": 1250,
                "error": None,
            },
        )
    )

    state = transport.get_client_state("client-1")
    assert "req-1" not in state.running_request_ids
    assert state.results["req-1"].status == "completed"
    assert state.status == "ready"


def test_emergency_stop_status_updates_transport_state():
    async def send_to_client(_client_uid, _message):
        return None

    transport = AutomationTransport(send_to_client)
    transport.register_client("client-1")

    _run(
        transport.handle_incoming_message(
            "client-1",
            {
                "type": "automation/emergency_stop",
                "emergency_stopped": True,
                "reason": "manual stop",
            },
        )
    )

    state = transport.get_client_state("client-1")
    assert state.emergency_stopped is True
    assert state.status == "emergency_stopped"


def test_duplicate_request_ids_are_rejected():
    async def send_to_client(_client_uid, _message):
        return None

    transport = AutomationTransport(send_to_client)
    transport.register_client("client-1")
    _run(
        transport.handle_incoming_message(
            "client-1",
            {
                "type": "automation/status",
                "status": "ready",
                "profiles": [
                    {
                        "profile_id": "profile-1",
                        "display_name": "Profile 1",
                        "commands": [
                            {
                                "command_id": "command-1",
                                "label": "Command 1",
                                "enabled": True,
                            }
                        ],
                    }
                ],
                "running_requests": [],
            },
        )
    )

    _run(
        transport.request_execution(
            "client-1",
            "profile-1",
            "command-1",
            source="manual",
            request_id="req-1",
        )
    )

    with pytest.raises(ValueError, match="Duplicate automation request ID"):
        _run(
            transport.request_execution(
                "client-1",
                "profile-1",
                "command-1",
                source="manual",
                request_id="req-1",
            )
        )
