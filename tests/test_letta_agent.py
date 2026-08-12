import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_llm_vtuber.agent.agents.letta_agent import LettaAgent


class _FakeMessagesCreateStream:
    def __init__(self):
        self.calls = []

    def create_stream(self, **kwargs):
        self.calls.append(("create_stream", kwargs))
        return iter(["alpha"])


class _FakeMessagesStream:
    def __init__(self):
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(("stream", kwargs))
        return iter(["beta"])


class _FakeMessagesCreate:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return iter(["gamma"])


class _FakeAgents:
    def __init__(self, messages):
        self.messages = messages


class _FakeClient:
    def __init__(self, messages):
        self.agents = _FakeAgents(messages)


def _make_agent():
    return LettaAgent(
        live2d_model=None,
        id="agent-test",
    )


def test_letta_agent_prefers_create_stream_when_available():
    agent = _make_agent()
    messages_resource = _FakeMessagesCreateStream()
    agent.client = _FakeClient(messages_resource)

    stream = agent._create_message_stream([{"role": "user", "content": "hello"}])

    assert list(stream) == ["alpha"]
    assert messages_resource.calls == [
        (
            "create_stream",
            {
                "agent_id": "agent-test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream_tokens": True,
            },
        )
    ]


def test_letta_agent_falls_back_to_stream():
    agent = _make_agent()
    messages_resource = _FakeMessagesStream()
    agent.client = _FakeClient(messages_resource)

    stream = agent._create_message_stream([{"role": "user", "content": "hello"}])

    assert list(stream) == ["beta"]
    assert messages_resource.calls == [
        (
            "stream",
            {
                "agent_id": "agent-test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream_tokens": True,
            },
        )
    ]


def test_letta_agent_falls_back_to_create_streaming_true():
    agent = _make_agent()
    messages_resource = _FakeMessagesCreate()
    agent.client = _FakeClient(messages_resource)

    stream = agent._create_message_stream([{"role": "user", "content": "hello"}])

    assert list(stream) == ["gamma"]
    assert messages_resource.calls == [
        (
            "create",
            {
                "streaming": True,
                "agent_id": "agent-test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream_tokens": True,
            },
        )
    ]
