import pytest

from src.open_llm_vtuber.conversations.conversation_utils import (
    finalize_conversation_turn,
)
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager


@pytest.mark.asyncio
async def test_finalize_conversation_turn_ignores_closed_websocket_send():
    async def completed_task():
        return None

    async def failing_send(_message: str):
        raise AssertionError()

    tts_manager = TTSTaskManager()
    tts_manager.task_list.append(completed_task())

    await finalize_conversation_turn(
        tts_manager=tts_manager,
        websocket_send=failing_send,
        client_uid="client-1",
    )
