from typing import AsyncIterator, List, Dict, Any
from .agent_interface import AgentInterface
from ..output_types import SentenceOutput
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from ..file_prompting import render_file_attachments_for_prompt
from letta_client import Letta


class LettaAgent(AgentInterface):
    """
    Custom Letta class to interface with the Letta server.
    """

    def __init__(
        self,
        live2d_model,
        id,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        host: str = "localhost",
        port: int = 8283,
    ):
        super().__init__()
        self.url = f"http://{host}:{port}"
        self.client = Letta(base_url=self.url)
        self.id = id
        # Initialize decorator parameters
        self._tts_preprocessor_config = tts_preprocessor_config
        self._live2d_model = live2d_model
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method

        # Delay decorator application
        self.chat = tts_filter(self._tts_preprocessor_config)(
            display_processor()(
                actions_extractor(self._live2d_model)(
                    sentence_divider(
                        faster_first_response=self._faster_first_response,
                        segment_method=self._segment_method,
                        valid_tags=["think"],
                    )(self.chat)
                )
            )
        )

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        # The Letta Server automatically stores historical messages, so this part is not needed
        pass

    def handle_interrupt(self, heard_response: str) -> None:
        pass

    async def generator_to_async(self, gen):
        for item in gen:
            yield item

    def _create_message_stream(self, messages):
        messages_resource = self.client.agents.messages
        stream_kwargs = {
            "agent_id": self.id,
            "messages": messages,
            "stream_tokens": True,
        }

        if hasattr(messages_resource, "create_stream"):
            return messages_resource.create_stream(**stream_kwargs)
        if hasattr(messages_resource, "stream"):
            return messages_resource.stream(**stream_kwargs)
        if hasattr(messages_resource, "create"):
            return messages_resource.create(streaming=True, **stream_kwargs)

        raise AttributeError(
            "The configured Letta client does not expose a supported streaming "
            "messages API. Expected one of: create_stream, stream, or create."
        )

    async def chat(self, input_data: BatchInput) -> AsyncIterator[SentenceOutput]:
        messages = self._to_messages(input_data)
        stream = self.generator_to_async(self._create_message_stream(messages))

        complete_response = ""
        async for token in stream:
            if token.message_type == "reasoning_message":
                # This part is reasoning information and should not be displayed
                token = token.reasoning
                continue
            elif token.message_type == "assistant_message":
                # This part is the result that needs to be displayed, it is the final result
                # logger.info('Test message')
                # logger.info(token)
                token = token.content
            else:
                continue

            yield token
            complete_response += token

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """
        Format BatchInput into a prompt string for the LLM.

        Args:
            input_data: BatchInput - The input data containing texts

        Returns:
            str - Formatted message string
        """
        message_parts = []

        # Process text inputs in order
        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(f"[Clipboard content: {text_data.content}]")

        if input_data.images:
            message_parts.append("[User has also provided images]")

        file_prompt = render_file_attachments_for_prompt(input_data.files)
        if file_prompt:
            message_parts.append(file_prompt)

        return "\n".join(message_parts)

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """
        Prepare messages list without image support.
        """
        messages = []
        assistant_context_text = None
        if input_data.metadata:
            assistant_context_text = input_data.metadata.get("assistant_context_text")
        if isinstance(assistant_context_text, str) and assistant_context_text.strip():
            messages.append({"role": "system", "content": assistant_context_text})

        if input_data.images:
            content = []
            text_content = self._to_text_prompt(input_data)
            content.append({"type": "text", "text": text_content})
            user_message = {"role": "user", "content": content}
        else:
            user_message = {"role": "user", "content": self._to_text_prompt(input_data)}

        messages.append(user_message)

        return messages
