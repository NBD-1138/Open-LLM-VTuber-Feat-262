import atexit
import json
from typing import Any, AsyncIterator, Dict, List

import httpx
import requests
from loguru import logger
from openai import NOT_GIVEN, NotGiven

from .openai_compatible_llm import AsyncLLM


class OllamaLLM(AsyncLLM):
    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
        keep_alive: float = -1,
        unload_at_exit: bool = True,
    ):
        self.keep_alive = keep_alive
        self.unload_at_exit = unload_at_exit
        self.cleaned = False
        self.native_base_url = self._resolve_native_base_url(base_url)
        self._cached_capabilities: set[str] | None = None
        super().__init__(
            model=model,
            base_url=base_url,
            llm_api_key=llm_api_key,
            organization_id=organization_id,
            project_id=project_id,
            temperature=temperature,
        )
        try:
            # preload model
            logger.info("Preloading model for Ollama")
            # Send the POST request to preload model
            logger.debug(
                requests.post(
                    self.native_base_url + "/api/chat",
                    json={
                        "model": model,
                        "keep_alive": keep_alive,
                    },
                )
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to preload model: {e}")
            logger.critical(
                "Fail to connect to Ollama backend. Is Ollama server running? Try running `ollama list` to start the server and try again.\nThe AI will repeat 'Error connecting chat endpoint' until the server is running."
            )
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
        # If keep_alive is less than 0, register cleanup to unload the model
        if unload_at_exit:
            atexit.register(self.cleanup)

    def __del__(self):
        """Destructor to unload the model"""
        self.cleanup()

    def cleanup(self):
        """Clean up function to unload the model when exitting"""
        if not self.cleaned and self.unload_at_exit:
            logger.info(f"Ollama: Unloading model: {self.model}")
            # Unload the model
            # unloading is just the same as preload, but with keep alive set to 0
            logger.debug(
                requests.post(
                    self.native_base_url + "/api/chat",
                    json={
                        "model": self.model,
                        "keep_alive": 0,
                    },
                )
            )
            self.cleaned = True

    @staticmethod
    def _resolve_native_base_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            return normalized[:-3]
        return normalized

    @staticmethod
    def _message_contains_images(messages: List[Dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "image_url":
                    return True
        return False

    @staticmethod
    def _extract_base64_image_data(block: Dict[str, Any]) -> str | None:
        image_url = block.get("image_url")
        if not isinstance(image_url, dict):
            return None
        url = image_url.get("url")
        if not isinstance(url, str) or not url.startswith("data:image"):
            return None
        if "," not in url:
            return None
        return url.split(",", 1)[1]

    @classmethod
    def _convert_messages_to_native_format(
        cls, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        native_messages: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content")

            if isinstance(content, str):
                native_messages.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                native_messages.append({"role": role, "content": str(content or "")})
                continue

            text_parts: List[str] = []
            image_payloads: List[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                elif block_type == "image_url":
                    image_data = cls._extract_base64_image_data(block)
                    if image_data:
                        image_payloads.append(image_data)

            native_message: Dict[str, Any] = {
                "role": role,
                "content": "\n".join(text_parts).strip(),
            }
            if image_payloads:
                native_message["images"] = image_payloads
            native_messages.append(native_message)

        return native_messages

    @staticmethod
    def _infer_capabilities_from_model_name(model: str) -> set[str]:
        normalized = model.lower()
        capabilities = {"completion"}
        vision_markers = (
            "vision",
            "llava",
            "bakllava",
            "minicpm-v",
            "minicpmv",
            "moondream",
            "qwen-vl",
            "qwen2.5-vl",
            "qwen2.5vl",
            "qwen3.5-vl",
            "qwen3.5vl",
            "gemma3",
            "pixtral",
            "llama3.2-vision",
            "llama3.2:vision",
            "internvl",
        )
        if any(marker in normalized for marker in vision_markers):
            capabilities.add("vision")
        return capabilities

    async def _get_model_capabilities(self) -> set[str]:
        if self._cached_capabilities is not None:
            return self._cached_capabilities

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    self.native_base_url + "/api/show",
                    json={"model": self.model},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            logger.warning(
                f"Failed to query Ollama model capabilities for {self.model}: {exc}"
            )
            self._cached_capabilities = self._infer_capabilities_from_model_name(
                self.model
            )
            return self._cached_capabilities

        capabilities = payload.get("capabilities")
        if isinstance(capabilities, list):
            normalized = {
                str(capability).strip().lower()
                for capability in capabilities
                if str(capability).strip()
            }
            if normalized:
                self._cached_capabilities = normalized
                return normalized

        self._cached_capabilities = self._infer_capabilities_from_model_name(
            self.model
        )
        return self._cached_capabilities

    async def _supports_vision(self) -> bool:
        return "vision" in await self._get_model_capabilities()

    async def _native_image_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
    ) -> AsyncIterator[str]:
        messages_with_system = messages
        if system:
            messages_with_system = [{"role": "system", "content": system}, *messages]

        native_messages = self._convert_messages_to_native_format(messages_with_system)
        summary = [
            {
                "role": message.get("role"),
                "content_length": len(str(message.get("content") or "")),
                "image_count": len(message.get("images") or []),
            }
            for message in native_messages
        ]
        logger.debug(f"Ollama native chat message summary: {summary}")

        payload = {
            "model": self.model,
            "messages": native_messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=None, write=60.0, pool=60.0)
        ) as client:
            async with client.stream(
                "POST", self.native_base_url + "/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])

                    message = chunk.get("message")
                    if not isinstance(message, dict):
                        continue

                    content = message.get("content")
                    if isinstance(content, str) and content:
                        yield content

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] | NotGiven = NOT_GIVEN,
    ) -> AsyncIterator[str]:
        if not self._message_contains_images(messages):
            async for chunk in super().chat_completion(messages, system, tools=tools):
                yield chunk
            return

        if not await self._supports_vision():
            logger.warning(
                f"Ollama model {self.model} does not advertise vision support."
            )
            yield (
                f'The current Ollama model "{self.model}" cannot inspect attached '
                "images. Switch to a vision-capable Ollama model and try again."
            )
            return

        if tools is not NOT_GIVEN:
            logger.info(
                "Skipping OpenAI-compatible tool definitions for Ollama native image chat."
            )

        try:
            async for chunk in self._native_image_chat_completion(messages, system):
                yield chunk
        except httpx.ConnectError as exc:
            logger.error(
                "Error calling the chat endpoint: Connection error. "
                f"Failed to connect to the Ollama API. {exc}"
            )
            yield (
                "Error calling the chat endpoint: Connection error. Failed to connect "
                "to the LLM API. Check the configurations and the reachability of the "
                "LLM backend. See the logs for details."
            )
        except httpx.HTTPStatusError as exc:
            response_text = ""
            if exc.response is not None:
                try:
                    response_text = exc.response.text
                except Exception:
                    response_text = "<unavailable>"
            logger.error(
                "LLM API: Error occurred during Ollama native image chat: "
                f"{exc}. Response: {response_text}"
            )
            yield (
                "Error calling the chat endpoint: Error occurred while generating "
                "response. See the logs for details."
            )
        except Exception as exc:
            logger.error(
                f"LLM API: Error occurred during Ollama native image chat: {exc}"
            )
            yield (
                "Error calling the chat endpoint: Error occurred while generating "
                "response. See the logs for details."
            )
