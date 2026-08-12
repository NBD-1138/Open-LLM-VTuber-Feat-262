from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from ..agent.output_types import DisplayText
from ..tts.edge_tts import TTSEngine as EdgeTTSEngine
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from ..websocket_handler import WebSocketHandler
from .config import TalkbackConf, TalkbackTTSConf


class LivePublisher:
    """Utility to publish live integration events to connected frontend clients."""

    def __init__(
        self,
        ws_handler: WebSocketHandler,
        platform: str = "twitch",
        talkback_config: Optional[TalkbackConf] = None,
    ):
        self._ws_handler = ws_handler
        self._platform = platform
        self._log = logging.getLogger("live.publisher")

        self._talkback_tts_enabled = False
        self._talkback_tts_engine_name = "edge_tts"
        self._talkback_tts_voice = "en-US-AvaMultilingualNeural"
        self._talkback_tts_engine: Optional[TTSInterface] = None
        self._tts_lock = asyncio.Lock()

        if talkback_config is not None:
            self.configure_talkback_tts(talkback_config.tts)

    async def _broadcast(self, message: Dict[str, Any]) -> None:
        self._log.debug("Broadcasting live message: %s", message)
        await self._ws_handler.broadcast_json(message)

    def configure_talkback_tts(self, tts_conf: TalkbackTTSConf) -> None:
        """Initialise talkback TTS configuration from live config."""
        self._talkback_tts_enabled = bool(tts_conf.enabled)
        self._talkback_tts_engine_name = tts_conf.engine or "edge_tts"
        voice = (tts_conf.voice or "").strip()
        if voice:
            self._talkback_tts_voice = voice
        self._reset_talkback_tts_engine()
        self._log.info(
            "Talkback TTS configured: enabled=%s engine=%s voice=%s",
            self._talkback_tts_enabled,
            self._talkback_tts_engine_name,
            self._talkback_tts_voice,
        )

    def update_talkback_tts(
        self,
        *,
        enabled: Optional[bool] = None,
        voice: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> None:
        """Update talkback TTS settings at runtime."""
        changed = False
        if enabled is not None and enabled != self._talkback_tts_enabled:
            self._talkback_tts_enabled = enabled
            changed = True
        if engine:
            engine = engine.strip()
            if engine and engine != self._talkback_tts_engine_name:
                self._talkback_tts_engine_name = engine
                changed = True
        if voice is not None:
            voice = voice.strip()
            if voice and voice != self._talkback_tts_voice:
                self._talkback_tts_voice = voice
                changed = True
        if changed:
            self._reset_talkback_tts_engine()
            self._log.info(
                "Talkback TTS updated: enabled=%s engine=%s voice=%s",
                self._talkback_tts_enabled,
                self._talkback_tts_engine_name,
                self._talkback_tts_voice,
            )

    def get_talkback_tts_status(self) -> Dict[str, Any]:
        """Expose current talkback TTS runtime settings for frontend status snapshots."""
        return {
            "enabled": self._talkback_tts_enabled,
            "engine": self._talkback_tts_engine_name,
            "voice": self._talkback_tts_voice,
        }

    def _reset_talkback_tts_engine(self) -> None:
        self._talkback_tts_engine = None

    async def _ensure_talkback_tts_engine(self) -> Optional[TTSInterface]:
        if not self._talkback_tts_enabled:
            return None
        if self._talkback_tts_engine is not None:
            return self._talkback_tts_engine
        if self._talkback_tts_engine_name != "edge_tts":
            self._log.warning(
                "Unsupported talkback TTS engine requested: %s",
                self._talkback_tts_engine_name,
            )
            return None
        self._talkback_tts_engine = EdgeTTSEngine(voice=self._talkback_tts_voice)
        return self._talkback_tts_engine

    async def send_chat(
        self,
        user: str,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "type": "external-chat",
            "source": self._platform,
            "user": user,
            "text": text,
        }
        if metadata:
            payload["metadata"] = metadata
        await self._broadcast(payload)
        await self._speak_talkback(user, text, metadata)

    async def forward_to_backend(
        self, text: str, metadata: Dict[str, Any]
    ) -> None:
        await self._broadcast(
            {
                "type": "talkback-forward",
                "text": text,
                "metadata": metadata,
            }
        )

    async def send_notification(
        self,
        text: str,
        *,
        category: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "type": "external-notification",
            "source": self._platform,
            "category": category,
            "text": text,
        }
        if metadata:
            payload["metadata"] = metadata

        await self._broadcast(payload)
        # Echo notification to backend as a system chat so the VTuber can speak it
        await self._ws_handler.handle_system_notification(
            text,
            category,
            self._platform,
            metadata=metadata,
        )

    async def publish_eventsub(self, data: Dict[str, Any]) -> None:
        await self._broadcast({"type": "eventsub", "payload": data})

    async def _speak_talkback(
        self,
        user: str,
        display_text: str,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if self._platform != "twitch":
            self._log.debug("Skipping talkback TTS: platform %s not twitch", self._platform)
            return
        if not self._talkback_tts_enabled:
            self._log.debug("Talkback TTS disabled; skipping speech for: %s", display_text)
            return
        spoken_text = ""
        metadata = metadata or {}
        user_name = user
        if isinstance(metadata, dict):
            spoken_text = str(metadata.get("spoken_text") or "").strip()
            user_name = str(metadata.get("user") or user or "viewer")
        if spoken_text:
            voiceover_text = f"{user_name} says {spoken_text}"
        else:
            voiceover_text = display_text
        voiceover_text = voiceover_text.strip()
        if not voiceover_text:
            self._log.debug("Talkback TTS skipping empty voiceover text for user %s", user_name)
            return

        engine_ref: Optional[TTSInterface] = None
        async with self._tts_lock:
            engine = await self._ensure_talkback_tts_engine()
            if engine is None:
                self._log.warning("Talkback TTS engine unavailable; skipping speech.")
                return
            try:
                audio_path = await engine.async_generate_audio(
                    text=voiceover_text,
                    file_name_no_ext=f"talkback_{uuid.uuid4().hex}",
                )
                engine_ref = engine
                self._log.debug("Talkback TTS generated audio at %s", audio_path)
            except Exception as exc:  # pragma: no cover - defensive
                self._log.error("Talkback TTS generation failed: %s", exc)
                return

        if not audio_path:
            self._log.warning("Talkback TTS returned empty audio path; skipping.")
            return

        try:
            display_user = metadata.get("user") if isinstance(metadata, dict) else None
            payload = prepare_audio_payload(
                audio_path=audio_path,
                display_text=DisplayText(
                    text=display_text,
                    name=str(display_user or user or "Twitch"),
                ),
                forwarded=True,
            )
            await self._broadcast(payload)
            self._log.debug(
                "Talkback TTS broadcasted audio for %s (display '%s')",
                user_name,
                display_text,
            )
        finally:
            try:
                if engine_ref:
                    engine_ref.remove_file(audio_path, verbose=False)
            except Exception:  # pragma: no cover - cleanup safety
                self._log.debug("Failed to remove talkback audio cache '%s'", audio_path)
