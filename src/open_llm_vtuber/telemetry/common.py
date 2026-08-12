from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

from loguru import logger

from .models import (
    StructuredTelemetrySnapshot,
    TelemetryAdapterSnapshot,
    TelemetryCapabilityAdvice,
    TelemetryDirectoryStatus,
    TelemetryEvent,
    TelemetrySettings,
    redact_directory_label,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def clip_summary(value: str, limit: int = 160) -> str:
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def normalize_token_text(value: str) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def discover_configured_directory(
    settings: TelemetrySettings,
    default_directory: Path,
) -> tuple[Optional[Path], TelemetryDirectoryStatus, Optional[str]]:
    if settings.automatic_journal_discovery:
        if default_directory.exists():
            return (
                default_directory,
                "automatic",
                redact_directory_label(default_directory),
            )
        return (
            default_directory,
            "missing",
            redact_directory_label(default_directory),
        )

    if settings.manual_journal_directory:
        path = Path(settings.manual_journal_directory)
        if path.exists():
            return path, "manual", redact_directory_label(path)
        return path, "missing", redact_directory_label(path)
    return None, "unavailable", None


@dataclass(slots=True)
class TextFilePollResult:
    lines: list[str]
    current_file: Optional[str]


class AppendOnlyTextFileTailer:
    def __init__(self) -> None:
        self._current_path: Optional[Path] = None
        self._position = 0
        self._partial = b""

    def reset(self) -> None:
        self._current_path = None
        self._position = 0
        self._partial = b""

    async def poll(self, path: Path) -> TextFilePollResult:
        return await asyncio.to_thread(self._poll_sync, path)

    def _poll_sync(self, path: Path) -> TextFilePollResult:
        if self._current_path != path:
            self.reset()
            self._current_path = path

        try:
            size = path.stat().st_size
        except FileNotFoundError:
            self.reset()
            return TextFilePollResult(lines=[], current_file=None)

        if size < self._position:
            self._position = 0
            self._partial = b""

        try:
            with path.open("rb") as handle:
                handle.seek(self._position)
                chunk = handle.read()
                self._position = handle.tell()
        except FileNotFoundError:
            self.reset()
            return TextFilePollResult(lines=[], current_file=None)

        if not chunk:
            return TextFilePollResult(lines=[], current_file=path.name)

        data = self._partial + chunk
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith((b"\n", b"\r")):
            self._partial = lines.pop()
        else:
            self._partial = b""

        parsed_lines: list[str] = []
        for raw_line in lines:
            text = raw_line.decode("utf-8", errors="replace").strip()
            if text:
                parsed_lines.append(text)

        return TextFilePollResult(lines=parsed_lines, current_file=path.name)


class BaseTelemetryAdapter:
    adapter_id = "telemetry-adapter"
    game_id = "unknown"
    game_display_name = "Unknown Game"
    source_kind = "log"
    supported_process_names: tuple[str, ...] = ()

    def __init__(
        self,
        on_event: Callable[[TelemetryEvent], Awaitable[None]],
        on_state_change: Callable[[], Awaitable[None]],
    ) -> None:
        self._on_event = on_event
        self._on_state_change = on_state_change
        self._settings = TelemetrySettings()
        self._snapshot = TelemetryAdapterSnapshot(
            adapter_id=self.adapter_id,
            game_id=self.game_id,
            game_display_name=self.game_display_name,
            source_kind=self.source_kind,
            status="stopped",
        )
        self._task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._last_payload_signature: Optional[str] = None

    @classmethod
    def match_score(
        cls,
        *,
        profile_id: Optional[str],
        profile_display_name: Optional[str],
        profile_process_names: Iterable[str],
        detected_process_name: Optional[str],
        detected_window_title: Optional[str],
    ) -> int:
        del profile_id, profile_display_name, detected_window_title
        process_names = {
            normalize_token_text(item)
            for item in profile_process_names
            if isinstance(item, str) and item.strip()
        }
        supported = {
            normalize_token_text(item)
            for item in getattr(cls, "supported_process_names", ())
        }
        if process_names.intersection(supported):
            return 100
        if (
            detected_process_name
            and normalize_token_text(detected_process_name) in supported
        ):
            return 100
        return 0

    async def start(self, settings: TelemetrySettings) -> None:
        self._settings = settings
        self._closed = False
        if self._task and not self._task.done():
            await self._emit_state_if_changed()
            return
        self._snapshot.status = "starting"
        self._snapshot.error = None
        await self._emit_state_if_changed()
        self._task = asyncio.create_task(
            self._run_wrapper(), name=f"{self.game_id}-telemetry"
        )

    async def stop(self) -> None:
        self._closed = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        await self._reset_runtime_state()
        self._snapshot.status = "stopped"
        self._snapshot.error = None
        self._snapshot.source_directory_status = "unavailable"
        self._snapshot.source_directory_label = None
        self._snapshot.current_source = None
        self._snapshot.journal_directory_status = "unavailable"
        self._snapshot.journal_directory_label = None
        self._snapshot.current_journal_file = None
        self._snapshot.last_event_at = None
        self._snapshot.snapshot = self._current_structured_snapshot()
        await self._emit_state_if_changed()

    async def update_runtime_context(
        self,
        *,
        profile_id: Optional[str],
        profile_display_name: Optional[str],
        profile_process_names: Iterable[str],
        detected_process_name: Optional[str],
        detected_window_title: Optional[str],
    ) -> None:
        del (
            profile_id,
            profile_display_name,
            profile_process_names,
            detected_process_name,
            detected_window_title,
        )

    def current_snapshot(self) -> TelemetryAdapterSnapshot:
        self._snapshot.snapshot = self._current_structured_snapshot()
        return self._snapshot

    def build_capability_advice(
        self, capabilities: Iterable[dict[str, Any]]
    ) -> dict[str, TelemetryCapabilityAdvice]:
        del capabilities
        return {}

    async def _run_wrapper(self) -> None:
        try:
            await self._run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(f"{self.game_display_name} telemetry adapter failed: {exc}")
            self._snapshot.status = "error"
            self._snapshot.error = str(exc)
            self._snapshot.snapshot = self._current_structured_snapshot()
            await self._emit_state_if_changed()

    async def _run(self) -> None:
        raise NotImplementedError

    async def _reset_runtime_state(self) -> None:
        raise NotImplementedError

    def _current_structured_snapshot(self) -> StructuredTelemetrySnapshot:
        raise NotImplementedError

    async def _emit_event(self, event: TelemetryEvent) -> None:
        self._snapshot.last_event_at = isoformat(utcnow())
        await self._on_event(event)

    async def _emit_state_if_changed(self) -> None:
        self._snapshot.snapshot = self._current_structured_snapshot()
        payload = self._snapshot.to_payload()
        signature = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if signature == self._last_payload_signature:
            return
        self._last_payload_signature = signature
        await self._on_state_change()


async def maybe_update_adapter_runtime_context(
    adapter: Any,
    *,
    profile_id: Optional[str],
    profile_display_name: Optional[str],
    profile_process_names: Iterable[str],
    detected_process_name: Optional[str],
    detected_window_title: Optional[str],
) -> None:
    method = getattr(adapter, "update_runtime_context", None)
    if method is None:
        return
    result = method(
        profile_id=profile_id,
        profile_display_name=profile_display_name,
        profile_process_names=profile_process_names,
        detected_process_name=detected_process_name,
        detected_window_title=detected_window_title,
    )
    if inspect.isawaitable(result):
        await result
