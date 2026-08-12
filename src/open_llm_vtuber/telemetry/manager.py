from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable, Optional

from .common import maybe_update_adapter_runtime_context
from .elite_dangerous import EliteDangerousTelemetryAdapter
from .factorio import FactorioTelemetryAdapter
from .minecraft import MinecraftTelemetryAdapter
from .models import (
    TelemetryAdapterSnapshot,
    TelemetryCapabilityAdvice,
    TelemetryEvent,
    TelemetrySettings,
)
from .msfs import MicrosoftFlightSimulatorTelemetryAdapter

DEFAULT_ADAPTER_TYPES = (
    EliteDangerousTelemetryAdapter,
    FactorioTelemetryAdapter,
    MinecraftTelemetryAdapter,
    MicrosoftFlightSimulatorTelemetryAdapter,
)


class ClientTelemetryRuntime:
    def __init__(
        self,
        on_event: Callable[[TelemetryEvent], Awaitable[None]],
        on_state_change: Callable[[], Awaitable[None]],
    ) -> None:
        self._on_event = on_event
        self._on_state_change = on_state_change
        self._settings = TelemetrySettings()
        self._profile_id: Optional[str] = None
        self._profile_display_name: Optional[str] = None
        self._profile_process_names: list[str] = []
        self._detected_process_name: Optional[str] = None
        self._detected_window_title: Optional[str] = None
        self._active_adapter: Optional[Any] = None
        self._active_adapter_type: Optional[type[Any]] = None

    async def update_settings(self, raw_settings: Any) -> None:
        self._settings = TelemetrySettings.from_raw(raw_settings)
        await self._reconcile_adapter()

    async def update_context(
        self,
        *,
        profile_id: Optional[str],
        profile_display_name: Optional[str] = None,
        profile_process_names: Iterable[str],
        detected_process_name: Optional[str],
        detected_window_title: Optional[str] = None,
    ) -> None:
        self._profile_id = profile_id
        self._profile_display_name = (
            profile_display_name.strip()
            if isinstance(profile_display_name, str)
            and profile_display_name.strip()
            else None
        )
        self._profile_process_names = [
            str(item).strip()
            for item in profile_process_names
            if isinstance(item, str) and item.strip()
        ]
        self._detected_process_name = (
            detected_process_name.strip()
            if isinstance(detected_process_name, str) and detected_process_name.strip()
            else None
        )
        self._detected_window_title = (
            detected_window_title.strip()
            if isinstance(detected_window_title, str) and detected_window_title.strip()
            else None
        )
        await self._reconcile_adapter()

    async def dispose(self) -> None:
        if self._active_adapter is not None:
            await self._active_adapter.stop()
            self._active_adapter = None
            self._active_adapter_type = None

    def snapshot_payload(self) -> dict[str, Any] | None:
        if self._active_adapter is None:
            return TelemetryAdapterSnapshot().to_payload()
        return self._active_adapter.current_snapshot().to_payload()

    def build_capability_advice(
        self, capabilities: Iterable[dict[str, Any]]
    ) -> dict[str, TelemetryCapabilityAdvice]:
        if self._active_adapter is None:
            return {}
        return self._active_adapter.build_capability_advice(capabilities)

    async def _reconcile_adapter(self) -> None:
        if not self._settings.enabled or self._profile_id is None:
            if self._active_adapter is not None:
                await self._active_adapter.stop()
                self._active_adapter = None
                self._active_adapter_type = None
            return

        adapter_type = self._select_adapter_type()
        if adapter_type is None:
            if self._active_adapter is not None:
                await self._active_adapter.stop()
                self._active_adapter = None
                self._active_adapter_type = None
            return

        if (
            self._active_adapter is not None
            and self._active_adapter_type is not adapter_type
        ):
            await self._active_adapter.stop()
            self._active_adapter = None
            self._active_adapter_type = None

        if self._active_adapter is None:
            self._active_adapter = adapter_type(
                self._on_event,
                self._on_state_change,
            )
            self._active_adapter_type = adapter_type
        await self._active_adapter.start(self._settings)
        await maybe_update_adapter_runtime_context(
            self._active_adapter,
            profile_id=self._profile_id,
            profile_display_name=self._profile_display_name,
            profile_process_names=self._profile_process_names,
            detected_process_name=self._detected_process_name,
            detected_window_title=self._detected_window_title,
        )

    def _select_adapter_type(self) -> Optional[type[Any]]:
        best_match: Optional[type[Any]] = None
        best_score = 0
        for adapter_type in DEFAULT_ADAPTER_TYPES:
            match_score = getattr(adapter_type, "match_score", None)
            if callable(match_score):
                score = int(
                    match_score(
                        profile_id=self._profile_id,
                        profile_display_name=self._profile_display_name,
                        profile_process_names=self._profile_process_names,
                        detected_process_name=self._detected_process_name,
                        detected_window_title=self._detected_window_title,
                    )
                )
            else:
                score = self._legacy_match_score(adapter_type)
            if score > best_score:
                best_score = score
                best_match = adapter_type
        return best_match

    def _legacy_match_score(self, adapter_type: type[Any]) -> int:
        supported_process_names = {
            str(item).strip().lower()
            for item in getattr(adapter_type, "supported_process_names", ())
            if isinstance(item, str) and item.strip()
        }
        if not supported_process_names:
            return 0
        process_names = {item.lower() for item in self._profile_process_names}
        if process_names.intersection(supported_process_names):
            return 100
        if (
            self._detected_process_name
            and self._detected_process_name.lower() in supported_process_names
        ):
            return 100
        return 0
