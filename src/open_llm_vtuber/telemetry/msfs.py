from __future__ import annotations

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

from .common import BaseTelemetryAdapter, clip_summary, isoformat, normalize_token_text, utcnow
from .models import (
    StructuredTelemetrySnapshot,
    TelemetryCapabilityAdvice,
    TelemetryEvent,
    TelemetryHighlight,
    TelemetrySettings,
)

POLL_INTERVAL_SECONDS = 1.5
STALE_AFTER_SECONDS = 15


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_altitude(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{round(value):,} ft"


def _format_speed(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{round(value)} kts"


def _format_vertical_speed(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    sign = "+" if value > 0 else ""
    return f"{sign}{round(value)} fpm"


def _format_position(latitude: Optional[float], longitude: Optional[float]) -> Optional[str]:
    if latitude is None or longitude is None:
        return None
    return f"{latitude:.3f}, {longitude:.3f}"


def _normalize_heading(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if abs(value) <= math.pi * 2 + 0.01:
        value = math.degrees(value)
    return round((value + 360.0) % 360.0, 1)


def _compute_flight_phase(on_ground: Optional[bool], altitude_agl_ft: Optional[float]) -> Optional[str]:
    if on_ground is None:
        return None
    if on_ground:
        return "ground"
    if altitude_agl_ft is not None and altitude_agl_ft < 1500:
        return "departure_or_approach"
    return "airborne"


@dataclass(slots=True)
class MsfsRuntimeContext:
    profile_id: Optional[str] = None
    profile_display_name: Optional[str] = None
    detected_process_name: Optional[str] = None
    detected_window_title: Optional[str] = None


class MsfsStateAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._session = {
            "aircraftTitle": None,
            "onGround": None,
            "flightPhase": None,
            "windowTitle": None,
            "profileDisplayName": None,
        }
        self._ship = {
            "landingGearDeployed": None,
            "isGearRetractable": None,
        }
        self._navigation = {
            "latitude": None,
            "longitude": None,
            "headingTrueDegrees": None,
            "nextWaypoint": None,
        }
        self._world = {
            "altitudeFt": None,
            "altitudeAglFt": None,
            "airspeedKts": None,
            "verticalSpeedFpm": None,
            "stallWarning": None,
            "overspeedWarning": None,
            "autopilotMaster": None,
        }
        self._last_update_at: Optional[datetime] = None
        self._last_source_timestamp: Optional[datetime] = None

    def set_runtime_context(self, context: MsfsRuntimeContext) -> None:
        if context.profile_display_name:
            self._session["profileDisplayName"] = context.profile_display_name
        if context.detected_window_title:
            self._session["windowTitle"] = " ".join(
                context.detected_window_title.split()
            ).strip()
        self._touch(None)

    def snapshot(self) -> StructuredTelemetrySnapshot:
        now = utcnow()
        stale = True
        if self._last_update_at is not None:
            stale = (now - self._last_update_at).total_seconds() > STALE_AFTER_SECONDS
        confidence = 0.0
        if self._last_update_at is not None:
            confidence = 0.88 if not stale else 0.5

        highlights: list[TelemetryHighlight] = []
        if self._session.get("aircraftTitle"):
            highlights.append(
                TelemetryHighlight(
                    key="aircraft",
                    label="Aircraft",
                    value=str(self._session["aircraftTitle"]),
                )
            )
        if self._session.get("flightPhase"):
            highlights.append(
                TelemetryHighlight(
                    key="phase",
                    label="Flight state",
                    value=str(self._session["flightPhase"]).replace("_", " ").title(),
                )
            )
        altitude_text = _format_altitude(_as_float(self._world.get("altitudeFt")))
        if altitude_text:
            highlights.append(
                TelemetryHighlight(
                    key="altitude",
                    label="Altitude",
                    value=altitude_text,
                )
            )
        speed_text = _format_speed(_as_float(self._world.get("airspeedKts")))
        if speed_text:
            highlights.append(
                TelemetryHighlight(
                    key="airspeed",
                    label="Airspeed",
                    value=speed_text,
                )
            )
        vertical_speed_text = _format_vertical_speed(
            _as_float(self._world.get("verticalSpeedFpm"))
        )
        if vertical_speed_text:
            highlights.append(
                TelemetryHighlight(
                    key="vertical_speed",
                    label="Vertical speed",
                    value=vertical_speed_text,
                )
            )
        if self._navigation.get("nextWaypoint"):
            highlights.append(
                TelemetryHighlight(
                    key="waypoint",
                    label="Next waypoint",
                    value=str(self._navigation["nextWaypoint"]),
                )
            )
        if self._world.get("stallWarning"):
            highlights.append(
                TelemetryHighlight(
                    key="stall",
                    label="Stall warning",
                    value="Active",
                    level="danger",
                )
            )
        elif self._world.get("overspeedWarning"):
            highlights.append(
                TelemetryHighlight(
                    key="overspeed",
                    label="Overspeed warning",
                    value="Active",
                    level="warning",
                )
            )

        return StructuredTelemetrySnapshot(
            source_timestamp=isoformat(self._last_source_timestamp),
            last_update_timestamp=isoformat(self._last_update_at),
            confidence=round(confidence, 2),
            stale=stale,
            session=dict(self._session),
            ship=dict(self._ship),
            navigation=dict(self._navigation),
            combat={},
            missions={},
            world=dict(self._world),
            automation={},
            highlights=highlights,
        )

    def apply_sample(
        self,
        sample: dict[str, Any],
        settings: TelemetrySettings,
    ) -> list[TelemetryEvent]:
        del settings
        previous_on_ground = self._session.get("onGround")
        previous_stall = self._world.get("stallWarning")
        previous_overspeed = self._world.get("overspeedWarning")
        previous_waypoint = self._navigation.get("nextWaypoint")

        aircraft_title = clip_summary(str(sample.get("title") or ""), limit=96) or None
        if aircraft_title:
            self._session["aircraftTitle"] = aircraft_title

        on_ground = _as_bool(sample.get("onGround"))
        altitude_ft = _as_float(sample.get("altitudeFt"))
        altitude_agl_ft = _as_float(sample.get("altitudeAglFt"))
        airspeed_kts = _as_float(sample.get("airspeedKts"))
        vertical_speed_fpm = _as_float(sample.get("verticalSpeedFpm"))
        gear_down = _as_bool(sample.get("gearDown"))
        gear_retractable = _as_bool(sample.get("gearRetractable"))
        stall_warning = _as_bool(sample.get("stallWarning"))
        overspeed_warning = _as_bool(sample.get("overspeedWarning"))
        autopilot_master = _as_bool(sample.get("autopilotMaster"))
        next_waypoint = clip_summary(str(sample.get("nextWaypoint") or ""), limit=48) or None

        self._session["onGround"] = on_ground
        self._session["flightPhase"] = _compute_flight_phase(on_ground, altitude_agl_ft)
        self._ship["landingGearDeployed"] = gear_down
        self._ship["isGearRetractable"] = gear_retractable
        self._navigation["latitude"] = _as_float(sample.get("latitude"))
        self._navigation["longitude"] = _as_float(sample.get("longitude"))
        self._navigation["headingTrueDegrees"] = _normalize_heading(
            _as_float(sample.get("headingTrue"))
        )
        self._navigation["nextWaypoint"] = next_waypoint
        self._world["altitudeFt"] = altitude_ft
        self._world["altitudeAglFt"] = altitude_agl_ft
        self._world["airspeedKts"] = airspeed_kts
        self._world["verticalSpeedFpm"] = vertical_speed_fpm
        self._world["stallWarning"] = stall_warning
        self._world["overspeedWarning"] = overspeed_warning
        self._world["autopilotMaster"] = autopilot_master

        events: list[TelemetryEvent] = []
        if previous_on_ground is True and on_ground is False:
            events.append(
                TelemetryEvent(
                    event_type="msfs.takeoff",
                    summary="Takeoff detected.",
                    novelty_key="msfs.takeoff",
                    importance=5,
                    urgency=3,
                    confidence=0.9,
                    cooldown_ms=30_000,
                    conversational=True,
                )
            )
        elif previous_on_ground is False and on_ground is True:
            events.append(
                TelemetryEvent(
                    event_type="msfs.landing",
                    summary="Landing detected.",
                    novelty_key="msfs.landing",
                    importance=5,
                    urgency=3,
                    confidence=0.9,
                    cooldown_ms=30_000,
                    conversational=True,
                )
            )

        if previous_stall is False and stall_warning is True:
            events.append(
                TelemetryEvent(
                    event_type="msfs.stall.warning",
                    summary="Stall warning is active.",
                    novelty_key="msfs.stall.warning",
                    importance=9,
                    urgency=9,
                    confidence=0.95,
                    cooldown_ms=10_000,
                    high_priority=True,
                )
            )
        if previous_overspeed is False and overspeed_warning is True:
            events.append(
                TelemetryEvent(
                    event_type="msfs.overspeed.warning",
                    summary="Overspeed warning is active.",
                    novelty_key="msfs.overspeed.warning",
                    importance=8,
                    urgency=8,
                    confidence=0.93,
                    cooldown_ms=10_000,
                    high_priority=True,
                )
            )
        if (
            next_waypoint
            and next_waypoint != previous_waypoint
            and next_waypoint.lower() not in {"none", "null"}
        ):
            events.append(
                TelemetryEvent(
                    event_type="msfs.navigation.waypoint",
                    summary=f"Next waypoint updated to {next_waypoint}.",
                    novelty_key=f"msfs.waypoint:{next_waypoint.lower()}",
                    importance=3,
                    urgency=2,
                    confidence=0.82,
                    cooldown_ms=45_000,
                    conversational=True,
                )
            )

        self._touch(None)
        return events

    def _touch(self, source_timestamp: Optional[datetime]) -> None:
        self._last_update_at = utcnow()
        if source_timestamp is not None:
            self._last_source_timestamp = source_timestamp


class MicrosoftFlightSimulatorTelemetryAdapter(BaseTelemetryAdapter):
    adapter_id = "msfs-simconnect"
    game_id = "microsoft-flight-simulator"
    game_display_name = "Microsoft Flight Simulator"
    source_kind = "simconnect"
    supported_process_names = (
        "flightsimulator.exe",
        "microsoft flight simulator",
        "microsoft flight simulator 2024",
    )

    def __init__(
        self,
        on_event: Callable[[TelemetryEvent], Awaitable[None]],
        on_state_change: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(on_event, on_state_change)
        self._state = MsfsStateAccumulator()
        self._runtime_context = MsfsRuntimeContext()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="msfs-telemetry")
        self._simconnect: Any = None
        self._aircraft_requests: Any = None

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
        del profile_process_names
        normalized_profile = " ".join(
            filter(
                None,
                [
                    normalize_token_text(profile_id or ""),
                    normalize_token_text(profile_display_name or ""),
                ],
            )
        )
        normalized_title = normalize_token_text(detected_window_title or "")
        normalized_process = normalize_token_text(detected_process_name or "")

        if normalized_process == "flightsimulator.exe":
            return 100
        if "microsoft flight simulator" in normalized_title or "msfs" in normalized_title:
            return 96
        if "microsoft flight simulator" in normalized_profile or "msfs" in normalized_profile:
            return 85
        return 0

    async def update_runtime_context(
        self,
        *,
        profile_id: Optional[str],
        profile_display_name: Optional[str],
        profile_process_names: Iterable[str],
        detected_process_name: Optional[str],
        detected_window_title: Optional[str],
    ) -> None:
        del profile_process_names
        self._runtime_context = MsfsRuntimeContext(
            profile_id=profile_id,
            profile_display_name=profile_display_name,
            detected_process_name=detected_process_name,
            detected_window_title=detected_window_title,
        )
        self._state.set_runtime_context(self._runtime_context)
        await self._emit_state_if_changed()

    async def _reset_runtime_state(self) -> None:
        self._state.reset()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._disconnect_sync)

    def _current_structured_snapshot(self) -> StructuredTelemetrySnapshot:
        return self._state.snapshot()

    async def _run(self) -> None:
        while not self._closed:
            self._snapshot.source_directory_status = "automatic"
            self._snapshot.source_directory_label = "Local SimConnect"
            self._snapshot.current_source = (
                self._state.snapshot().session.get("aircraftTitle") or "Live aircraft data"
            )
            try:
                loop = asyncio.get_running_loop()
                sample = await loop.run_in_executor(self._executor, self._poll_sync)
                self._snapshot.status = "running"
                self._snapshot.error = None
                self._snapshot.current_source = (
                    clip_summary(str(sample.get("title") or ""), limit=96)
                    or "Live aircraft data"
                )
                for event in self._state.apply_sample(sample, self._settings):
                    await self._emit_event(event)
            except Exception as exc:
                self._snapshot.status = "error"
                self._snapshot.error = str(exc)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, self._disconnect_sync)
            await self._emit_state_if_changed()
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def build_capability_advice(
        self, capabilities: Iterable[dict[str, Any]]
    ) -> dict[str, TelemetryCapabilityAdvice]:
        snapshot = self._state.snapshot()
        session = snapshot.session
        ship = snapshot.ship
        world = snapshot.world

        advice: dict[str, TelemetryCapabilityAdvice] = {}
        for capability in capabilities:
            command_id = str(
                capability.get("command_id") or capability.get("commandId") or ""
            )
            label = str(capability.get("label") or "")
            text = normalize_token_text(f"{command_id} {label}")
            entry = TelemetryCapabilityAdvice()

            if "landing gear" in text or text.endswith(" gear") or " gear " in text:
                wants_retract = any(word in text for word in ("retract", "raise", "stow", "gear up"))
                wants_extend = any(word in text for word in ("extend", "deploy", "lower", "gear down"))
                on_ground = bool(session.get("onGround"))
                gear_down = ship.get("landingGearDeployed")
                altitude_agl = _as_float(world.get("altitudeAglFt")) or 0.0
                retractable = ship.get("isGearRetractable")

                if retractable is False:
                    entry.available_override = False
                    entry.blocked_reason = "This aircraft does not have retractable gear."
                elif wants_retract:
                    if gear_down is False:
                        entry.available_override = False
                        entry.blocked_reason = "Landing gear is already retracted."
                    elif on_ground:
                        entry.avoid_reason = "Retracting the gear on the ground is not advisable."
                    elif altitude_agl > 1500:
                        entry.recommended = True
                elif wants_extend:
                    if gear_down is True:
                        entry.available_override = False
                        entry.blocked_reason = "Landing gear is already deployed."
                    elif on_ground or altitude_agl < 1500:
                        entry.recommended = True
            elif "autopilot" in text:
                if session.get("onGround") is True:
                    entry.avoid_reason = "Autopilot changes are usually safer once airborne."
                else:
                    entry.recommended = True

            if (
                entry.available_override is not None
                or entry.recommended
                or entry.avoid_reason
                or entry.blocked_reason
            ):
                advice[command_id] = entry
        return advice

    def _disconnect_sync(self) -> None:
        if self._simconnect is not None:
            try:
                self._simconnect.exit()
            except Exception:
                pass
        self._simconnect = None
        self._aircraft_requests = None

    def _poll_sync(self) -> dict[str, Any]:
        if self._aircraft_requests is None or self._simconnect is None:
            try:
                from SimConnect import AircraftRequests, SimConnect
            except Exception as exc:  # pragma: no cover - import path varies
                raise RuntimeError(
                    "MSFS telemetry needs the optional Python package 'SimConnect'. Install it in the backend environment and start Microsoft Flight Simulator."
                ) from exc

            try:
                self._simconnect = SimConnect()
                self._aircraft_requests = AircraftRequests(self._simconnect, _time=0)
            except Exception as exc:
                self._disconnect_sync()
                raise RuntimeError(
                    "Unable to connect to Microsoft Flight Simulator through SimConnect."
                ) from exc

        requests = self._aircraft_requests

        def get(name: str) -> Any:
            try:
                return requests.get(name)
            except Exception:
                return None

        return {
            "title": get("TITLE"),
            "onGround": get("SIM_ON_GROUND"),
            "altitudeFt": get("PLANE_ALTITUDE"),
            "altitudeAglFt": get("PLANE_ALT_ABOVE_GROUND"),
            "airspeedKts": get("AIRSPEED_INDICATED"),
            "verticalSpeedFpm": get("VERTICAL_SPEED"),
            "nextWaypoint": get("GPS_WP_NEXT_ID"),
            "gearDown": get("GEAR_HANDLE_POSITION"),
            "gearRetractable": get("IS_GEAR_RETRACTABLE"),
            "stallWarning": get("STALL_WARNING"),
            "overspeedWarning": get("OVERSPEED_WARNING"),
            "autopilotMaster": get("AUTOPILOT_MASTER"),
            "latitude": get("PLANE_LATITUDE"),
            "longitude": get("PLANE_LONGITUDE"),
            "headingTrue": get("PLANE_HEADING_DEGREES_TRUE"),
        }
