from __future__ import annotations

import asyncio
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
    TelemetryEvent,
    TelemetryHighlight,
    TelemetrySettings,
    redact_directory_label,
)

STATUS_FLAGS: dict[int, str] = {
    0: "docked",
    1: "landed",
    2: "landing_gear_deployed",
    3: "shields_up",
    4: "supercruise",
    5: "flight_assist_off",
    6: "hardpoints_deployed",
    7: "in_wing",
    8: "lights_on",
    9: "cargo_scoop_deployed",
    10: "silent_running",
    11: "scooping_fuel",
    16: "mass_locked",
    17: "fsd_charging",
    19: "low_fuel",
    20: "overheating",
    22: "in_danger",
    23: "being_interdicted",
    24: "in_main_ship",
    25: "in_fighter",
    26: "in_srv",
    27: "analysis_mode",
}

STATUS_FLAGS2: dict[int, str] = {
    0: "on_foot",
    1: "in_taxi",
    2: "in_multicrew",
    3: "on_foot_in_station",
    4: "on_foot_on_planet",
    12: "glide_mode",
    13: "on_foot_in_hangar",
    14: "on_foot_social_space",
    15: "on_foot_exterior",
}

DEFAULT_JOURNAL_DIR = (
    Path.home() / "Saved Games" / "Frontier Developments" / "Elite Dangerous"
)
JOURNAL_GLOB = "Journal.*.log"
STATUS_FILE_NAME = "Status.json"
POLL_INTERVAL_SECONDS = 0.5
STALE_AFTER_SECONDS = 30
HEAT_EVENT_COOLDOWN_MS = 20_000
DOCKING_EVENT_COOLDOWN_MS = 12_000
DANGER_EVENT_COOLDOWN_MS = 20_000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _as_percentage(value: Any) -> Optional[float]:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    if numeric <= 1.5:
        numeric *= 100.0
    return round(max(0.0, min(100.0, numeric)), 1)


def _decode_flags(value: Any, mapping: dict[int, str]) -> dict[str, bool]:
    if not isinstance(value, int):
        return {}
    return {name: bool(value & (1 << bit)) for bit, name in mapping.items()}


def _mode_category(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered == "open":
        return "open"
    if lowered == "solo":
        return "solo"
    if "group" in lowered:
        return "private_group" if "private" in lowered else "group"
    return "unknown"


def _clip_summary(value: str, limit: int = 160) -> str:
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _normalize_command_text(command_id: str, label: str) -> str:
    return f"{command_id} {label}".strip().lower().replace("_", " ")


def discover_elite_journal_directory(
    settings: TelemetrySettings,
) -> tuple[Optional[Path], str, Optional[str]]:
    if settings.automatic_journal_discovery:
        if DEFAULT_JOURNAL_DIR.exists():
            return (
                DEFAULT_JOURNAL_DIR,
                "automatic",
                redact_directory_label(DEFAULT_JOURNAL_DIR),
            )
        return (
            DEFAULT_JOURNAL_DIR,
            "missing",
            redact_directory_label(DEFAULT_JOURNAL_DIR),
        )
    if settings.manual_journal_directory:
        path = Path(settings.manual_journal_directory)
        if path.exists():
            return path, "manual", redact_directory_label(path)
        return path, "missing", redact_directory_label(path)
    return None, "unavailable", None


@dataclass(slots=True)
class JournalPollResult:
    events: list[dict[str, Any]]
    current_file: Optional[str]
    parse_errors: int = 0


class JournalTailer:
    def __init__(self) -> None:
        self._directory: Optional[Path] = None
        self._current_path: Optional[Path] = None
        self._position = 0
        self._partial = b""

    def reset(self) -> None:
        self._directory = None
        self._current_path = None
        self._position = 0
        self._partial = b""

    async def poll(self, directory: Path) -> JournalPollResult:
        return await asyncio.to_thread(self._poll_sync, directory)

    def _poll_sync(self, directory: Path) -> JournalPollResult:
        if self._directory != directory:
            self.reset()
            self._directory = directory

        current_path = self._select_latest_file(directory)
        if current_path is None:
            self._current_path = None
            self._position = 0
            self._partial = b""
            return JournalPollResult(events=[], current_file=None)

        if self._current_path != current_path:
            self._current_path = current_path
            self._position = 0
            self._partial = b""

        try:
            size = self._current_path.stat().st_size
        except FileNotFoundError:
            self._current_path = None
            self._position = 0
            self._partial = b""
            return JournalPollResult(events=[], current_file=None)

        if size < self._position:
            self._position = 0
            self._partial = b""

        try:
            with self._current_path.open("rb") as handle:
                handle.seek(self._position)
                chunk = handle.read()
                self._position = handle.tell()
        except FileNotFoundError:
            self._current_path = None
            self._position = 0
            self._partial = b""
            return JournalPollResult(events=[], current_file=None)

        if not chunk:
            return JournalPollResult(
                events=[],
                current_file=self._current_path.name,
            )

        data = self._partial + chunk
        parse_errors = 0
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith((b"\n", b"\r")):
            self._partial = lines.pop()
        else:
            self._partial = b""

        parsed_events: list[dict[str, Any]] = []
        for raw_line in lines:
            text = raw_line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if isinstance(parsed, dict):
                parsed_events.append(parsed)

        return JournalPollResult(
            events=parsed_events,
            current_file=self._current_path.name,
            parse_errors=parse_errors,
        )

    def _select_latest_file(self, directory: Path) -> Optional[Path]:
        try:
            candidates = list(directory.glob(JOURNAL_GLOB))
        except FileNotFoundError:
            return None
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda path: (
                path.stat().st_mtime_ns if path.exists() else 0,
                path.name,
            ),
        )


class StatusReader:
    def __init__(self) -> None:
        self._directory: Optional[Path] = None
        self._last_mtime_ns: Optional[int] = None
        self._last_size: Optional[int] = None

    def reset(self) -> None:
        self._directory = None
        self._last_mtime_ns = None
        self._last_size = None

    async def poll(self, directory: Path) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self._poll_sync, directory)

    def _poll_sync(self, directory: Path) -> Optional[dict[str, Any]]:
        if self._directory != directory:
            self.reset()
            self._directory = directory

        path = directory / STATUS_FILE_NAME
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None

        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == (self._last_mtime_ns, self._last_size):
            return None

        self._last_mtime_ns, self._last_size = signature
        try:
            text = path.read_text(encoding="utf-8")
            parsed = json.loads(text)
        except (OSError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None


class EliteDangerousStateAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._session = {
            "gameMode": None,
            "commanderName": None,
            "modeCategory": None,
            "shipType": None,
            "shipName": None,
            "shipIdent": None,
            "starSystem": None,
            "stationName": None,
            "bodyName": None,
            "docked": None,
            "landed": None,
            "activityMode": None,
        }
        self._ship = {
            "hullPercent": None,
            "shieldsUp": None,
            "fuelLevel": None,
            "fuelCapacity": None,
            "cargoCount": None,
            "cargoCapacity": None,
            "landingGearDeployed": None,
            "hardpointsDeployed": None,
            "cargoScoopDeployed": None,
            "lightsOn": None,
            "silentRunning": None,
            "flightAssistOff": None,
            "heatWarning": None,
            "lowFuelWarning": None,
            "fsdStatus": None,
            "supercruise": None,
        }
        self._navigation = {
            "currentSystem": None,
            "destinationSystem": None,
            "jumpInProgress": None,
            "lastJumpCompletedAt": None,
            "dockingRequested": None,
            "dockingGranted": None,
            "dockingDenied": None,
            "docked": None,
            "undockedAt": None,
        }
        self._combat = {
            "inDanger": None,
            "interdicted": None,
            "interdicting": None,
            "shieldsFailed": None,
            "lastHullThreshold": None,
            "targetName": None,
            "targetShieldHealth": None,
            "bountyAwarded": None,
            "combatBondAwarded": None,
            "lastDeathAt": None,
        }
        self._missions = {
            "activeMissionCount": 0,
            "lastAcceptedMission": None,
            "lastCompletedMission": None,
            "lastFailedMission": None,
            "lastCollectedMaterial": None,
            "lastDiscovery": None,
        }
        self._last_source_timestamp: Optional[datetime] = None
        self._last_update_at: Optional[datetime] = None
        self._status_semantics: dict[str, Any] = {}
        self._triggered_hull_thresholds: set[int] = set()
        self._low_fuel_announced = False
        self._critical_fuel_announced = False
        self._last_event_at: dict[str, datetime] = {}

    def snapshot(self) -> StructuredTelemetrySnapshot:
        now = _utcnow()
        stale = True
        if self._last_update_at is not None:
            stale = (now - self._last_update_at).total_seconds() > STALE_AFTER_SECONDS
        confidence = 0.0
        if self._last_update_at is not None:
            confidence = 0.85 if not stale else 0.4
            if self._last_source_timestamp is not None:
                confidence = 0.95 if not stale else 0.5
        highlights: list[TelemetryHighlight] = []
        star_system = self._session.get("starSystem")
        if star_system:
            highlights.append(
                TelemetryHighlight(
                    key="system",
                    label="Star system",
                    value=str(star_system),
                )
            )
        ship_name = self._session.get("shipName") or self._session.get("shipType")
        if ship_name:
            highlights.append(
                TelemetryHighlight(
                    key="ship",
                    label="Ship",
                    value=str(ship_name),
                )
            )
        if self._session.get("docked") is not None:
            highlights.append(
                TelemetryHighlight(
                    key="docked",
                    label="Docked",
                    value="Yes" if self._session.get("docked") else "No",
                )
            )
        if self._combat.get("inDanger") is not None:
            highlights.append(
                TelemetryHighlight(
                    key="danger",
                    label="Danger",
                    value="Yes" if self._combat.get("inDanger") else "No",
                    level="danger" if self._combat.get("inDanger") else "default",
                )
            )
        hull_percent = self._ship.get("hullPercent")
        if hull_percent is not None:
            highlights.append(
                TelemetryHighlight(
                    key="hull",
                    label="Hull",
                    value=f"{hull_percent}%",
                    level="danger"
                    if hull_percent <= 25
                    else ("warning" if hull_percent <= 50 else "default"),
                )
            )
        fuel_level = self._ship.get("fuelLevel")
        fuel_capacity = self._ship.get("fuelCapacity")
        if fuel_level is not None and fuel_capacity:
            ratio = float(fuel_level) / float(fuel_capacity)
            highlights.append(
                TelemetryHighlight(
                    key="fuel",
                    label="Fuel",
                    value=f"{round(ratio * 100)}%",
                    level="danger"
                    if ratio <= 0.10
                    else ("warning" if ratio <= 0.25 else "default"),
                )
            )
        return StructuredTelemetrySnapshot(
            source_timestamp=_isoformat(self._last_source_timestamp),
            last_update_timestamp=_isoformat(self._last_update_at),
            confidence=round(confidence, 2),
            stale=stale,
            session=dict(self._session),
            ship=dict(self._ship),
            navigation=dict(self._navigation),
            combat=dict(self._combat),
            missions=dict(self._missions),
            world={},
            automation={},
            highlights=highlights,
        )

    def apply_status(
        self,
        payload: dict[str, Any],
        settings: TelemetrySettings,
    ) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        source_timestamp = _parse_time(payload.get("timestamp"))
        semantics = _decode_flags(payload.get("Flags"), STATUS_FLAGS)
        semantics.update(_decode_flags(payload.get("Flags2"), STATUS_FLAGS2))

        fuel = payload.get("Fuel") if isinstance(payload.get("Fuel"), dict) else {}
        fuel_main = _safe_float(fuel.get("FuelMain"))
        if fuel_main is not None:
            self._ship["fuelLevel"] = round(fuel_main, 3)

        if semantics:
            self._ship["landingGearDeployed"] = semantics.get("landing_gear_deployed")
            self._ship["hardpointsDeployed"] = semantics.get("hardpoints_deployed")
            self._ship["cargoScoopDeployed"] = semantics.get("cargo_scoop_deployed")
            self._ship["lightsOn"] = semantics.get("lights_on")
            self._ship["silentRunning"] = semantics.get("silent_running")
            self._ship["flightAssistOff"] = semantics.get("flight_assist_off")
            self._ship["heatWarning"] = semantics.get("overheating")
            self._ship["lowFuelWarning"] = semantics.get("low_fuel")
            self._ship["supercruise"] = semantics.get("supercruise")
            self._ship["shieldsUp"] = semantics.get("shields_up")
            self._session["docked"] = semantics.get("docked")
            self._session["landed"] = semantics.get("landed")
            self._navigation["docked"] = semantics.get("docked")
            self._combat["inDanger"] = semantics.get("in_danger")
            self._session["activityMode"] = (
                "analysis"
                if semantics.get("analysis_mode")
                else (
                    "combat"
                    if semantics.get("hardpoints_deployed")
                    else self._session["activityMode"]
                )
            )
            if semantics.get("fsd_charging"):
                self._ship["fsdStatus"] = "charging"
            elif semantics.get("supercruise"):
                self._ship["fsdStatus"] = "supercruise"
            elif self._navigation["jumpInProgress"]:
                self._ship["fsdStatus"] = "jumping"
            else:
                self._ship["fsdStatus"] = None

        previous = self._status_semantics
        if (
            previous.get("supercruise") is False
            and semantics.get("supercruise") is True
        ):
            events.append(
                TelemetryEvent(
                    event_type="elite.supercruise.entered",
                    summary="Entered supercruise.",
                    novelty_key="elite.supercruise.entered",
                    importance=4,
                    urgency=3,
                    confidence=0.92,
                    conversational=True,
                )
            )
        if (
            previous.get("supercruise") is True
            and semantics.get("supercruise") is False
        ):
            events.append(
                TelemetryEvent(
                    event_type="elite.supercruise.exited",
                    summary="Exited supercruise.",
                    novelty_key="elite.supercruise.exited",
                    importance=4,
                    urgency=3,
                    confidence=0.92,
                    conversational=True,
                )
            )

        if previous.get("shields_up") is True and semantics.get("shields_up") is False:
            self._combat["shieldsFailed"] = True
            events.append(
                TelemetryEvent(
                    event_type="elite.shields.failed",
                    summary="Shields failed.",
                    novelty_key="elite.shields.failed",
                    importance=9,
                    urgency=9,
                    confidence=0.97,
                    cooldown_ms=10_000,
                    high_priority=True,
                )
            )
        if previous.get("shields_up") is False and semantics.get("shields_up") is True:
            self._combat["shieldsFailed"] = False
            events.append(
                TelemetryEvent(
                    event_type="elite.shields.restored",
                    summary="Shields restored.",
                    novelty_key="elite.shields.restored",
                    importance=6,
                    urgency=5,
                    confidence=0.97,
                    conversational=True,
                )
            )

        if previous.get("in_danger") is False and semantics.get("in_danger") is True:
            if self._event_ready("danger.started", DANGER_EVENT_COOLDOWN_MS):
                events.append(
                    TelemetryEvent(
                        event_type="elite.danger.started",
                        summary="Danger detected.",
                        novelty_key="elite.danger.started",
                        importance=9,
                        urgency=9,
                        confidence=0.96,
                        cooldown_ms=DANGER_EVENT_COOLDOWN_MS,
                        high_priority=True,
                    )
                )
                if semantics.get("hardpoints_deployed"):
                    events.append(
                        TelemetryEvent(
                            event_type="elite.combat.entered",
                            summary="Combat state entered.",
                            novelty_key="elite.combat.entered",
                            importance=8,
                            urgency=8,
                            confidence=0.93,
                            cooldown_ms=DANGER_EVENT_COOLDOWN_MS,
                            high_priority=True,
                        )
                    )
        if previous.get("in_danger") is True and semantics.get("in_danger") is False:
            events.append(
                TelemetryEvent(
                    event_type="elite.danger.ended",
                    summary="Danger has passed.",
                    novelty_key="elite.danger.ended",
                    importance=5,
                    urgency=4,
                    confidence=0.93,
                    conversational=True,
                )
            )

        if (
            previous.get("hardpoints_deployed") is False
            and semantics.get("hardpoints_deployed") is True
            and semantics.get("in_danger")
        ):
            events.append(
                TelemetryEvent(
                    event_type="elite.combat.entered",
                    summary="Hardpoints deployed during danger.",
                    novelty_key="elite.combat.entered.hardpoints",
                    importance=8,
                    urgency=8,
                    confidence=0.9,
                    cooldown_ms=DANGER_EVENT_COOLDOWN_MS,
                    high_priority=True,
                )
            )

        if semantics.get("being_interdicted") and self._event_ready(
            "interdiction", 12_000
        ):
            self._combat["interdicted"] = True
            events.append(
                TelemetryEvent(
                    event_type="elite.interdiction",
                    summary="Interdiction detected.",
                    novelty_key="elite.interdiction.status",
                    importance=10,
                    urgency=10,
                    confidence=0.98,
                    cooldown_ms=12_000,
                    high_priority=True,
                )
            )

        if semantics.get("overheating") and self._event_ready(
            "heat.warning", HEAT_EVENT_COOLDOWN_MS
        ):
            events.append(
                TelemetryEvent(
                    event_type="elite.heat.warning",
                    summary="Heat warning detected.",
                    novelty_key="elite.heat.warning",
                    importance=9,
                    urgency=9,
                    confidence=0.97,
                    cooldown_ms=HEAT_EVENT_COOLDOWN_MS,
                    high_priority=True,
                )
            )

        self._status_semantics = semantics
        events.extend(self._update_fuel_thresholds(settings))
        self._touch(source_timestamp)
        return events

    def apply_journal_entry(
        self,
        entry: dict[str, Any],
        settings: TelemetrySettings,
    ) -> list[TelemetryEvent]:
        event_name = str(entry.get("event") or "").strip()
        lowered = event_name.lower()
        source_timestamp = _parse_time(entry.get("timestamp"))
        events: list[TelemetryEvent] = []

        if lowered == "loadgame":
            self._session["gameMode"] = entry.get("GameMode")
            self._session["modeCategory"] = _mode_category(entry.get("GameMode"))
            self._session["commanderName"] = entry.get("Commander")
            self._session["shipType"] = entry.get("Ship")
            self._session["shipName"] = entry.get("ShipName")
            self._session["shipIdent"] = entry.get("ShipIdent")
            self._ship["fuelLevel"] = _safe_float(entry.get("FuelLevel"))
            fuel_capacity = entry.get("FuelCapacity")
            if isinstance(fuel_capacity, dict):
                self._ship["fuelCapacity"] = _safe_float(
                    fuel_capacity.get("Main")
                    or fuel_capacity.get("FuelMain")
                    or fuel_capacity.get("Capacity")
                )
            else:
                self._ship["fuelCapacity"] = _safe_float(fuel_capacity)
            self._reset_threshold_state()
            events.append(
                TelemetryEvent(
                    event_type="elite.session.started",
                    summary=f"Elite Dangerous session started in {entry.get('GameMode') or 'Unknown mode'}.",
                    novelty_key=f"elite.session.started:{entry.get('Commander') or 'unknown'}",
                    importance=5,
                    urgency=4,
                    confidence=0.95,
                    conversational=True,
                )
            )
        elif lowered == "shutdown":
            events.append(
                TelemetryEvent(
                    event_type="elite.session.ended",
                    summary="Elite Dangerous session ended.",
                    novelty_key="elite.session.ended",
                    importance=5,
                    urgency=4,
                    confidence=0.9,
                    conversational=True,
                )
            )
        elif lowered == "loadout":
            previous_ship = self._session.get("shipType"), self._session.get("shipName")
            self._session["shipType"] = entry.get("Ship")
            self._session["shipName"] = entry.get("ShipName")
            self._session["shipIdent"] = entry.get("ShipIdent")
            self._ship["cargoCapacity"] = _safe_float(entry.get("CargoCapacity"))
            if previous_ship != (
                self._session.get("shipType"),
                self._session.get("shipName"),
            ):
                self._reset_threshold_state()
                events.append(
                    TelemetryEvent(
                        event_type="elite.ship.changed",
                        summary="Ship loadout changed.",
                        novelty_key=f"elite.ship.changed:{self._session.get('shipType')}:{self._session.get('shipName')}",
                        importance=4,
                        urgency=3,
                        confidence=0.88,
                        conversational=True,
                    )
                )
        elif lowered == "cargo":
            inventory = entry.get("Inventory")
            if isinstance(inventory, list):
                self._ship["cargoCount"] = int(
                    sum(
                        int(item.get("Count") or 0)
                        for item in inventory
                        if isinstance(item, dict)
                    )
                )
        elif lowered == "location":
            previous_location = (
                self._session.get("starSystem"),
                self._session.get("stationName"),
                self._session.get("bodyName"),
            )
            self._apply_location(entry)
            current_location = (
                self._session.get("starSystem"),
                self._session.get("stationName"),
                self._session.get("bodyName"),
            )
            if previous_location != current_location:
                events.append(
                    TelemetryEvent(
                        event_type="elite.location.changed",
                        summary=f"Location updated to {self._session.get('starSystem') or 'Unknown system'}.",
                        novelty_key=f"elite.location.changed:{self._session.get('starSystem')}:{self._session.get('stationName')}:{self._session.get('bodyName')}",
                        importance=4,
                        urgency=3,
                        confidence=0.9,
                        conversational=True,
                    )
                )
        elif lowered == "fsdtarget":
            self._navigation["destinationSystem"] = entry.get("Name") or entry.get(
                "StarSystem"
            )
        elif lowered == "startjump":
            self._navigation["jumpInProgress"] = True
            self._navigation["destinationSystem"] = entry.get("StarSystem")
            self._ship["fsdStatus"] = "charging"
            if settings.commentary.announce_jump_events:
                events.append(
                    TelemetryEvent(
                        event_type="elite.jump.started",
                        summary=f"Jump started to {entry.get('StarSystem') or 'the next system'}.",
                        novelty_key=f"elite.jump.started:{entry.get('StarSystem')}",
                        importance=5,
                        urgency=4,
                        confidence=0.95,
                        conversational=True,
                    )
                )
        elif lowered == "fsdjump":
            self._navigation["jumpInProgress"] = False
            self._navigation["lastJumpCompletedAt"] = _isoformat(source_timestamp)
            self._ship["fsdStatus"] = None
            self._apply_location(entry)
            self._navigation["destinationSystem"] = None
            if entry.get("FuelLevel") is not None:
                self._ship["fuelLevel"] = _safe_float(entry.get("FuelLevel"))
            if settings.commentary.announce_jump_events:
                events.append(
                    TelemetryEvent(
                        event_type="elite.jump.completed",
                        summary=f"Jump completed into {entry.get('StarSystem') or 'a new system'}.",
                        novelty_key=f"elite.jump.completed:{entry.get('StarSystem')}",
                        importance=5,
                        urgency=4,
                        confidence=0.95,
                        conversational=True,
                    )
                )
        elif lowered == "supercruiseentry":
            self._ship["supercruise"] = True
        elif lowered == "supercruiseexit":
            self._ship["supercruise"] = False
            if entry.get("Body"):
                self._session["bodyName"] = entry.get("Body")
        elif lowered == "dockingrequested":
            self._navigation["dockingRequested"] = True
            self._navigation["dockingDenied"] = False
            if settings.commentary.announce_docking_events and self._event_ready(
                "docking.requested", DOCKING_EVENT_COOLDOWN_MS
            ):
                events.append(
                    TelemetryEvent(
                        event_type="elite.docking.requested",
                        summary="Docking requested.",
                        novelty_key="elite.docking.requested",
                        importance=5,
                        urgency=4,
                        confidence=0.94,
                        cooldown_ms=DOCKING_EVENT_COOLDOWN_MS,
                        conversational=True,
                    )
                )
        elif lowered == "dockinggranted":
            self._navigation["dockingRequested"] = True
            self._navigation["dockingGranted"] = True
            self._navigation["dockingDenied"] = False
            if settings.commentary.announce_docking_events and self._event_ready(
                "docking.granted", DOCKING_EVENT_COOLDOWN_MS
            ):
                events.append(
                    TelemetryEvent(
                        event_type="elite.docking.granted",
                        summary="Docking granted.",
                        novelty_key="elite.docking.granted",
                        importance=5,
                        urgency=4,
                        confidence=0.96,
                        cooldown_ms=DOCKING_EVENT_COOLDOWN_MS,
                        conversational=True,
                    )
                )
        elif lowered == "dockingdenied":
            self._navigation["dockingRequested"] = True
            self._navigation["dockingGranted"] = False
            self._navigation["dockingDenied"] = True
            if settings.commentary.announce_docking_events and self._event_ready(
                "docking.denied", DOCKING_EVENT_COOLDOWN_MS
            ):
                events.append(
                    TelemetryEvent(
                        event_type="elite.docking.denied",
                        summary="Docking denied.",
                        novelty_key="elite.docking.denied",
                        importance=9,
                        urgency=9,
                        confidence=0.96,
                        cooldown_ms=DOCKING_EVENT_COOLDOWN_MS,
                        high_priority=True,
                    )
                )
        elif lowered == "docked":
            self._apply_location(entry)
            self._session["docked"] = True
            self._navigation["docked"] = True
            self._navigation["dockingRequested"] = False
            self._navigation["dockingGranted"] = False
            self._navigation["dockingDenied"] = False
            if settings.commentary.announce_docking_events and self._event_ready(
                "docked", DOCKING_EVENT_COOLDOWN_MS
            ):
                events.append(
                    TelemetryEvent(
                        event_type="elite.docked",
                        summary=f"Docked at {entry.get('StationName') or 'the station'}.",
                        novelty_key=f"elite.docked:{entry.get('StationName')}",
                        importance=5,
                        urgency=4,
                        confidence=0.97,
                        cooldown_ms=DOCKING_EVENT_COOLDOWN_MS,
                        conversational=True,
                    )
                )
        elif lowered == "undocked":
            self._session["docked"] = False
            self._navigation["docked"] = False
            self._navigation["undockedAt"] = _isoformat(source_timestamp)
            if settings.commentary.announce_docking_events and self._event_ready(
                "undocked", DOCKING_EVENT_COOLDOWN_MS
            ):
                events.append(
                    TelemetryEvent(
                        event_type="elite.undocked",
                        summary="Undocked.",
                        novelty_key="elite.undocked",
                        importance=4,
                        urgency=3,
                        confidence=0.95,
                        cooldown_ms=DOCKING_EVENT_COOLDOWN_MS,
                        conversational=True,
                    )
                )
        elif lowered == "touchdown":
            self._session["landed"] = True
        elif lowered == "liftoff":
            self._session["landed"] = False
        elif lowered == "missionaccepted":
            mission_name = entry.get("Name") or entry.get("LocalisedName")
            self._missions["activeMissionCount"] = (
                int(self._missions.get("activeMissionCount") or 0) + 1
            )
            self._missions["lastAcceptedMission"] = mission_name
            if settings.commentary.announce_mission_events:
                events.append(
                    TelemetryEvent(
                        event_type="elite.mission.accepted",
                        summary=f"Mission accepted: {mission_name or 'Unknown mission'}.",
                        novelty_key=f"elite.mission.accepted:{mission_name}",
                        importance=4,
                        urgency=3,
                        confidence=0.9,
                        conversational=True,
                    )
                )
        elif lowered == "missioncompleted":
            mission_name = entry.get("Name") or entry.get("LocalisedName")
            self._missions["activeMissionCount"] = max(
                0, int(self._missions.get("activeMissionCount") or 0) - 1
            )
            self._missions["lastCompletedMission"] = mission_name
            if settings.commentary.announce_mission_events:
                events.append(
                    TelemetryEvent(
                        event_type="elite.mission.completed",
                        summary=f"Mission completed: {mission_name or 'Unknown mission'}.",
                        novelty_key=f"elite.mission.completed:{mission_name}",
                        importance=5,
                        urgency=4,
                        confidence=0.9,
                        conversational=True,
                    )
                )
        elif lowered == "missionfailed":
            mission_name = entry.get("Name") or entry.get("LocalisedName")
            self._missions["activeMissionCount"] = max(
                0, int(self._missions.get("activeMissionCount") or 0) - 1
            )
            self._missions["lastFailedMission"] = mission_name
            if settings.commentary.announce_mission_events:
                events.append(
                    TelemetryEvent(
                        event_type="elite.mission.failed",
                        summary=f"Mission failed: {mission_name or 'Unknown mission'}.",
                        novelty_key=f"elite.mission.failed:{mission_name}",
                        importance=8,
                        urgency=8,
                        confidence=0.9,
                        high_priority=True,
                    )
                )
        elif lowered == "materialcollected":
            material_name = (
                entry.get("Name_Localised")
                or entry.get("Name")
                or entry.get("Material")
            )
            self._missions["lastCollectedMaterial"] = material_name
            if settings.commentary.announce_material_collection:
                events.append(
                    TelemetryEvent(
                        event_type="elite.material.collected",
                        summary=f"Collected material: {material_name or 'unknown material'}.",
                        novelty_key=f"elite.material.collected:{material_name}",
                        importance=3,
                        urgency=2,
                        confidence=0.88,
                        conversational=True,
                    )
                )
        elif lowered in {"scan", "fssdiscoveryscan", "fssallbodiesfound"}:
            discovery_name = (
                entry.get("BodyName")
                or entry.get("SystemName")
                or entry.get("StarSystem")
            )
            self._missions["lastDiscovery"] = discovery_name
            if settings.commentary.announce_discoveries and self._event_ready(
                "discovery", settings.exploration_commentary_cooldown_ms
            ):
                events.append(
                    TelemetryEvent(
                        event_type="elite.discovery",
                        summary=f"Discovery update: {discovery_name or 'new scan data'}.",
                        novelty_key=f"elite.discovery:{discovery_name}",
                        importance=4,
                        urgency=2,
                        confidence=0.85,
                        cooldown_ms=settings.exploration_commentary_cooldown_ms,
                        conversational=True,
                    )
                )
        elif lowered == "hulldamage":
            hull_percent = _as_percentage(
                entry.get("Health") or entry.get("HullHealth") or entry.get("Hull")
            )
            if hull_percent is not None:
                self._ship["hullPercent"] = hull_percent
                events.extend(self._update_hull_thresholds(settings))
        elif lowered == "heatwarning":
            self._ship["heatWarning"] = True
            if self._event_ready("heat.warning", HEAT_EVENT_COOLDOWN_MS):
                events.append(
                    TelemetryEvent(
                        event_type="elite.heat.warning",
                        summary="Heat warning detected.",
                        novelty_key="elite.heat.warning.journal",
                        importance=9,
                        urgency=9,
                        confidence=0.95,
                        cooldown_ms=HEAT_EVENT_COOLDOWN_MS,
                        high_priority=True,
                    )
                )
        elif lowered == "interdicted":
            self._combat["interdicted"] = True
            if self._event_ready("interdiction", 12_000):
                events.append(
                    TelemetryEvent(
                        event_type="elite.interdiction",
                        summary="Interdicted.",
                        novelty_key="elite.interdicted",
                        importance=10,
                        urgency=10,
                        confidence=0.98,
                        cooldown_ms=12_000,
                        high_priority=True,
                    )
                )
        elif lowered == "interdiction":
            self._combat["interdicting"] = True
            if self._event_ready("interdiction", 12_000):
                events.append(
                    TelemetryEvent(
                        event_type="elite.interdiction",
                        summary="Interdiction in progress.",
                        novelty_key="elite.interdicting",
                        importance=8,
                        urgency=8,
                        confidence=0.9,
                        cooldown_ms=12_000,
                        high_priority=True,
                    )
                )
        elif lowered == "died":
            self._combat["lastDeathAt"] = _isoformat(source_timestamp)
            self._ship["hullPercent"] = 0.0
            self._reset_threshold_state()
            events.append(
                TelemetryEvent(
                    event_type="elite.death",
                    summary="Ship destroyed.",
                    novelty_key="elite.death",
                    importance=10,
                    urgency=10,
                    confidence=1.0,
                    high_priority=True,
                )
            )
        elif lowered == "resurrect":
            self._ship["hullPercent"] = 100.0
            self._combat["interdicted"] = False
            self._combat["interdicting"] = False
            self._combat["shieldsFailed"] = False
            self._reset_threshold_state()
        elif lowered in {"bounty", "commitcrime"}:
            self._combat["bountyAwarded"] = True
        elif lowered in {"factionkillbond", "capshipbond", "combatbond"}:
            self._combat["combatBondAwarded"] = True

        events.extend(self._update_fuel_thresholds(settings))
        self._touch(source_timestamp)
        return events

    def _apply_location(self, entry: dict[str, Any]) -> None:
        star_system = entry.get("StarSystem")
        station_name = entry.get("StationName")
        body_name = entry.get("Body") or entry.get("BodyName")
        docked = entry.get("Docked")
        if star_system is not None:
            self._session["starSystem"] = star_system
            self._navigation["currentSystem"] = star_system
        if station_name is not None:
            self._session["stationName"] = station_name
        if body_name is not None:
            self._session["bodyName"] = body_name
        if isinstance(docked, bool):
            self._session["docked"] = docked
            self._navigation["docked"] = docked

    def _touch(self, source_timestamp: Optional[datetime]) -> None:
        self._last_update_at = _utcnow()
        if source_timestamp is not None:
            self._last_source_timestamp = source_timestamp

    def _event_ready(self, key: str, cooldown_ms: int) -> bool:
        now = _utcnow()
        previous = self._last_event_at.get(key)
        if (
            previous is not None
            and (now - previous).total_seconds() * 1000 < cooldown_ms
        ):
            return False
        self._last_event_at[key] = now
        return True

    def _reset_threshold_state(self) -> None:
        self._triggered_hull_thresholds.clear()
        self._low_fuel_announced = False
        self._critical_fuel_announced = False

    def _update_hull_thresholds(
        self, settings: TelemetrySettings
    ) -> list[TelemetryEvent]:
        hull = self._ship.get("hullPercent")
        if hull is None:
            return []
        events: list[TelemetryEvent] = []
        for threshold in settings.hull_warning_thresholds:
            if hull <= threshold:
                if threshold not in self._triggered_hull_thresholds:
                    self._triggered_hull_thresholds.add(threshold)
                    self._combat["lastHullThreshold"] = threshold
                    events.append(
                        TelemetryEvent(
                            event_type="elite.hull.threshold",
                            summary=f"Hull dropped below {threshold} percent.",
                            novelty_key=f"elite.hull.threshold:{threshold}",
                            importance=9 if threshold <= 25 else 7,
                            urgency=10
                            if threshold <= 10
                            else (9 if threshold <= 25 else 7),
                            confidence=0.94,
                            cooldown_ms=5_000,
                            high_priority=threshold <= 25,
                        )
                    )
            else:
                self._triggered_hull_thresholds.discard(threshold)
        return events

    def _update_fuel_thresholds(
        self, settings: TelemetrySettings
    ) -> list[TelemetryEvent]:
        fuel_level = self._ship.get("fuelLevel")
        fuel_capacity = self._ship.get("fuelCapacity")
        if fuel_level is None or not fuel_capacity:
            return []
        ratio = float(fuel_level) / float(fuel_capacity) if fuel_capacity else None
        if ratio is None:
            return []

        events: list[TelemetryEvent] = []
        low_fuel_active = ratio <= settings.low_fuel_threshold or bool(
            self._ship.get("lowFuelWarning")
        )
        critical_active = ratio <= settings.critical_fuel_threshold
        if not low_fuel_active:
            self._low_fuel_announced = False
        if not critical_active:
            self._critical_fuel_announced = False

        if low_fuel_active and not self._low_fuel_announced:
            self._low_fuel_announced = True
            summary = "Fuel is running low."
            if critical_active:
                summary = "Fuel is critically low."
            events.append(
                TelemetryEvent(
                    event_type="elite.fuel.low",
                    summary=summary,
                    novelty_key=f"elite.fuel.low:{'critical' if critical_active else 'low'}",
                    importance=10 if critical_active else 8,
                    urgency=10 if critical_active else 8,
                    confidence=0.93,
                    cooldown_ms=10_000,
                    high_priority=True,
                )
            )
        elif critical_active and not self._critical_fuel_announced:
            self._critical_fuel_announced = True
            events.append(
                TelemetryEvent(
                    event_type="elite.fuel.low",
                    summary="Fuel is critically low.",
                    novelty_key="elite.fuel.low:critical",
                    importance=10,
                    urgency=10,
                    confidence=0.93,
                    cooldown_ms=10_000,
                    high_priority=True,
                )
            )
        return events


class EliteDangerousTelemetryAdapter:
    adapter_id = "elite-dangerous-journal"
    game_id = "elite-dangerous"
    game_display_name = "Elite Dangerous"
    source_kind = "journal"
    supported_process_names = (
        "elitedangerous64.exe",
        "elitedangerous.exe",
        "elite dangerous",
    )

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
        supported = {item.lower() for item in cls.supported_process_names}
        process_names = {
            str(item).strip().lower()
            for item in profile_process_names
            if isinstance(item, str) and item.strip()
        }
        if process_names.intersection(supported):
            return 100
        if detected_process_name and detected_process_name.strip().lower() in supported:
            return 100

        profile_text = " ".join(
            text.lower()
            for text in [profile_id or "", profile_display_name or ""]
            if text
        )
        window_title = (detected_window_title or "").lower()
        if "elite dangerous" in window_title:
            return 96
        if "elite" in profile_text and "dangerous" in profile_text:
            return 80
        return 0

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
        self._state = EliteDangerousStateAccumulator()
        self._journal_tailer = JournalTailer()
        self._status_reader = StatusReader()
        self._task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._last_payload_signature: Optional[str] = None

    async def start(self, settings: TelemetrySettings) -> None:
        self._settings = settings
        self._closed = False
        if self._task and not self._task.done():
            await self._emit_state_if_changed()
            return
        self._snapshot.status = "starting"
        await self._emit_state_if_changed()
        self._task = asyncio.create_task(self._run(), name="elite-dangerous-telemetry")

    async def stop(self) -> None:
        self._closed = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._journal_tailer.reset()
        self._status_reader.reset()
        self._snapshot.status = "stopped"
        self._snapshot.error = None
        self._snapshot.current_journal_file = None
        self._snapshot.journal_directory_status = "unavailable"
        self._snapshot.journal_directory_label = None
        self._snapshot.last_event_at = None
        self._snapshot.snapshot = self._state.snapshot()
        await self._emit_state_if_changed()

    def current_snapshot(self) -> TelemetryAdapterSnapshot:
        self._snapshot.snapshot = self._state.snapshot()
        return self._snapshot

    async def process_journal_entry(self, entry: dict[str, Any]) -> None:
        for event in self._state.apply_journal_entry(entry, self._settings):
            await self._emit_event(event)
        await self._emit_state_if_changed()

    async def process_status_payload(self, payload: dict[str, Any]) -> None:
        for event in self._state.apply_status(payload, self._settings):
            await self._emit_event(event)
        await self._emit_state_if_changed()

    def build_capability_advice(
        self, capabilities: Iterable[dict[str, Any]]
    ) -> dict[str, TelemetryCapabilityAdvice]:
        snapshot = self._state.snapshot()
        ship = snapshot.ship
        navigation = snapshot.navigation
        combat = snapshot.combat

        advice: dict[str, TelemetryCapabilityAdvice] = {}
        for capability in capabilities:
            command_id = str(
                capability.get("command_id") or capability.get("commandId")
            )
            label = str(capability.get("label") or "")
            text = _normalize_command_text(command_id, label)
            entry = TelemetryCapabilityAdvice()

            if "hardpoint" in text or "weapon" in text:
                if "retract" in text or "stow" in text:
                    if ship.get("hardpointsDeployed") is False:
                        entry.available_override = False
                        entry.blocked_reason = "Hardpoints are already retracted."
                    elif combat.get("inDanger"):
                        entry.avoid_reason = (
                            "Retracting hardpoints during danger is discouraged."
                        )
                    elif navigation.get("dockingGranted") or navigation.get(
                        "dockingRequested"
                    ):
                        entry.recommended = True
                else:
                    if ship.get("hardpointsDeployed") is True:
                        entry.available_override = False
                        entry.blocked_reason = "Hardpoints are already deployed."
                    elif combat.get("inDanger"):
                        entry.recommended = True
            elif "landing gear" in text or text.endswith(" gear") or " gear " in text:
                if ship.get("landingGearDeployed") is True and (
                    snapshot.session.get("docked") or snapshot.session.get("landed")
                ):
                    entry.available_override = False
                    entry.blocked_reason = "Landing gear is already deployed."
                elif navigation.get("dockingGranted") or snapshot.session.get("landed"):
                    entry.recommended = True
            elif "request" in text and "docking" in text:
                if snapshot.session.get("docked"):
                    entry.available_override = False
                    entry.blocked_reason = "The ship is already docked."
                elif navigation.get("dockingGranted"):
                    entry.available_override = False
                    entry.blocked_reason = "Docking is already granted."
                elif snapshot.session.get("stationName"):
                    entry.recommended = True

            if (
                entry.available_override is not None
                or entry.recommended
                or entry.avoid_reason
                or entry.blocked_reason
            ):
                advice[command_id] = entry
        return advice

    async def _run(self) -> None:
        try:
            while not self._closed:
                directory, directory_status, label = discover_elite_journal_directory(
                    self._settings
                )
                self._snapshot.status = "running"
                self._snapshot.journal_directory_status = directory_status
                self._snapshot.journal_directory_label = label
                self._snapshot.error = None

                if directory is None or not directory.exists():
                    self._snapshot.current_journal_file = None
                    self._snapshot.snapshot = self._state.snapshot()
                    await self._emit_state_if_changed()
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                journal_result = await self._journal_tailer.poll(directory)
                self._snapshot.current_journal_file = journal_result.current_file
                if journal_result.parse_errors:
                    self._snapshot.error = f"Ignored {journal_result.parse_errors} malformed journal line(s)."
                for entry in journal_result.events:
                    await self.process_journal_entry(entry)

                status_payload = await self._status_reader.poll(directory)
                if status_payload:
                    await self.process_status_payload(status_payload)

                self._snapshot.snapshot = self._state.snapshot()
                await self._emit_state_if_changed()
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(f"Elite Dangerous telemetry adapter failed: {exc}")
            self._snapshot.status = "error"
            self._snapshot.error = str(exc)
            self._snapshot.snapshot = self._state.snapshot()
            await self._emit_state_if_changed()

    async def _emit_event(self, event: TelemetryEvent) -> None:
        self._snapshot.last_event_at = _isoformat(_utcnow())
        await self._on_event(event)

    async def _emit_state_if_changed(self) -> None:
        self._snapshot.snapshot = self._state.snapshot()
        payload = self._snapshot.to_payload()
        signature = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if signature == self._last_payload_signature:
            return
        self._last_payload_signature = signature
        await self._on_state_change()


class EliteDangerousReplayHarness:
    def __init__(self, adapter: EliteDangerousTelemetryAdapter) -> None:
        self._adapter = adapter
        self._paused = asyncio.Event()
        self._paused.set()

    def pause(self) -> None:
        self._paused.clear()

    def resume(self) -> None:
        self._paused.set()

    async def replay(
        self,
        *,
        journal_lines: Iterable[str],
        status_snapshots: Optional[Iterable[dict[str, Any]]] = None,
        speed: float = 1.0,
    ) -> None:
        snapshot = self._adapter.current_snapshot()
        previous_status = snapshot.status
        previous_replay_mode = snapshot.replay_mode
        snapshot.status = "replay"
        snapshot.replay_mode = True
        await self._adapter._emit_state_if_changed()
        previous_timestamp: Optional[datetime] = None
        status_iter = iter(status_snapshots or [])
        try:
            for raw_line in journal_lines:
                await self._paused.wait()
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.debug(
                        "Replay harness ignored malformed journal fixture line."
                    )
                    continue
                current_timestamp = _parse_time(entry.get("timestamp"))
                if previous_timestamp and current_timestamp and speed > 0:
                    delay = (
                        current_timestamp - previous_timestamp
                    ).total_seconds() / speed
                    if delay > 0:
                        await asyncio.sleep(min(delay, 1.0))
                previous_timestamp = current_timestamp or previous_timestamp
                await self._adapter.process_journal_entry(entry)
                try:
                    status_payload = next(status_iter)
                except StopIteration:
                    status_payload = None
                if isinstance(status_payload, dict):
                    await self._adapter.process_status_payload(status_payload)
        finally:
            snapshot = self._adapter.current_snapshot()
            snapshot.status = previous_status
            snapshot.replay_mode = previous_replay_mode
            await self._adapter._emit_state_if_changed()
