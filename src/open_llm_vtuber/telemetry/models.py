from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

TelemetryAdapterStatus = Literal["stopped", "starting", "running", "error", "replay"]
TelemetryDirectoryStatus = Literal[
    "automatic",
    "manual",
    "missing",
    "error",
    "unavailable",
]
TelemetryHighlightLevel = Literal["default", "good", "warning", "danger"]

DEFAULT_HULL_THRESHOLDS = [75, 50, 25, 10]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalize_thresholds(value: Any) -> list[int]:
    if not isinstance(value, list):
        return list(DEFAULT_HULL_THRESHOLDS)
    normalized = sorted(
        {
            max(1, min(100, int(round(float(item)))))
            for item in value
            if isinstance(item, (int, float))
        },
        reverse=True,
    )
    return normalized or list(DEFAULT_HULL_THRESHOLDS)


@dataclass(slots=True)
class TelemetryCommentarySettings:
    announce_docking_events: bool = True
    announce_jump_events: bool = True
    announce_mission_events: bool = True
    announce_discoveries: bool = True
    announce_material_collection: bool = False

    @classmethod
    def from_raw(cls, raw: Any) -> "TelemetryCommentarySettings":
        if not isinstance(raw, dict):
            return cls()
        defaults = cls()
        return cls(
            announce_docking_events=bool(
                raw.get("announceDockingEvents", defaults.announce_docking_events)
            ),
            announce_jump_events=bool(
                raw.get("announceJumpEvents", defaults.announce_jump_events)
            ),
            announce_mission_events=bool(
                raw.get("announceMissionEvents", defaults.announce_mission_events)
            ),
            announce_discoveries=bool(
                raw.get("announceDiscoveries", defaults.announce_discoveries)
            ),
            announce_material_collection=bool(
                raw.get(
                    "announceMaterialCollection",
                    defaults.announce_material_collection,
                )
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "announceDockingEvents": self.announce_docking_events,
            "announceJumpEvents": self.announce_jump_events,
            "announceMissionEvents": self.announce_mission_events,
            "announceDiscoveries": self.announce_discoveries,
            "announceMaterialCollection": self.announce_material_collection,
        }


@dataclass(slots=True)
class TelemetrySettings:
    enabled: bool = True
    automatic_journal_discovery: bool = True
    manual_journal_directory: Optional[str] = None
    low_fuel_threshold: float = 0.25
    critical_fuel_threshold: float = 0.10
    hull_warning_thresholds: list[int] = field(
        default_factory=lambda: list(DEFAULT_HULL_THRESHOLDS)
    )
    exploration_commentary_cooldown_ms: int = 90_000
    commentary: TelemetryCommentarySettings = field(
        default_factory=TelemetryCommentarySettings
    )

    @classmethod
    def from_raw(cls, raw: Any) -> "TelemetrySettings":
        if not isinstance(raw, dict):
            return cls()
        defaults = cls()
        low_fuel_threshold = (
            float(raw.get("lowFuelThreshold"))
            if isinstance(raw.get("lowFuelThreshold"), (float, int))
            else defaults.low_fuel_threshold
        )
        low_fuel_threshold = _clamp(low_fuel_threshold, 0.01, 0.95)
        critical_fuel_threshold = (
            float(raw.get("criticalFuelThreshold"))
            if isinstance(raw.get("criticalFuelThreshold"), (float, int))
            else defaults.critical_fuel_threshold
        )
        critical_fuel_threshold = _clamp(
            critical_fuel_threshold, 0.01, low_fuel_threshold
        )
        cooldown_ms = (
            int(raw.get("explorationCommentaryCooldownMs"))
            if isinstance(raw.get("explorationCommentaryCooldownMs"), (float, int))
            else defaults.exploration_commentary_cooldown_ms
        )
        return cls(
            enabled=bool(raw.get("enabled", defaults.enabled)),
            automatic_journal_discovery=bool(
                raw.get(
                    "automaticJournalDiscovery",
                    defaults.automatic_journal_discovery,
                )
            ),
            manual_journal_directory=(
                str(raw.get("manualJournalDirectory")).strip() or None
                if isinstance(raw.get("manualJournalDirectory"), str)
                else None
            ),
            low_fuel_threshold=low_fuel_threshold,
            critical_fuel_threshold=critical_fuel_threshold,
            hull_warning_thresholds=_normalize_thresholds(
                raw.get("hullWarningThresholds")
            ),
            exploration_commentary_cooldown_ms=max(5_000, min(600_000, cooldown_ms)),
            commentary=TelemetryCommentarySettings.from_raw(raw.get("commentary")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "automaticJournalDiscovery": self.automatic_journal_discovery,
            "manualJournalDirectory": self.manual_journal_directory,
            "lowFuelThreshold": self.low_fuel_threshold,
            "criticalFuelThreshold": self.critical_fuel_threshold,
            "hullWarningThresholds": list(self.hull_warning_thresholds),
            "explorationCommentaryCooldownMs": (
                self.exploration_commentary_cooldown_ms
            ),
            "commentary": self.commentary.to_payload(),
        }


@dataclass(slots=True)
class TelemetryEvent:
    event_type: str
    summary: str
    novelty_key: str
    importance: int
    urgency: int
    confidence: float
    expiry_ms: int = 60_000
    cooldown_ms: int = 15_000
    conversational: bool = False
    high_priority: bool = False
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TelemetryCapabilityAdvice:
    available_override: Optional[bool] = None
    recommended: bool = False
    avoid_reason: Optional[str] = None
    blocked_reason: Optional[str] = None


@dataclass(slots=True)
class TelemetryHighlight:
    key: str
    label: str
    value: str
    level: TelemetryHighlightLevel = "default"

    def to_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "level": self.level,
        }


@dataclass(slots=True)
class StructuredTelemetrySnapshot:
    source_timestamp: Optional[str] = None
    last_update_timestamp: Optional[str] = None
    confidence: float = 0.0
    stale: bool = True
    session: dict[str, Any] = field(default_factory=dict)
    ship: dict[str, Any] = field(default_factory=dict)
    navigation: dict[str, Any] = field(default_factory=dict)
    combat: dict[str, Any] = field(default_factory=dict)
    missions: dict[str, Any] = field(default_factory=dict)
    world: dict[str, Any] = field(default_factory=dict)
    automation: dict[str, Any] = field(default_factory=dict)
    highlights: list[TelemetryHighlight] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "sourceTimestamp": self.source_timestamp,
            "lastUpdateTimestamp": self.last_update_timestamp,
            "confidence": self.confidence,
            "stale": self.stale,
            "session": dict(self.session),
            "ship": dict(self.ship),
            "navigation": dict(self.navigation),
            "combat": dict(self.combat),
            "missions": dict(self.missions),
            "world": dict(self.world),
            "automation": dict(self.automation),
            "highlights": [highlight.to_payload() for highlight in self.highlights],
        }


@dataclass(slots=True)
class TelemetryAdapterSnapshot:
    adapter_id: Optional[str] = None
    game_id: Optional[str] = None
    game_display_name: Optional[str] = None
    status: TelemetryAdapterStatus = "stopped"
    replay_mode: bool = False
    error: Optional[str] = None
    journal_directory_status: TelemetryDirectoryStatus = "unavailable"
    journal_directory_label: Optional[str] = None
    current_journal_file: Optional[str] = None
    source_kind: Optional[str] = None
    source_directory_status: TelemetryDirectoryStatus = "unavailable"
    source_directory_label: Optional[str] = None
    current_source: Optional[str] = None
    last_event_at: Optional[str] = None
    snapshot: Optional[StructuredTelemetrySnapshot] = None

    def to_payload(self) -> dict[str, Any]:
        resolved_directory_status = (
            self.source_directory_status
            if self.source_directory_status != "unavailable"
            or self.journal_directory_status == "unavailable"
            else self.journal_directory_status
        )
        resolved_directory_label = (
            self.source_directory_label
            if self.source_directory_label is not None
            else self.journal_directory_label
        )
        resolved_current_source = (
            self.current_source
            if self.current_source is not None
            else self.current_journal_file
        )
        return {
            "adapterId": self.adapter_id,
            "gameId": self.game_id,
            "gameDisplayName": self.game_display_name,
            "status": self.status,
            "replayMode": self.replay_mode,
            "error": self.error,
            "journalDirectoryStatus": resolved_directory_status,
            "journalDirectoryLabel": resolved_directory_label,
            "currentJournalFile": resolved_current_source,
            "sourceKind": self.source_kind,
            "sourceDirectoryStatus": resolved_directory_status,
            "sourceDirectoryLabel": resolved_directory_label,
            "currentSource": resolved_current_source,
            "lastEventAt": self.last_event_at,
            "snapshot": self.snapshot.to_payload() if self.snapshot else None,
        }


def redact_directory_label(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        home = Path.home().resolve()
        resolved = path.resolve()
        relative = resolved.relative_to(home)
        return str(Path("~") / relative)
    except Exception:
        return path.name or str(path)
