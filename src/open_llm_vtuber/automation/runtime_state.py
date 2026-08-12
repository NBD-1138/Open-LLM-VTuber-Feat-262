from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from .transport import (
    AssistantActiveApplicationMessage,
    AssistantContextSyncMessage,
    AssistantPlayerStateMessage,
)

AssistantEventType = str

AssistantActivityStatus = Literal[
    "noticed",
    "queued",
    "suppressed",
    "sent_to_llm",
    "acted",
    "remained_silent",
]

AssistantSpeechMode = Literal["action", "exploration", "conversation"]
AssistantProfileSelectionMode = Literal["automatic", "manual", "disabled"]
AssistantProfileSelectionSource = Literal["automatic", "manual", "none"]

QUEUE_LIMIT = 24
ACTIVITY_LIMIT = 40
RECENT_OUTCOME_LIMIT = 8
RECENT_TWITCH_EVENT_LIMIT = 8

DEFAULT_EVENT_IMPORTANCE = 3
DEFAULT_EVENT_URGENCY = 3
DEFAULT_EVENT_EXPIRY_MS = 30_000
DEFAULT_EVENT_COOLDOWN_MS = 15_000

EVENT_DEFAULTS: dict[str, dict[str, Any]] = {
    "player.direct_request": {"importance": 10, "urgency": 10, "expiry_ms": 60_000},
    "application.changed": {"importance": 4, "urgency": 3, "expiry_ms": 45_000},
    "profile.changed": {"importance": 5, "urgency": 4, "expiry_ms": 45_000},
    "automation.requested": {"importance": 7, "urgency": 7, "expiry_ms": 60_000},
    "automation.confirmation_required": {
        "importance": 8,
        "urgency": 8,
        "expiry_ms": 60_000,
    },
    "automation.completed": {"importance": 2, "urgency": 2, "expiry_ms": 30_000},
    "automation.failed": {"importance": 8, "urgency": 8, "expiry_ms": 90_000},
    "automation.blocked": {"importance": 8, "urgency": 8, "expiry_ms": 90_000},
    "automation.cancelled": {"importance": 4, "urgency": 4, "expiry_ms": 45_000},
    "emergency_stop.activated": {"importance": 10, "urgency": 10, "expiry_ms": 120_000},
    "emergency_stop.reset": {"importance": 5, "urgency": 4, "expiry_ms": 60_000},
    "twitch.chat": {"importance": 3, "urgency": 2, "expiry_ms": 30_000},
    "twitch.subscription": {"importance": 4, "urgency": 3, "expiry_ms": 60_000},
    "twitch.redemption": {"importance": 5, "urgency": 4, "expiry_ms": 60_000},
    "system.warning": {"importance": 9, "urgency": 8, "expiry_ms": 120_000},
}

EVENT_COOLDOWNS_MS: dict[str, int] = {
    "player.direct_request": 5_000,
    "application.changed": 20_000,
    "profile.changed": 20_000,
    "automation.requested": 15_000,
    "automation.confirmation_required": 15_000,
    "automation.completed": 10_000,
    "automation.failed": 20_000,
    "automation.blocked": 20_000,
    "automation.cancelled": 15_000,
    "emergency_stop.activated": 10_000,
    "emergency_stop.reset": 10_000,
    "twitch.chat": 15_000,
    "twitch.subscription": 20_000,
    "twitch.redemption": 20_000,
    "system.warning": 20_000,
}

CONVERSATIONAL_EVENT_TYPES: set[str] = {
    "application.changed",
    "profile.changed",
    "twitch.chat",
    "twitch.subscription",
    "twitch.redemption",
}

HIGH_PRIORITY_EVENT_TYPES: set[str] = {
    "player.direct_request",
    "automation.confirmation_required",
    "automation.failed",
    "automation.blocked",
    "emergency_stop.activated",
    "system.warning",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.isoformat()


def _clip_text(value: str, limit: int = 200) -> str:
    clipped = " ".join(str(value or "").split()).strip()
    if len(clipped) <= limit:
        return clipped
    return f"{clipped[: limit - 3].rstrip()}..."


@dataclass
class AssistantActiveApplicationState:
    process_name: str
    window_title: str
    detected_at: datetime
    confidence: float


@dataclass
class AssistantEvent:
    event_id: str
    event_type: AssistantEventType
    source: str
    timestamp: datetime
    importance: int
    urgency: int
    confidence: float
    novelty_key: str
    expiry: datetime
    summary: str
    cooldown_ms: int = DEFAULT_EVENT_COOLDOWN_MS
    conversational: bool = False
    high_priority: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    score: int = 0

    def __post_init__(self) -> None:
        if not self.summary:
            raise ValueError("Assistant event summary cannot be empty")
        if self.importance < 0 or self.urgency < 0:
            raise ValueError("importance and urgency must be non-negative")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.cooldown_ms < 0:
            raise ValueError("cooldown_ms must be non-negative")


@dataclass
class AssistantActivityEntry:
    activity_id: str
    event_id: Optional[str]
    event_type: str
    status: AssistantActivityStatus
    summary: str
    policy_reason: Optional[str]
    created_at: datetime


@dataclass
class AssistantRuntimeState:
    version: int = 0
    frontend_version: int = -1
    active_application: Optional[AssistantActiveApplicationState] = None
    matched_profile_id: Optional[str] = None
    effective_profile_id: Optional[str] = None
    manual_profile_id: Optional[str] = None
    profile_selection_mode: AssistantProfileSelectionMode = "manual"
    profile_selection_source: AssistantProfileSelectionSource = "none"
    profile_match_confidence: Optional[float] = None
    last_profile_transition_at: Optional[datetime] = None
    capability_revision: int = -1
    emergency_stopped: bool = False
    current_executions: set[str] = field(default_factory=set)
    pending_confirmation_count: int = 0
    vtuber_speaking: bool = False
    player_speaking: Optional[bool] = None
    speech_mode: AssistantSpeechMode = "action"
    proactive_commentary_enabled: bool = False
    minimum_commentary_interval_ms: int = 15_000
    max_queued_conversation_events: int = 5
    announce_application_changes: bool = False
    acknowledge_routine_automation_success: bool = False
    last_player_request: Optional[str] = None
    last_player_request_at: Optional[datetime] = None
    last_commentary_at: Optional[datetime] = None
    recent_automation_outcomes: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=RECENT_OUTCOME_LIMIT)
    )
    recent_twitch_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=RECENT_TWITCH_EVENT_LIMIT)
    )
    telemetry: Optional[dict[str, Any]] = None


class AssistantRuntimeController:
    def __init__(self) -> None:
        self.state = AssistantRuntimeState()
        self._queue: list[AssistantEvent] = []
        self._activities: deque[AssistantActivityEntry] = deque(maxlen=ACTIVITY_LIMIT)
        self._recent_novelty_seen: dict[str, datetime] = {}
        self._suppressed_event_count = 0

    def apply_active_application(
        self, message: AssistantActiveApplicationMessage
    ) -> bool:
        detected_at = self._parse_time(message.detected_at)
        next_state = AssistantActiveApplicationState(
            process_name=message.process_name,
            window_title=message.window_title,
            detected_at=detected_at,
            confidence=message.confidence,
        )
        current = self.state.active_application
        signature = self._application_signature(next_state)
        if current and self._application_signature(current) == signature:
            return False

        self.state.active_application = next_state
        self.record_event(
            event_type="application.changed",
            source="frontend",
            summary=f"Active application changed to {next_state.window_title}.",
            novelty_key=f"app:{signature}",
            data={
                "process_name": next_state.process_name,
                "window_title": next_state.window_title,
            },
        )
        return True

    def apply_context_sync(self, message: AssistantContextSyncMessage) -> bool:
        if message.version <= self.state.frontend_version:
            return False

        previous_effective_profile = self.state.effective_profile_id
        previous_selection_source = self.state.profile_selection_source
        self.state.frontend_version = message.version
        self.state.matched_profile_id = message.matched_profile_id
        self.state.effective_profile_id = message.effective_profile_id
        self.state.manual_profile_id = message.manual_profile_id
        self.state.profile_selection_mode = message.profile_selection_mode
        self.state.profile_selection_source = message.profile_selection_source
        self.state.profile_match_confidence = message.profile_match_confidence
        self.state.last_profile_transition_at = (
            self._parse_time(message.last_profile_transition_at)
            if message.last_profile_transition_at
            else self.state.last_profile_transition_at
        )
        self.state.speech_mode = message.speech_mode
        self.state.proactive_commentary_enabled = message.proactive_commentary_enabled
        self.state.minimum_commentary_interval_ms = (
            message.minimum_commentary_interval_ms
        )
        self.state.max_queued_conversation_events = (
            message.max_queued_conversation_events
        )
        self.state.announce_application_changes = message.announce_application_changes
        self.state.acknowledge_routine_automation_success = (
            message.acknowledge_routine_automation_success
        )
        self.state.player_speaking = message.player_speaking

        changed = self._bump_version()
        if (
            previous_effective_profile != self.state.effective_profile_id
            or previous_selection_source != self.state.profile_selection_source
        ):
            summary = (
                f"Active profile changed to {self.state.effective_profile_id}."
                if self.state.effective_profile_id
                else "No active automation profile is currently selected."
            )
            self.record_event(
                event_type="profile.changed",
                source="frontend",
                summary=summary,
                novelty_key=(
                    f"profile:{self.state.effective_profile_id or 'none'}:"
                    f"{self.state.profile_selection_source}"
                ),
                data={
                    "effective_profile_id": self.state.effective_profile_id,
                    "selection_source": self.state.profile_selection_source,
                },
            )
            changed = True

        return changed

    def apply_player_state(self, message: AssistantPlayerStateMessage) -> bool:
        self.state.player_speaking = message.speaking
        return self._bump_version()

    def set_capability_revision(self, revision: int) -> bool:
        if revision == self.state.capability_revision:
            return False
        self.state.capability_revision = revision
        return self._bump_version()

    def set_pending_confirmation_count(self, count: int) -> bool:
        count = max(0, count)
        if count == self.state.pending_confirmation_count:
            return False
        self.state.pending_confirmation_count = count
        return self._bump_version()

    def set_vtuber_speaking(self, speaking: bool) -> bool:
        if speaking == self.state.vtuber_speaking:
            return False
        self.state.vtuber_speaking = speaking
        return self._bump_version()

    def set_emergency_stopped(self, active: bool) -> bool:
        if active == self.state.emergency_stopped:
            return False
        self.state.emergency_stopped = active
        self.record_event(
            event_type=(
                "emergency_stop.activated" if active else "emergency_stop.reset"
            ),
            source="automation",
            summary=(
                "Emergency stop is active."
                if active
                else "Emergency stop has been reset."
            ),
            novelty_key=f"emergency:{'on' if active else 'off'}",
        )
        return True

    def set_current_executions(self, request_ids: list[str]) -> bool:
        next_request_ids = set(request_ids)
        if next_request_ids == self.state.current_executions:
            return False
        self.state.current_executions = next_request_ids
        return self._bump_version()

    def record_direct_request(self, text: str, *, source: str = "player") -> bool:
        summary = _clip_text(text, limit=180)
        self.state.last_player_request = summary
        self.state.last_player_request_at = _utcnow()
        return self.record_event(
            event_type="player.direct_request",
            source=source,
            summary=f"Player directly asked: {summary}",
            novelty_key=f"player:{summary.lower()}",
            data={"text": summary},
        )

    def record_automation_result(
        self,
        *,
        status: Literal["completed", "failed", "blocked", "cancelled"],
        command_label: str,
        request_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> bool:
        outcome = {
            "status": status,
            "command_label": command_label,
            "request_id": request_id,
            "error": error,
            "timestamp": _isoformat(_utcnow()),
        }
        self.state.recent_automation_outcomes.appendleft(outcome)
        event_type: AssistantEventType
        summary: str
        if status == "completed":
            event_type = "automation.completed"
            summary = f"{command_label} completed successfully."
        elif status == "failed":
            event_type = "automation.failed"
            summary = f"{command_label} failed."
        elif status == "blocked":
            event_type = "automation.blocked"
            summary = f"{command_label} was blocked."
        else:
            event_type = "automation.cancelled"
            summary = f"{command_label} was cancelled."
        return self.record_event(
            event_type=event_type,
            source="automation",
            summary=summary,
            novelty_key=f"automation:{request_id or command_label}:{status}",
            data={"request_id": request_id, "error": error, "command_label": command_label},
        )

    def record_automation_request(
        self,
        *,
        command_label: str,
        reason: str,
        request_id: Optional[str] = None,
        confirmation_required: bool = False,
    ) -> bool:
        return self.record_event(
            event_type=(
                "automation.confirmation_required"
                if confirmation_required
                else "automation.requested"
            ),
            source="automation",
            summary=(
                f"{command_label} needs confirmation."
                if confirmation_required
                else f"{command_label} was requested."
            ),
            novelty_key=(
                f"automation-request:{request_id or command_label}:"
                f"{'confirm' if confirmation_required else 'start'}"
            ),
            data={"command_label": command_label, "reason": reason, "request_id": request_id},
        )

    def record_twitch_event(
        self,
        *,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        metadata = metadata or {}
        event_name = str(metadata.get("event") or metadata.get("category") or "").lower()
        if "redeem" in event_name or "reward" in event_name:
            event_type: AssistantEventType = "twitch.redemption"
        elif "sub" in event_name:
            event_type = "twitch.subscription"
        else:
            event_type = "twitch.chat"
        summary = _clip_text(text, limit=180)
        self.state.recent_twitch_events.appendleft(
            {
                "type": event_type,
                "summary": summary,
                "timestamp": _isoformat(_utcnow()),
            }
        )
        novelty_key = (
            f"{event_type}:{metadata.get('message_id') or metadata.get('event') or summary.lower()}"
        )
        return self.record_event(
            event_type=event_type,
            source="twitch",
            summary=summary,
            novelty_key=novelty_key,
            data=metadata,
        )

    def record_system_warning(self, text: str, *, source: str = "system") -> bool:
        return self.record_event(
            event_type="system.warning",
            source=source,
            summary=_clip_text(text, limit=180),
            novelty_key=f"warning:{_clip_text(text, limit=120).lower()}",
        )

    def record_event(
        self,
        *,
        event_type: AssistantEventType,
        source: str,
        summary: str,
        novelty_key: Optional[str] = None,
        confidence: float = 1.0,
        data: Optional[dict[str, Any]] = None,
        importance: Optional[int] = None,
        urgency: Optional[int] = None,
        expiry_ms: Optional[int] = None,
        cooldown_ms: Optional[int] = None,
        conversational: Optional[bool] = None,
        high_priority: Optional[bool] = None,
    ) -> bool:
        defaults = EVENT_DEFAULTS.get(event_type, {})
        now = _utcnow()
        self._expire(now)
        event_importance = (
            importance
            if importance is not None
            else int(defaults.get("importance", DEFAULT_EVENT_IMPORTANCE))
        )
        event_urgency = (
            urgency
            if urgency is not None
            else int(defaults.get("urgency", DEFAULT_EVENT_URGENCY))
        )
        event_expiry_ms = (
            expiry_ms
            if expiry_ms is not None
            else int(defaults.get("expiry_ms", DEFAULT_EVENT_EXPIRY_MS))
        )
        event_cooldown_ms = (
            cooldown_ms
            if cooldown_ms is not None
            else EVENT_COOLDOWNS_MS.get(event_type, DEFAULT_EVENT_COOLDOWN_MS)
        )
        is_conversational = (
            conversational
            if conversational is not None
            else event_type in CONVERSATIONAL_EVENT_TYPES
        )
        is_high_priority = (
            high_priority
            if high_priority is not None
            else event_type in HIGH_PRIORITY_EVENT_TYPES
        )
        event = AssistantEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            source=source,
            timestamp=now,
            importance=event_importance,
            urgency=event_urgency,
            confidence=confidence,
            novelty_key=novelty_key or f"{event_type}:{summary.lower()}",
            expiry=now + timedelta(milliseconds=event_expiry_ms),
            summary=_clip_text(summary, limit=180),
            cooldown_ms=event_cooldown_ms,
            conversational=is_conversational,
            high_priority=is_high_priority,
            data=data or {},
            score=0,
        )
        event.score = self._score_event(event)
        self._record_activity(
            event=event,
            status="noticed",
            policy_reason=None,
        )

        suppress_reason = self._get_suppression_reason(event, now)
        self._recent_novelty_seen[event.novelty_key] = now
        if suppress_reason:
            self._suppressed_event_count += 1
            self._record_activity(
                event=event,
                status="suppressed",
                policy_reason=suppress_reason,
            )
            return self._bump_version()

        self._queue.append(event)
        self._queue.sort(key=lambda item: (-item.score, item.timestamp))
        if len(self._queue) > QUEUE_LIMIT:
            dropped = self._queue.pop()
            self._suppressed_event_count += 1
            self._record_activity(
                event=dropped,
                status="suppressed",
                policy_reason="The attention queue limit was reached.",
            )
        self._record_activity(
            event=event,
            status="queued",
            policy_reason=None,
        )
        return self._bump_version()

    def set_telemetry_snapshot(self, telemetry: Optional[dict[str, Any]]) -> bool:
        if self.state.telemetry == telemetry:
            return False
        self.state.telemetry = telemetry
        return self._bump_version()

    def mark_acted(self, *, summary: str, event_type: str = "automation.requested") -> bool:
        self._record_activity(
            event=None,
            status="acted",
            policy_reason=None,
            summary=_clip_text(summary, limit=180),
            event_type=event_type,
        )
        return self._bump_version()

    def mark_remained_silent(self, reason: str) -> bool:
        self._record_activity(
            event=None,
            status="remained_silent",
            policy_reason=reason,
            summary="The assistant decided to remain silent.",
            event_type="remain_silent",
        )
        return self._bump_version()

    def build_context_prompt(
        self,
        *,
        capabilities: list[dict[str, Any]],
        pending_confirmations: list[dict[str, Any]],
    ) -> str:
        now = _utcnow()
        self._expire(now)
        selected_events = self._consume_events_for_llm(now)
        capability_labels = [
            str(capability.get("label") or capability.get("command_label") or "").strip()
            for capability in capabilities
            if capability.get("llm_access") != "none"
        ]
        capability_labels = [label for label in capability_labels if label][:8]

        active_application = (
            self.state.active_application.window_title
            if self.state.active_application
            else "Unknown"
        )
        active_profile = self.state.effective_profile_id or "None"
        selection = self.state.profile_selection_source
        if selection == "none":
            selection = self.state.profile_selection_mode
        pending_confirmation_text = (
            "none"
            if not pending_confirmations
            else f"{len(pending_confirmations)} pending"
        )

        lines = [
            "ACTIVE CONTEXT",
            f"- Active application: {active_application}",
            f"- Active profile: {active_profile}",
            f"- Profile selection: {selection}",
            f"- Speech mode: {self.state.speech_mode}",
            f"- Emergency stop: {'active' if self.state.emergency_stopped else 'inactive'}",
            f"- Pending confirmation: {pending_confirmation_text}",
            "",
            "IMPORTANT RECENT EVENTS",
        ]
        if selected_events:
            for event in selected_events[:4]:
                lines.append(f"- {event.summary}")
        else:
            lines.append("- No important recent events.")

        lines.extend(["", "AVAILABLE CAPABILITIES"])
        if capability_labels:
            for label in capability_labels:
                lines.append(f"- {label}")
        else:
            lines.append("- No callable automation capabilities are currently available.")

        telemetry = self.state.telemetry or {}
        telemetry_snapshot = telemetry.get("snapshot") if isinstance(telemetry, dict) else None
        if telemetry_snapshot:
            session = telemetry_snapshot.get("session") or {}
            ship = telemetry_snapshot.get("ship") or {}
            combat = telemetry_snapshot.get("combat") or {}
            highlights = telemetry_snapshot.get("highlights") or []
            lines.extend(["", "GAME TELEMETRY"])
            lines.append(
                f"- Game: {telemetry.get('game_display_name') or telemetry.get('gameDisplayName') or telemetry.get('game_id') or telemetry.get('gameId') or 'Unknown'}"
            )
            lines.append(
                f"- Adapter: {telemetry.get('adapter_id') or telemetry.get('adapterId') or 'Unknown'}"
            )
            if highlights:
                for highlight in highlights[:8]:
                    if not isinstance(highlight, dict):
                        continue
                    label = str(highlight.get("label") or highlight.get("key") or "").strip()
                    value = str(highlight.get("value") or "").strip()
                    if label and value:
                        lines.append(f"- {label}: {value}")
            else:
                star_system = (
                    session.get("star_system") or session.get("starSystem") or "Unknown"
                )
                ship_name = session.get("ship_name") or session.get("shipName") or "Unknown"
                lines.append(f"- Star system: {star_system}")
                lines.append(f"- Ship: {ship_name}")
                if session.get("docked") is not None:
                    lines.append(f"- Docked: {'yes' if session.get('docked') else 'no'}")
                if combat.get("in_danger") is not None or combat.get("inDanger") is not None:
                    in_danger = combat.get("in_danger")
                    if in_danger is None:
                        in_danger = combat.get("inDanger")
                    lines.append(f"- In danger: {'yes' if in_danger else 'no'}")
                if ship.get("hull_percent") is not None or ship.get("hullPercent") is not None:
                    hull_percent = ship.get("hull_percent")
                    if hull_percent is None:
                        hull_percent = ship.get("hullPercent")
                    lines.append(f"- Hull: {hull_percent}%")
                if ship.get("fuel_level") is not None or ship.get("fuelLevel") is not None:
                    fuel_level = ship.get("fuel_level")
                    if fuel_level is None:
                        fuel_level = ship.get("fuelLevel")
                    lines.append(f"- Fuel: {fuel_level}")
            if telemetry.get("error"):
                lines.append(f"- Telemetry error: {telemetry.get('error')}")
            if telemetry_snapshot.get("stale"):
                lines.append("- Telemetry: stale")

        speech_rule = {
            "action": "Be brief during action.",
            "exploration": "Use moderate detail during exploration.",
            "conversation": "Use natural conversational speech when useful.",
        }[self.state.speech_mode]

        lines.extend(
            [
                "",
                "BEHAVIOR",
                f"- {speech_rule}",
                "- Prefer silence over unnecessary narration.",
                "- Do not claim to observe game details that are not in the active context.",
                "- Do not invent telemetry.",
                "- Do not repeatedly announce application changes.",
                "- Never claim a command completed until the executor reports success.",
            ]
        )
        return "\n".join(lines)

    def build_payload(self) -> dict[str, Any]:
        self._expire(_utcnow())
        current_event = self._queue[0] if self._queue else None
        return {
            "state_version": self.state.version,
            "detected_application": self._serialize_active_application(
                self.state.active_application
            ),
            "matched_profile_id": self.state.matched_profile_id,
            "effective_profile_id": self.state.effective_profile_id,
            "profile_selection_mode": self.state.profile_selection_mode,
            "profile_selection_source": self.state.profile_selection_source,
            "profile_match_confidence": self.state.profile_match_confidence,
            "last_profile_transition_at": (
                _isoformat(self.state.last_profile_transition_at)
                if self.state.last_profile_transition_at
                else None
            ),
            "speech_mode": self.state.speech_mode,
            "telemetry": self.state.telemetry,
            "queue_summary": {
                "total": len(self._queue),
                "high_priority": sum(
                    1 for event in self._queue if event.high_priority
                ),
                "conversational": sum(
                    1 for event in self._queue if event.conversational
                ),
                "suppressed": self._suppressed_event_count,
            },
            "current_event": self._serialize_event(current_event) if current_event else None,
            "suppressed_event_count": self._suppressed_event_count,
            "activity": [self._serialize_activity(entry) for entry in self._activities],
            "vtuber_speaking": self.state.vtuber_speaking,
            "player_speaking": self.state.player_speaking,
            "last_player_request": self.state.last_player_request,
        }

    def should_suppress_twitch_forward(self) -> Optional[str]:
        now = _utcnow()
        if self.state.speech_mode != "conversation":
            return "Twitch events are limited to conversation mode."
        if not self.state.proactive_commentary_enabled:
            return "Proactive commentary is disabled."
        if (
            self.state.last_player_request_at
            and (now - self.state.last_player_request_at).total_seconds() * 1000 < 20_000
        ):
            return "Direct player requests take priority over Twitch events."
        if self.state.vtuber_speaking:
            return "The VTuber is already speaking."
        if (
            self.state.last_commentary_at
            and (now - self.state.last_commentary_at).total_seconds() * 1000
            < self.state.minimum_commentary_interval_ms
        ):
            return "The commentary cooldown is active."
        return None

    def _consume_events_for_llm(self, now: datetime) -> list[AssistantEvent]:
        if not self._queue:
            return []
        selected = self._queue[:4]
        self._queue = self._queue[4:]
        self.state.last_commentary_at = now
        for event in selected:
            self._record_activity(event=event, status="sent_to_llm", policy_reason=None)
        self._bump_version()
        return selected

    def _get_suppression_reason(
        self,
        event: AssistantEvent,
        now: datetime,
    ) -> Optional[str]:
        previous_seen = self._recent_novelty_seen.get(event.novelty_key)
        cooldown_ms = event.cooldown_ms
        if previous_seen and (now - previous_seen).total_seconds() * 1000 < cooldown_ms:
            return "Repeated event suppressed."

        if event.event_type == "automation.completed" and not self.state.acknowledge_routine_automation_success:
            return "Routine automation success is silent."
        if event.event_type in {"application.changed", "profile.changed"} and not self.state.announce_application_changes:
            return "Application and profile announcements are disabled."
        if event.event_type.startswith("twitch.") and self.state.speech_mode != "conversation":
            return "Twitch events are limited to conversation mode."
        if (
            event.event_type == "twitch.chat"
            and not self.state.proactive_commentary_enabled
        ):
            return "Proactive commentary is disabled."
        if (
            event.event_type.startswith("twitch.")
            and self.state.last_player_request_at
            and (now - self.state.last_player_request_at).total_seconds() * 1000 < 20_000
        ):
            return "Direct player requests take priority over Twitch events."
        if (
            self.state.last_commentary_at
            and not event.high_priority
            and (now - self.state.last_commentary_at).total_seconds() * 1000
            < self.state.minimum_commentary_interval_ms
        ):
            return "The commentary cooldown is active."
        if (
            event.conversational
            and self._conversation_queue_count() >= self.state.max_queued_conversation_events
        ):
            return "The conversational queue limit has been reached."
        if (
            self.state.vtuber_speaking
            and not event.high_priority
        ):
            return "The VTuber is already speaking."
        return None

    def _score_event(self, event: AssistantEvent) -> int:
        source_bonus = {
            "player": 30,
            "player_voice": 30,
            "automation": 20,
            "system": 20,
            "frontend": 10,
            "twitch": 5,
        }.get(event.source, 0)
        speech_mode_penalty = {
            "action": 12 if event.conversational else 0,
            "exploration": 6 if event.conversational else 0,
            "conversation": 0,
        }[self.state.speech_mode]
        confidence_score = round(event.confidence * 10)
        return (
            event.importance * 10
            + event.urgency * 6
            + confidence_score
            + source_bonus
            - speech_mode_penalty
        )

    def _conversation_queue_count(self) -> int:
        return sum(1 for event in self._queue if event.conversational)

    def _record_activity(
        self,
        *,
        event: Optional[AssistantEvent],
        status: AssistantActivityStatus,
        policy_reason: Optional[str],
        summary: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> None:
        entry = AssistantActivityEntry(
            activity_id=str(uuid4()),
            event_id=event.event_id if event else None,
            event_type=event_type or (event.event_type if event else "system"),
            status=status,
            summary=summary or (event.summary if event else "Assistant activity"),
            policy_reason=policy_reason,
            created_at=_utcnow(),
        )
        self._activities.appendleft(entry)

    def _expire(self, now: datetime) -> None:
        self._queue = [event for event in self._queue if event.expiry > now]
        cutoff = now - timedelta(minutes=5)
        self._recent_novelty_seen = {
            key: value
            for key, value in self._recent_novelty_seen.items()
            if value >= cutoff
        }

    def _serialize_active_application(
        self, value: Optional[AssistantActiveApplicationState]
    ) -> Optional[dict[str, Any]]:
        if value is None:
            return None
        return {
            "process_name": value.process_name,
            "window_title": value.window_title,
            "detected_at": _isoformat(value.detected_at),
            "confidence": value.confidence,
        }

    def _serialize_event(self, value: AssistantEvent) -> dict[str, Any]:
        return {
            "event_id": value.event_id,
            "event_type": value.event_type,
            "summary": value.summary,
            "importance": value.importance,
            "urgency": value.urgency,
            "confidence": value.confidence,
            "timestamp": _isoformat(value.timestamp),
            "policy_reason": None,
        }

    def _serialize_activity(self, value: AssistantActivityEntry) -> dict[str, Any]:
        return {
            "activity_id": value.activity_id,
            "event_id": value.event_id,
            "event_type": value.event_type,
            "status": value.status,
            "summary": value.summary,
            "policy_reason": value.policy_reason,
            "created_at": _isoformat(value.created_at),
        }

    def _application_signature(
        self, value: AssistantActiveApplicationState
    ) -> str:
        return f"{value.process_name.lower()}::{value.window_title.lower()}"

    def _parse_time(self, value: str) -> datetime:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return _utcnow()

    def _bump_version(self) -> bool:
        self.state.version += 1
        return True
