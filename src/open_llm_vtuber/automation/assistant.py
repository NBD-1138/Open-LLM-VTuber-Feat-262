from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from loguru import logger

from ..mcpp.types import FormattedTool
from ..telemetry import (
    ClientTelemetryRuntime,
    TelemetryCapabilityAdvice,
    TelemetryEvent,
    TelemetrySettings,
)
from .runtime_state import AssistantRuntimeController
from .transport import (
    AssistantActiveApplicationMessage,
    AssistantContextSyncMessage,
    AssistantPlayerStateMessage,
    AssistantVoiceCommandStateMessage,
    AutomationAssistantSettingsMessage,
    AutomationCapabilitiesMessage,
    AutomationCapability,
    AutomationConfirmationResponseMessage,
    AutomationDecisionType,
    AutomationEmergencyStopMessage,
    AutomationLlmAccess,
    AutomationResultMessage,
    AutomationStatusMessage,
    AutomationTransport,
    MAX_REASON_LENGTH,
    VoiceCommandActivationMode,
    AutomationVoiceCommandAmbiguityResponseMessage,
    AutomationVoiceCommandResolveTestMessage,
)

SendToClient = Callable[[str, dict[str, Any]], Awaitable[None]]
SpeakToClient = Callable[[str, str], Awaitable[None]]
CanStartConversation = Callable[[str], bool]
StartConversation = Callable[[str, str, dict[str, Any]], Awaitable[bool]]

AUTOMATION_LOCAL_SERVER = "assistant_automation"
RECENT_DECISION_LIMIT = 20
RECENT_RESOLVER_ACTIVITY_LIMIT = 20
REPEAT_SUGGESTION_COOLDOWN_MS = 15_000
AUTOMATIC_PROFILE_ACTIVATION_COMMENTARY_COOLDOWN = timedelta(minutes=15)
GAME_FACTOID_COMMENTARY_COOLDOWN = timedelta(minutes=15)
GAME_FACTOID_RESEARCH_REFRESH = timedelta(minutes=45)
GAME_FACTOID_START_DELAY = timedelta(minutes=3)
GAME_FACTOID_LOOP_INTERVAL_SECONDS = 45
GAME_FACTOID_MAX_PER_PROFILE = 12
GAME_FACTOID_MIN_CACHE_BEFORE_REFRESH = 3
GAME_FACTOID_MAX_TEXT_LENGTH = 220
GAME_FACTOID_MAX_SOURCE_LENGTH = 120
GAME_FACTOID_MAX_CATEGORY_LENGTH = 48
GAME_FACTOID_MAX_TAGS = 8
SAFE_ERROR_LENGTH = 160
VOICE_COMMAND_EDGE_PUNCTUATION = re.compile(
    r'^[\s"\'`“”‘’.,!?;:()\[\]{}<>_-]+|[\s"\'`“”‘’.,!?;:()\[\]{}<>_-]+$'
)


VOICE_COMMAND_LEADING_PUNCTUATION = re.compile(
    r'^[\s"\'`鈥溾€濃€樷€?,!?;:()\[\]{}<>_-]+'
)

VOICE_COMMAND_HARD_BLOCK_REASONS = {
    "Imported automation profiles must be reviewed before voice use.",
    "Imported placeholder actions require manual replacement before execution.",
    "Imported response placeholders require manual replacement before execution.",
}

GAME_FACTOID_SPOILER_HINTS = (
    "ending",
    "final boss",
    "post-credit",
    "post credit",
    "secret ending",
    "true ending",
    "late game",
    "late-game",
    "final chapter",
    "plot twist",
    "major death",
    "betrays",
    "identity reveal",
)


def build_default_assistant_settings() -> AutomationAssistantSettingsMessage:
    return AutomationAssistantSettingsMessage(
        confirmation_phrases=["confirm", "do it", "yes, run it"],
        cancellation_phrases=["cancel", "never mind", "stop"],
        voice_command_wake_phrases=["assistant", "computer"],
        telemetry=TelemetrySettings().to_payload(),
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    return dt.isoformat()


def _normalize_phrase(value: str) -> str:
    lowered = value.strip().lower().replace("\u2019", "'")
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _sanitize_reason(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("reason cannot be empty")
    if len(value) > MAX_REASON_LENGTH:
        value = value[: MAX_REASON_LENGTH - 3].rstrip() + "..."
    return value


def _sanitize_error_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    sanitized = value.strip()
    sanitized = re.sub(r"[A-Za-z]:\\[^\s]+", "[path]", sanitized)
    sanitized = re.sub(r"file://\S+", "[path]", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized)
    if len(sanitized) > SAFE_ERROR_LENGTH:
        sanitized = sanitized[: SAFE_ERROR_LENGTH - 3].rstrip() + "..."
    return sanitized


def _clip_text(value: str, limit: int) -> str:
    cleaned = " ".join(str(value or "").split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3].rstrip()}..."


def _sanitize_game_factoid_text(value: str) -> str:
    cleaned = _clip_text(value, GAME_FACTOID_MAX_TEXT_LENGTH)
    if not cleaned:
        raise ValueError("fact_text cannot be empty")
    return cleaned


def _sanitize_short_optional_text(
    value: Optional[str],
    *,
    limit: int,
) -> Optional[str]:
    if value is None:
        return None
    cleaned = _clip_text(value, limit)
    return cleaned or None


def _sanitize_factoid_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("tags must be a list of short labels")
    tags: list[str] = []
    seen: set[str] = set()
    for raw in value[:GAME_FACTOID_MAX_TAGS]:
        tag = _clip_text(str(raw or ""), 24).strip().lower()
        if not tag:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _looks_spoilery_factoid(text: str) -> bool:
    lowered = _normalize_voice_phrase(text)
    return any(hint in lowered for hint in GAME_FACTOID_SPOILER_HINTS)


def _tool_text_result(
    payload: dict[str, Any],
    *,
    is_error: bool = False,
    internal_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata = dict(payload)
    if internal_metadata:
        metadata.update(internal_metadata)
    return {
        "metadata": metadata,
        "content_items": [
            {
                "type": "error" if is_error else "text",
                "text": json.dumps(payload, ensure_ascii=False),
            }
        ],
    }


def _normalize_voice_exact_phrase(value: str) -> str:
    lowered = value.replace("\u2019", "'").strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _normalize_voice_phrase(value: str) -> str:
    normalized = _normalize_voice_exact_phrase(value)
    normalized = VOICE_COMMAND_EDGE_PUNCTUATION.sub("", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _normalize_command_reference(value: str) -> str:
    camel_spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value.strip())
    camel_spaced = camel_spaced.replace("_", " ").replace("-", " ")
    return _normalize_voice_phrase(camel_spaced)


def _get_voice_command_phrases(capability: AutomationCapability) -> list[str]:
    phrases = [capability.label, *capability.aliases]
    seen: set[str] = set()
    result: list[str] = []
    for phrase in phrases:
        cleaned = phrase.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


@dataclass
class PendingConfirmation:
    request_id: str
    profile_id: str
    command_id: str
    command_label: str
    concise_reason: str
    risk: str
    created_at: datetime
    expires_at: datetime
    source: str = "vtuber"


@dataclass
class PendingAmbiguity:
    ambiguity_id: str
    transcript: str
    normalized_phrase: str
    candidate_command_ids: list[str]
    created_at: datetime
    expires_at: datetime


@dataclass
class DecisionRecord:
    decision_id: str
    decision_type: AutomationDecisionType
    profile_id: Optional[str]
    command_id: Optional[str]
    command_label: Optional[str]
    reason: Optional[str]
    risk: Optional[str]
    blocked_reason: Optional[str]
    request_id: Optional[str]
    created_at: datetime


@dataclass
class RequestRecord:
    request_id: str
    profile_id: str
    command_id: str
    command_label: str
    reason: str
    risk: str
    status: str
    created_at: datetime
    source: str = "vtuber"


@dataclass
class PendingProactiveCommentary:
    commentary_id: str
    commentary_type: str
    profile_id: str
    profile_display_name: str
    application_title: str
    created_at: datetime
    prompt_text: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameFactoidRecord:
    factoid_id: str
    profile_id: str
    profile_display_name: str
    application_title: str
    text: str
    source_name: Optional[str]
    source_url: Optional[str]
    category: Optional[str]
    tags: list[str]
    created_at: datetime
    last_selected_at: Optional[datetime] = None
    selection_count: int = 0


@dataclass
class ActiveGameContext:
    profile_id: str
    profile_display_name: str
    application_title: str
    process_name: str
    profile_selection_source: str


@dataclass
class ResolverActivityRecord:
    activity_id: str
    transcript: str
    normalized_phrase: Optional[str]
    activation_mode: VoiceCommandActivationMode
    activation_met: bool
    result: str
    matched_command_id: Optional[str]
    matched_command_label: Optional[str]
    candidate_command_labels: list[str]
    blocked_reason: Optional[str]
    request_id: Optional[str]
    fell_back_to_conversation: bool
    created_at: datetime


@dataclass
class AssistantClientState:
    active_profile_id: Optional[str] = None
    capability_revision: int = -1
    capabilities: dict[str, AutomationCapability] = field(default_factory=dict)
    settings: AutomationAssistantSettingsMessage = field(
        default_factory=build_default_assistant_settings
    )
    pending_confirmations: dict[str, PendingConfirmation] = field(default_factory=dict)
    request_records: dict[str, RequestRecord] = field(default_factory=dict)
    recent_decisions: list[DecisionRecord] = field(default_factory=list)
    proactive_commentary_cooldowns: dict[str, datetime] = field(default_factory=dict)
    pending_proactive_commentary: Optional[PendingProactiveCommentary] = None
    pending_ambiguity: Optional[PendingAmbiguity] = None
    resolver_activity: list[ResolverActivityRecord] = field(default_factory=list)
    voice_command_listening_active: bool = False
    last_blocked_reason: Optional[str] = None
    allow_remain_silent_tool: bool = False
    game_factoids: dict[str, list[GameFactoidRecord]] = field(default_factory=dict)
    game_fact_last_researched_at: dict[str, datetime] = field(default_factory=dict)
    runtime: AssistantRuntimeController = field(
        default_factory=AssistantRuntimeController
    )
    telemetry: Optional[ClientTelemetryRuntime] = None


class AutomationAssistantCoordinator:
    def __init__(
        self,
        transport: AutomationTransport,
        send_to_client: SendToClient,
        speak_to_client: SpeakToClient,
        can_start_conversation: CanStartConversation,
        start_conversation: StartConversation,
    ) -> None:
        self._transport = transport
        self._send_to_client = send_to_client
        self._speak_to_client = speak_to_client
        self._can_start_conversation = can_start_conversation
        self._start_conversation = start_conversation
        self._client_states: dict[str, AssistantClientState] = {}
        self._proactive_tasks: dict[str, asyncio.Task[None]] = {}

    def register_client(self, client_uid: str) -> None:
        if client_uid in self._client_states:
            self._ensure_proactive_task(client_uid)
            return

        async def on_event(event: TelemetryEvent) -> None:
            await self._handle_telemetry_event(client_uid, event)

        async def on_state_change() -> None:
            await self._handle_telemetry_state_change(client_uid)

        self._client_states[client_uid] = AssistantClientState(
            telemetry=ClientTelemetryRuntime(on_event, on_state_change)
        )
        self._ensure_proactive_task(client_uid)

    def unregister_client(self, client_uid: str) -> None:
        state = self._client_states.pop(client_uid, None)
        task = self._proactive_tasks.pop(client_uid, None)
        if task and not task.done():
            task.cancel()
        if state and state.telemetry:
            asyncio.create_task(self._dispose_client_telemetry(client_uid, state))

    async def _dispose_client_telemetry(
        self, client_uid: str, state: AssistantClientState
    ) -> None:
        if state.telemetry is None:
            return
        try:
            await state.telemetry.dispose()
        except Exception as e:
            logger.debug(
                f"Ignoring telemetry shutdown error for disconnected client {client_uid}: {e!r}"
            )

    def get_client_state(self, client_uid: str) -> AssistantClientState:
        self.register_client(client_uid)
        return self._client_states[client_uid]

    def _ensure_proactive_task(self, client_uid: str) -> None:
        if client_uid not in self._client_states:
            return
        existing = self._proactive_tasks.get(client_uid)
        if existing and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._proactive_tasks[client_uid] = loop.create_task(
            self._run_periodic_proactive_commentary_loop(client_uid)
        )

    def _set_remain_silent_allowed(
        self,
        state: AssistantClientState,
        *,
        allowed: bool,
    ) -> None:
        state.allow_remain_silent_tool = allowed

    def has_pending_confirmations(self, client_uid: str) -> bool:
        state = self.get_client_state(client_uid)
        self._expire_pending(state)
        return bool(state.pending_confirmations)

    def build_local_tools(self, client_uid: str) -> dict[str, FormattedTool]:
        return {
            "list_automation_capabilities": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                related_server=AUTOMATION_LOCAL_SERVER,
                description=(
                    "Return the current safe automation capability catalogue. "
                    "Use this before naming a command when you are unsure what is available."
                ),
                handler=lambda tool_input: self._handle_list_tool(
                    client_uid, tool_input
                ),
            ),
            "suggest_automation_command": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {
                        "profile_id": {
                            "type": "string",
                            "description": "Optional. The automation profile ID. Omit this to use the current active profile.",
                        },
                        "command_id": {
                            "type": "string",
                            "description": "The named automation command to recommend. Exact command_id is preferred, but the command label or alias is also accepted.",
                        },
                        "concise_reason": {
                            "type": "string",
                            "description": "A brief natural-language reason for the recommendation.",
                        },
                    },
                    "required": ["command_id", "concise_reason"],
                    "additionalProperties": False,
                },
                related_server=AUTOMATION_LOCAL_SERVER,
                description=(
                    "Recommend a currently available named automation command without executing it."
                ),
                handler=lambda tool_input: self._handle_suggest_tool(
                    client_uid, tool_input
                ),
            ),
            "request_automation_command": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {
                        "profile_id": {
                            "type": "string",
                            "description": "Optional. The automation profile ID. Omit this to use the current active profile.",
                        },
                        "command_id": {
                            "type": "string",
                            "description": "The named automation command to request. Exact command_id is preferred, but the command label or alias is also accepted.",
                        },
                        "concise_reason": {
                            "type": "string",
                            "description": "A brief natural-language reason for the request.",
                        },
                    },
                    "required": ["command_id", "concise_reason"],
                    "additionalProperties": False,
                },
                related_server=AUTOMATION_LOCAL_SERVER,
                description=(
                    "Request execution of a safe named automation command. "
                    "The backend will enforce autonomy, risk, confirmation, and availability rules."
                ),
                handler=lambda tool_input: self._handle_request_tool(
                    client_uid, tool_input
                ),
            ),
            "cancel_automation_command": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {
                        "request_id": {
                            "type": "string",
                            "description": "The automation request ID to cancel when one is known.",
                        }
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                related_server=AUTOMATION_LOCAL_SERVER,
                description="Cancel a pending confirmation or running automation command when appropriate.",
                handler=lambda tool_input: self._handle_cancel_tool(
                    client_uid, tool_input
                ),
            ),
            "get_game_factoid_context": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {
                        "profile_id": {
                            "type": "string",
                            "description": "Optional. Automation profile ID to inspect. Omit this to use the current active game profile.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Optional. Maximum number of cached factoids to return.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                related_server=AUTOMATION_LOCAL_SERVER,
                description=(
                    "Return the active game context, cached spoiler-safe factoids, and whether fresh web research is recommended."
                ),
                handler=lambda tool_input: self._handle_game_factoid_context_tool(
                    client_uid, tool_input
                ),
            ),
            "remember_game_factoid": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {
                        "profile_id": {
                            "type": "string",
                            "description": "Optional. Automation profile ID for the factoid. Omit this to use the current active game profile.",
                        },
                        "fact_text": {
                            "type": "string",
                            "description": "A short spoiler-safe fact about the game. Keep it broadly safe for players who are early in the game.",
                        },
                        "source_name": {
                            "type": "string",
                            "description": "Optional. Short source or publication name.",
                        },
                        "source_url": {
                            "type": "string",
                            "description": "Optional. Source URL for traceability.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional. A short category such as lore, development, soundtrack, feature, or trivia.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional short labels for the factoid.",
                        },
                    },
                    "required": ["fact_text"],
                    "additionalProperties": False,
                },
                related_server=AUTOMATION_LOCAL_SERVER,
                description=(
                    "Store one spoiler-safe game factoid in the local cache so later proactive commentary can reuse it without repeating the same search."
                ),
                handler=lambda tool_input: self._handle_remember_game_factoid_tool(
                    client_uid, tool_input
                ),
            ),
            "list_game_factoids": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {
                        "profile_id": {
                            "type": "string",
                            "description": "Optional. Automation profile ID to inspect. Omit this to use the current active game profile.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Optional. Maximum number of factoids to return.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                related_server=AUTOMATION_LOCAL_SERVER,
                description="List cached spoiler-safe game factoids for the current or specified game profile.",
                handler=lambda tool_input: self._handle_list_game_factoids_tool(
                    client_uid, tool_input
                ),
            ),
            "remain_silent": FormattedTool(
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                related_server=AUTOMATION_LOCAL_SERVER,
                description=(
                    "Explicitly decide that no automation comment or action is needed. "
                    "Call this once and then end the turn without any additional text."
                ),
                handler=lambda tool_input: self._handle_remain_silent_tool(
                    client_uid, tool_input
                ),
            ),
        }

    async def handle_transport_message(self, client_uid: str, message: Any) -> None:
        state = self.get_client_state(client_uid)
        updated = False

        if isinstance(message, AssistantActiveApplicationMessage):
            updated = state.runtime.apply_active_application(message) or updated
            await self._sync_telemetry_context(client_uid, state)
        elif isinstance(message, AssistantContextSyncMessage):
            previous_effective_profile_id = state.runtime.state.effective_profile_id
            updated = state.runtime.apply_context_sync(message) or updated
            self._queue_profile_activation_commentary(
                client_uid,
                state,
                message,
                previous_effective_profile_id=previous_effective_profile_id,
            )
            await self._sync_telemetry_context(client_uid, state)
        elif isinstance(message, AssistantPlayerStateMessage):
            updated = state.runtime.apply_player_state(message) or updated
        elif isinstance(message, AssistantVoiceCommandStateMessage):
            if state.voice_command_listening_active != message.listening_active:
                state.voice_command_listening_active = message.listening_active
                updated = True
        elif isinstance(message, AutomationVoiceCommandAmbiguityResponseMessage):
            handled = await self.handle_voice_command_ambiguity_response(
                client_uid, message
            )
            if handled:
                return
        elif isinstance(message, AutomationVoiceCommandResolveTestMessage):
            await self.handle_voice_command_resolve_test(client_uid, message)
            return
        elif isinstance(message, AutomationCapabilitiesMessage):
            if message.revision >= state.capability_revision:
                state.capability_revision = message.revision
                state.active_profile_id = message.active_profile_id
                state.capabilities = {
                    capability.command_id: capability
                    for capability in message.capabilities
                }
                updated = (
                    state.runtime.set_capability_revision(message.revision) or True
                )
                updated = True
                await self._sync_telemetry_context(client_uid, state)
            else:
                logger.debug(
                    "Ignoring stale automation capability revision "
                    f"{message.revision} for client {client_uid}; current revision is "
                    f"{state.capability_revision}."
                )
        elif isinstance(message, AutomationAssistantSettingsMessage):
            state.settings = message
            if state.telemetry is not None:
                await state.telemetry.update_settings(message.telemetry)
                state.runtime.set_telemetry_snapshot(state.telemetry.snapshot_payload())
            updated = True
        elif isinstance(message, AutomationStatusMessage):
            updated = (
                state.runtime.set_current_executions(message.running_requests)
                or updated
            )
            updated = (
                state.runtime.set_emergency_stopped(message.emergency_stopped)
                or updated
            )
            updated = self._expire_pending(state) or True
            await self._sync_telemetry_context(client_uid, state)
        elif isinstance(message, AutomationEmergencyStopMessage):
            updated = (
                state.runtime.set_emergency_stopped(message.emergency_stopped)
                or updated
            )
            if message.emergency_stopped:
                updated = (
                    self._cancel_all_pending(
                        state,
                        blocked_reason="Emergency stop is active.",
                        decision_type="blocked",
                    )
                    or updated
                )
        elif isinstance(message, AutomationResultMessage):
            updated = await self._handle_result_message(
                client_uid,
                state,
                message,
            )

        if updated:
            self._reconcile_pending_against_capabilities(state)
            state.runtime.set_pending_confirmation_count(
                len(state.pending_confirmations)
            )
            await self._emit_state(client_uid)
        await self._maybe_trigger_pending_proactive_commentary(client_uid, state)

    async def handle_confirmation_response(
        self,
        client_uid: str,
        message: AutomationConfirmationResponseMessage,
    ) -> bool:
        state = self.get_client_state(client_uid)
        self._expire_pending(state)
        pending = state.pending_confirmations.get(message.request_id)
        if not pending:
            await self._emit_state(client_uid)
            return False

        if message.action == "reject":
            state.pending_confirmations.pop(message.request_id, None)
            tracking = state.request_records.get(message.request_id)
            if tracking:
                tracking.status = "rejected"
            self._record_decision(
                state,
                decision_type="rejected",
                profile_id=pending.profile_id,
                command_id=pending.command_id,
                command_label=pending.command_label,
                reason=pending.concise_reason,
                risk=pending.risk,
                request_id=pending.request_id,
            )
            state.runtime.set_pending_confirmation_count(
                len(state.pending_confirmations)
            )
            state.runtime.mark_acted(
                summary=f"Rejected {pending.command_label}.",
                event_type="automation.confirmation_required",
            )
            await self._emit_state(client_uid)
            return True

        state.pending_confirmations.pop(message.request_id, None)
        self._record_decision(
            state,
            decision_type="confirmed",
            profile_id=pending.profile_id,
            command_id=pending.command_id,
            command_label=pending.command_label,
            reason=pending.concise_reason,
            risk=pending.risk,
            request_id=pending.request_id,
        )
        state.runtime.set_pending_confirmation_count(len(state.pending_confirmations))

        capability = state.capabilities.get(pending.command_id)
        if capability is None:
            blocked_reason = "The requested automation command is no longer available."
            state.last_blocked_reason = blocked_reason
            self._record_decision(
                state,
                decision_type="blocked",
                profile_id=pending.profile_id,
                command_id=pending.command_id,
                command_label=pending.command_label,
                reason=pending.concise_reason,
                risk=pending.risk,
                blocked_reason=blocked_reason,
                request_id=pending.request_id,
            )
            state.runtime.mark_acted(
                summary=f"{pending.command_label} is no longer available.",
                event_type="automation.blocked",
            )
            if pending.source == "player_voice":
                await self._maybe_speak_voice_command_stage(
                    client_uid,
                    state,
                    stage="blocked",
                    blocked_reason=blocked_reason,
                )
            await self._emit_state(client_uid)
            return False

        try:
            await self._start_execution(
                client_uid,
                state,
                capability,
                reason=pending.concise_reason,
                request_id=pending.request_id,
                source=pending.source,
            )
            if pending.source == "player_voice":
                await self._maybe_speak_voice_command_stage(
                    client_uid,
                    state,
                    stage="executing",
                    capability=capability,
                )
        except ValueError as exc:
            blocked_reason = _sanitize_error_text(str(exc)) or "The request is blocked."
            state.last_blocked_reason = blocked_reason
            tracking = state.request_records.get(pending.request_id)
            if tracking:
                tracking.status = "blocked"
            self._record_decision(
                state,
                decision_type="blocked",
                profile_id=pending.profile_id,
                command_id=pending.command_id,
                command_label=pending.command_label,
                reason=pending.concise_reason,
                risk=pending.risk,
                blocked_reason=blocked_reason,
                request_id=pending.request_id,
            )
            state.runtime.record_automation_result(
                status="blocked",
                command_label=pending.command_label,
                request_id=pending.request_id,
                error=blocked_reason,
            )
            if pending.source == "player_voice":
                await self._maybe_speak_voice_command_stage(
                    client_uid,
                    state,
                    stage="blocked",
                    capability=capability,
                    blocked_reason=blocked_reason,
                )
        await self._emit_state(client_uid)
        return True

    async def maybe_handle_confirmation_phrase(
        self,
        client_uid: str,
        text: str,
    ) -> bool:
        state = self.get_client_state(client_uid)
        self._expire_pending(state)
        normalized = _normalize_phrase(text)
        if not normalized or not state.pending_confirmations:
            return False

        confirmation_phrases = {
            _normalize_phrase(item) for item in state.settings.confirmation_phrases
        }
        cancellation_phrases = {
            _normalize_phrase(item) for item in state.settings.cancellation_phrases
        }

        if normalized in confirmation_phrases:
            latest_pending = self._get_latest_pending(state)
            if latest_pending is None:
                return False
            message = AutomationConfirmationResponseMessage(
                request_id=latest_pending.request_id,
                action="confirm",
                source="player_voice",
            )
            return await self.handle_confirmation_response(client_uid, message)

        if normalized in cancellation_phrases:
            latest_pending = self._get_latest_pending(state)
            if latest_pending is None:
                return False
            message = AutomationConfirmationResponseMessage(
                request_id=latest_pending.request_id,
                action="reject",
                source="player_voice",
            )
            return await self.handle_confirmation_response(client_uid, message)

        return False

    def _strip_leading_wake_phrase(
        self,
        text: str,
        wake_phrases: list[str],
    ) -> tuple[bool, str]:
        exact_text = _normalize_voice_exact_phrase(text)
        if not exact_text:
            return False, exact_text

        candidate = VOICE_COMMAND_LEADING_PUNCTUATION.sub("", exact_text)
        phrases = sorted(
            (
                _normalize_voice_exact_phrase(phrase)
                for phrase in wake_phrases
                if phrase and phrase.strip()
            ),
            key=len,
            reverse=True,
        )
        for wake_phrase in phrases:
            if not wake_phrase or not candidate.startswith(wake_phrase):
                continue
            remainder = candidate[len(wake_phrase) :]
            if remainder and not VOICE_COMMAND_LEADING_PUNCTUATION.match(remainder):
                continue
            return True, VOICE_COMMAND_LEADING_PUNCTUATION.sub("", remainder)
        return False, exact_text

    def _evaluate_voice_command_activation(
        self,
        state: AssistantClientState,
        text: str,
    ) -> tuple[bool, str, VoiceCommandActivationMode]:
        activation_mode = state.settings.voice_command_activation_mode
        exact_text = _normalize_voice_exact_phrase(text)
        if activation_mode == "disabled":
            return False, exact_text, activation_mode
        if activation_mode == "always_listening":
            return True, exact_text, activation_mode
        if activation_mode == "push_to_command":
            return (
                state.voice_command_listening_active,
                exact_text,
                activation_mode,
            )
        activated, stripped_text = self._strip_leading_wake_phrase(
            text,
            state.settings.voice_command_wake_phrases,
        )
        return activated, stripped_text, activation_mode

    def _is_voice_command_capability_eligible(
        self,
        state: AssistantClientState,
        capability: AutomationCapability,
    ) -> bool:
        if not state.active_profile_id or capability.profile_id != state.active_profile_id:
            return False
        if not capability.enabled:
            return False
        if "player_voice" not in set(capability.allowed_trigger_sources):
            return False

        automation_state = self._transport.get_client_state(self._find_client_uid(state))
        profile = automation_state.profiles.get(capability.profile_id)
        if profile is None or not profile.enabled:
            return False
        command = automation_state.get_command(capability.profile_id, capability.command_id)
        if command is None or not command.enabled:
            return False
        if capability.blocked_reason in VOICE_COMMAND_HARD_BLOCK_REASONS:
            return False
        return True

    def _resolve_voice_command(
        self,
        state: AssistantClientState,
        text: str,
    ) -> dict[str, Any]:
        activation_met, command_text, activation_mode = (
            self._evaluate_voice_command_activation(state, text)
        )
        exact_phrase = _normalize_voice_exact_phrase(command_text)
        normalized_phrase = _normalize_voice_phrase(command_text)
        explicit_command_address = (
            activation_met and activation_mode in {"wake_phrase", "push_to_command"}
        )

        result: dict[str, Any] = {
            "activation_mode": activation_mode,
            "activation_met": activation_met,
            "exact_phrase": exact_phrase or None,
            "normalized_phrase": normalized_phrase or None,
            "capability": None,
            "candidates": [],
            "result": "activation_not_met",
            "blocked_reason": None,
            "explicit_command_address": explicit_command_address,
        }
        if not activation_met:
            return result

        if not state.active_profile_id:
            result["result"] = "blocked_match"
            result["blocked_reason"] = "No active automation profile is selected."
            return result

        eligible_capabilities = [
            capability
            for capability in state.capabilities.values()
            if self._is_voice_command_capability_eligible(state, capability)
        ]
        if not eligible_capabilities:
            result["result"] = "no_match"
            return result

        def _append_unique(
            matches: list[AutomationCapability],
            seen: set[str],
            capability: AutomationCapability,
        ) -> None:
            if capability.command_id in seen:
                return
            seen.add(capability.command_id)
            matches.append(capability)

        exact_matches: list[AutomationCapability] = []
        normalized_matches: list[AutomationCapability] = []
        exact_seen: set[str] = set()
        normalized_seen: set[str] = set()

        for capability in eligible_capabilities:
            for phrase in _get_voice_command_phrases(capability):
                if exact_phrase and _normalize_voice_exact_phrase(phrase) == exact_phrase:
                    _append_unique(exact_matches, exact_seen, capability)
                if (
                    normalized_phrase
                    and _normalize_voice_phrase(phrase) == normalized_phrase
                ):
                    _append_unique(normalized_matches, normalized_seen, capability)

        if len(exact_matches) == 1:
            result["capability"] = exact_matches[0]
            result["result"] = "exact_match"
            return result
        if len(exact_matches) > 1:
            result["candidates"] = exact_matches
            result["result"] = "ambiguous_match"
            return result
        if len(normalized_matches) == 1:
            result["capability"] = normalized_matches[0]
            result["result"] = "normalized_match"
            return result
        if len(normalized_matches) > 1:
            result["candidates"] = normalized_matches
            result["result"] = "ambiguous_match"
            return result

        result["result"] = "no_match"
        return result

    def _record_resolver_activity(
        self,
        state: AssistantClientState,
        *,
        transcript: str,
        normalized_phrase: Optional[str],
        activation_mode: VoiceCommandActivationMode,
        activation_met: bool,
        result: str,
        matched_command_id: Optional[str] = None,
        matched_command_label: Optional[str] = None,
        candidate_command_labels: Optional[list[str]] = None,
        blocked_reason: Optional[str] = None,
        request_id: Optional[str] = None,
        fell_back_to_conversation: bool = False,
    ) -> None:
        state.resolver_activity.insert(
            0,
            ResolverActivityRecord(
                activity_id=str(uuid4()),
                transcript=transcript,
                normalized_phrase=normalized_phrase,
                activation_mode=activation_mode,
                activation_met=activation_met,
                result=result,
                matched_command_id=matched_command_id,
                matched_command_label=matched_command_label,
                candidate_command_labels=candidate_command_labels or [],
                blocked_reason=blocked_reason,
                request_id=request_id,
                fell_back_to_conversation=fell_back_to_conversation,
                created_at=_utcnow(),
            ),
        )
        if len(state.resolver_activity) > RECENT_RESOLVER_ACTIVITY_LIMIT:
            state.resolver_activity = state.resolver_activity[
                :RECENT_RESOLVER_ACTIVITY_LIMIT
            ]

    def _clear_pending_ambiguity(self, state: AssistantClientState) -> bool:
        if state.pending_ambiguity is None:
            return False
        state.pending_ambiguity = None
        return True

    def _create_pending_ambiguity(
        self,
        state: AssistantClientState,
        *,
        transcript: str,
        normalized_phrase: str,
        capabilities: list[AutomationCapability],
    ) -> PendingAmbiguity:
        created_at = _utcnow()
        pending = PendingAmbiguity(
            ambiguity_id=str(uuid4()),
            transcript=transcript,
            normalized_phrase=normalized_phrase,
            candidate_command_ids=[capability.command_id for capability in capabilities],
            created_at=created_at,
            expires_at=created_at
            + timedelta(milliseconds=state.settings.voice_command_ambiguity_timeout_ms),
        )
        state.pending_ambiguity = pending
        return pending

    def _expire_pending_ambiguity(self, state: AssistantClientState) -> bool:
        pending = state.pending_ambiguity
        if pending is None:
            return False
        if pending.expires_at > _utcnow():
            return False
        state.pending_ambiguity = None
        return True

    async def _maybe_handle_direct_command_phrase(
        self,
        client_uid: str,
        state: AssistantClientState,
        text: str,
        *,
        source: str,
    ) -> tuple[bool, Optional[str], bool]:
        if source not in {"player", "player_voice"}:
            return False, None, False

        resolution = self._resolve_voice_command(state, text)
        resolution_result = str(resolution["result"])
        activation_mode = resolution["activation_mode"]
        activation_met = bool(resolution["activation_met"])
        normalized_phrase = resolution.get("normalized_phrase")

        if resolution_result == "activation_not_met":
            self._record_resolver_activity(
                state,
                transcript=text,
                normalized_phrase=normalized_phrase,
                activation_mode=activation_mode,
                activation_met=False,
                result="activation_not_met",
            )
            return False, None, False

        self._clear_pending_ambiguity(state)

        if resolution_result == "no_match":
            fell_back = activation_met
            self._record_resolver_activity(
                state,
                transcript=text,
                normalized_phrase=normalized_phrase,
                activation_mode=activation_mode,
                activation_met=activation_met,
                result="no_match",
                fell_back_to_conversation=fell_back,
            )
            return False, None, bool(resolution["explicit_command_address"])

        if resolution_result == "blocked_match":
            blocked_reason = str(
                resolution.get("blocked_reason")
                or "Voice command resolution is unavailable."
            )
            self._record_resolver_activity(
                state,
                transcript=text,
                normalized_phrase=normalized_phrase,
                activation_mode=activation_mode,
                activation_met=activation_met,
                result="blocked_match",
                blocked_reason=blocked_reason,
                fell_back_to_conversation=True,
            )
            return False, blocked_reason, bool(resolution["explicit_command_address"])

        if resolution_result == "ambiguous_match":
            capabilities = list(resolution.get("candidates") or [])
            pending = self._create_pending_ambiguity(
                state,
                transcript=text,
                normalized_phrase=normalized_phrase or "",
                capabilities=capabilities,
            )
            candidate_labels = [capability.label for capability in capabilities]
            self._record_resolver_activity(
                state,
                transcript=text,
                normalized_phrase=normalized_phrase,
                activation_mode=activation_mode,
                activation_met=activation_met,
                result="ambiguous_match",
                candidate_command_labels=candidate_labels,
            )
            await self._emit_state(client_uid)
            await self._maybe_speak_voice_command_stage(
                client_uid,
                state,
                stage="ambiguous",
                candidate_labels=candidate_labels,
            )
            logger.debug(
                "Created pending voice-command ambiguity "
                f"{pending.ambiguity_id} for client {client_uid}."
            )
            return True, None, False

        capability = resolution.get("capability")
        if not isinstance(capability, AutomationCapability):
            return False, None, False

        reason = _sanitize_reason(
            f'The player explicitly requested "{capability.label}".'
        )
        result = await self._handle_request_tool(
            client_uid,
            {
                "profile_id": capability.profile_id,
                "command_id": capability.command_id,
                "concise_reason": reason,
            },
            request_source="player_voice",
        )
        payload = result.get("metadata", {}) if isinstance(result, dict) else {}
        status = str(payload.get("status") or "").strip().lower()
        blocked_reason = _sanitize_error_text(payload.get("blocked_reason"))
        request_id = str(payload.get("request_id") or "").strip() or None
        final_result = (
            "blocked_match"
            if status == "blocked"
            else resolution_result
        )
        self._record_resolver_activity(
            state,
            transcript=text,
            normalized_phrase=normalized_phrase,
            activation_mode=activation_mode,
            activation_met=activation_met,
            result=final_result,
            matched_command_id=capability.command_id,
            matched_command_label=capability.label,
            blocked_reason=blocked_reason,
            request_id=request_id,
        )

        if status == "started":
            await self._maybe_speak_voice_command_stage(
                client_uid,
                state,
                stage="executing",
                capability=capability,
            )
        elif status == "pending_confirmation":
            await self._maybe_speak_voice_command_stage(
                client_uid,
                state,
                stage="confirmation_required",
                capability=capability,
            )
        elif status == "blocked":
            await self._maybe_speak_voice_command_stage(
                client_uid,
                state,
                stage="blocked",
                capability=capability,
                blocked_reason=blocked_reason,
            )

        return True, None, False

    async def prepare_conversation_metadata(
        self,
        client_uid: str,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        *,
        source: str = "player",
    ) -> tuple[bool, dict[str, Any]]:
        state = self.get_client_state(client_uid)
        payload = dict(metadata or {})
        normalized_source = str(payload.get("source") or source or "player").lower()
        updated = self._expire_pending_ambiguity(state)
        self._set_remain_silent_allowed(state, allowed=False)

        if normalized_source == "twitch":
            if not payload.get("assistant_recorded"):
                state.runtime.record_twitch_event(text=text, metadata=payload)
            message_kind = str(payload.get("message_kind") or "").strip().lower()
            is_twitch_notification = (
                message_kind == "notification"
                or bool(payload.get("notification"))
                or str(payload.get("category") or "").strip().lower() == "system"
            )
            if is_twitch_notification:
                suppress_reason = state.runtime.should_suppress_twitch_forward()
                if suppress_reason:
                    state.runtime.mark_remained_silent(suppress_reason)
                    await self._emit_state(client_uid)
                    return True, payload
        else:
            state.runtime.record_direct_request(text, source=normalized_source)
            suppress, blocked_reason, explicit_command_address = (
                await self._maybe_handle_direct_command_phrase(
                client_uid,
                state,
                text,
                source=normalized_source,
                )
            )
            if suppress:
                return True, payload
            if blocked_reason:
                payload["assistant_voice_command_blocked_reason"] = blocked_reason
            if explicit_command_address:
                payload["assistant_voice_command_requested"] = True

        payload["assistant_context_text"] = state.runtime.build_context_prompt(
            capabilities=self._serialize_capabilities(state),
            pending_confirmations=self._serialize_pending(state),
        )
        payload["assistant_context_text"] += (
            "\n\nTOOL ROUTING\n"
            "- The remain_silent tool is reserved for proactive commentary turns only.\n"
            "- For direct player or Twitch conversations, reply normally instead of using remain_silent."
        )
        if payload.get("assistant_voice_command_requested"):
            payload["assistant_context_text"] += (
                "\n\nVOICE COMMAND ROUTING\n"
                "- The player explicitly addressed the assistant as a command, but no deterministic alias matched.\n"
                "- Reply normally or use the safe named-command tools if one clearly applies.\n"
                "- Never expose key bindings, macro steps, imported source text, or private paths."
            )
        elif payload.get("assistant_voice_command_blocked_reason"):
            payload["assistant_context_text"] += (
                "\n\nVOICE COMMAND ROUTING\n"
                f"- A direct voice-command attempt was blocked: {payload['assistant_voice_command_blocked_reason']}"
            )
        del updated
        await self._emit_state(client_uid)
        return False, payload

    def _build_voice_command_resolve_result_payload(
        self,
        resolution: dict[str, Any],
    ) -> dict[str, Any]:
        capability = resolution.get("capability")
        candidates = list(resolution.get("candidates") or [])
        blocked_reason = resolution.get("blocked_reason")
        return {
            "type": "automation/voice-command-resolve-result",
            "transcript": resolution.get("transcript"),
            "normalized_phrase": resolution.get("normalized_phrase"),
            "activation_mode": resolution["activation_mode"],
            "activation_met": resolution["activation_met"],
            "result": resolution["result"],
            "matched_command_id": capability.command_id
            if isinstance(capability, AutomationCapability)
            else None,
            "matched_command_label": capability.label
            if isinstance(capability, AutomationCapability)
            else None,
            "candidate_command_labels": [
                capability.label
                for capability in candidates
                if isinstance(capability, AutomationCapability)
            ],
            "blocked_reason": blocked_reason,
        }

    def _voice_command_stage_text(
        self,
        *,
        stage: str,
        capability: Optional[AutomationCapability] = None,
        blocked_reason: Optional[str] = None,
        candidate_labels: Optional[list[str]] = None,
    ) -> Optional[str]:
        label = capability.label if capability else "that command"
        if stage == "matched":
            return f"I heard {label}."
        if stage == "confirmation_required":
            return f"I can run {label}. Say confirm to continue."
        if stage == "executing":
            return f"Running {label}."
        if stage == "completed":
            return f"{label} finished."
        if stage == "failed":
            return f"I couldn't run {label}."
        if stage == "blocked":
            reason = blocked_reason or "It is blocked right now."
            return f"I couldn't run {label}: {reason}"
        if stage == "ambiguous":
            if candidate_labels:
                labels = ", ".join(candidate_labels[:3])
                return f"I found more than one command: {labels}."
            return "I found more than one matching command."
        return None

    async def _maybe_speak_voice_command_stage(
        self,
        client_uid: str,
        state: AssistantClientState,
        *,
        stage: str,
        capability: Optional[AutomationCapability] = None,
        blocked_reason: Optional[str] = None,
        candidate_labels: Optional[list[str]] = None,
    ) -> None:
        acknowledgement_mode = state.settings.voice_command_acknowledgement_mode
        if stage == "blocked" and not state.settings.announce_blocked_command_requests:
            return
        if stage in {"completed", "failed"} and not state.settings.result_acknowledgements_enabled:
            return
        if stage in {"matched", "executing", "ambiguous"} and acknowledgement_mode == "none":
            return

        text = self._voice_command_stage_text(
            stage=stage,
            capability=capability,
            blocked_reason=blocked_reason,
            candidate_labels=candidate_labels,
        )
        if not text:
            return

        # Preserve the acknowledgement-mode contract even though both modes
        # currently share the same owned fixed phrases.
        await self._speak_to_client(client_uid, text)

    async def handle_voice_command_resolve_test(
        self,
        client_uid: str,
        message: AutomationVoiceCommandResolveTestMessage,
    ) -> None:
        state = self.get_client_state(client_uid)
        self._expire_pending_ambiguity(state)
        resolution = self._resolve_voice_command(state, message.text)
        resolution["transcript"] = message.text
        capability = resolution.get("capability")
        if (
            resolution.get("result") in {"exact_match", "normalized_match"}
            and isinstance(capability, AutomationCapability)
        ):
            llm_access, blocked_reason = self._derive_access(
                state,
                capability,
                source="player_voice",
            )
            if blocked_reason or llm_access in {"none", "suggest"}:
                resolution["result"] = "blocked_match"
                resolution["blocked_reason"] = (
                    blocked_reason
                    or "That command may be suggested but not executed."
                )
        await self._send_to_client(
            client_uid,
            self._build_voice_command_resolve_result_payload(resolution),
        )
        await self._emit_state(client_uid)

    async def handle_voice_command_ambiguity_response(
        self,
        client_uid: str,
        message: AutomationVoiceCommandAmbiguityResponseMessage,
    ) -> bool:
        state = self.get_client_state(client_uid)
        self._expire_pending_ambiguity(state)
        pending = state.pending_ambiguity
        if pending is None or pending.ambiguity_id != message.ambiguity_id:
            await self._emit_state(client_uid)
            return False

        state.pending_ambiguity = None
        if message.action == "cancel":
            await self._emit_state(client_uid)
            return True

        capability = state.capabilities.get(message.command_id or "")
        if (
            capability is None
            or capability.command_id not in pending.candidate_command_ids
        ):
            state.last_blocked_reason = "The selected voice command is no longer available."
            await self._emit_state(client_uid)
            return False

        reason = _sanitize_reason(
            f'The player selected "{capability.label}" from a voice-command ambiguity.'
        )
        result = await self._handle_request_tool(
            client_uid,
            {
                "profile_id": capability.profile_id,
                "command_id": capability.command_id,
                "concise_reason": reason,
            },
            request_source="player_voice",
        )
        payload = result.get("metadata", {}) if isinstance(result, dict) else {}
        status = str(payload.get("status") or "").strip().lower()
        blocked_reason = _sanitize_error_text(payload.get("blocked_reason"))

        if status == "started":
            await self._maybe_speak_voice_command_stage(
                client_uid,
                state,
                stage="executing",
                capability=capability,
            )
        elif status == "pending_confirmation":
            await self._maybe_speak_voice_command_stage(
                client_uid,
                state,
                stage="confirmation_required",
                capability=capability,
            )
        elif status == "blocked":
            await self._maybe_speak_voice_command_stage(
                client_uid,
                state,
                stage="blocked",
                capability=capability,
                blocked_reason=blocked_reason,
            )

        await self._emit_state(client_uid)
        return True

    async def record_twitch_event_for_all_clients(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        for client_uid in list(self._client_states.keys()):
            state = self.get_client_state(client_uid)
            state.runtime.record_twitch_event(text=text, metadata=metadata or {})
            await self._emit_state(client_uid)

    async def record_system_warning_for_all_clients(
        self,
        text: str,
        *,
        source: str = "system",
    ) -> None:
        for client_uid in list(self._client_states.keys()):
            state = self.get_client_state(client_uid)
            state.runtime.record_system_warning(text, source=source)
            await self._emit_state(client_uid)

    async def set_vtuber_speaking(
        self,
        client_uid: str,
        speaking: bool,
    ) -> None:
        state = self.get_client_state(client_uid)
        if state.runtime.set_vtuber_speaking(speaking):
            await self._emit_state(client_uid)
        if not speaking:
            await self._maybe_trigger_pending_proactive_commentary(client_uid, state)

    def _build_proactive_commentary_cooldown_key(
        self,
        commentary_type: str,
        profile_id: str,
    ) -> str:
        return f"{commentary_type}:{profile_id.strip().casefold()}"

    def _commentary_cooldown_duration(self, commentary_type: str) -> timedelta:
        if commentary_type == "game_factoid_commentary":
            return GAME_FACTOID_COMMENTARY_COOLDOWN
        return AUTOMATIC_PROFILE_ACTIVATION_COMMENTARY_COOLDOWN

    def _prune_proactive_commentary_cooldowns(
        self,
        state: AssistantClientState,
        now: datetime,
    ) -> None:
        expired_keys = [
            key
            for key, started_at in state.proactive_commentary_cooldowns.items()
            if now - started_at
            >= self._commentary_cooldown_duration(key.partition(":")[0])
        ]
        for key in expired_keys:
            state.proactive_commentary_cooldowns.pop(key, None)

    def _is_proactive_commentary_on_cooldown(
        self,
        state: AssistantClientState,
        commentary_type: str,
        profile_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        current_time = now or _utcnow()
        self._prune_proactive_commentary_cooldowns(state, current_time)
        cooldown_key = self._build_proactive_commentary_cooldown_key(
            commentary_type,
            profile_id,
        )
        previous_started_at = state.proactive_commentary_cooldowns.get(cooldown_key)
        return (
            previous_started_at is not None
            and current_time - previous_started_at
            < self._commentary_cooldown_duration(commentary_type)
        )

    def _mark_proactive_commentary_started(
        self,
        state: AssistantClientState,
        commentary_type: str,
        profile_id: str,
        *,
        started_at: Optional[datetime] = None,
    ) -> None:
        current_time = started_at or _utcnow()
        self._prune_proactive_commentary_cooldowns(state, current_time)
        cooldown_key = self._build_proactive_commentary_cooldown_key(
            commentary_type,
            profile_id,
        )
        state.proactive_commentary_cooldowns[cooldown_key] = current_time

    def _queue_profile_activation_commentary(
        self,
        client_uid: str,
        state: AssistantClientState,
        message: AssistantContextSyncMessage,
        *,
        previous_effective_profile_id: Optional[str],
    ) -> None:
        if (
            message.profile_selection_mode != "automatic"
            or message.profile_selection_source != "automatic"
        ):
            state.pending_proactive_commentary = None
            return
        if not message.effective_profile_id:
            state.pending_proactive_commentary = None
            return
        if previous_effective_profile_id == message.effective_profile_id:
            return
        if self._is_proactive_commentary_on_cooldown(
            state,
            "automatic_profile_activated",
            message.effective_profile_id,
        ):
            state.pending_proactive_commentary = None
            return

        active_application = state.runtime.state.active_application
        if active_application is None:
            return

        profile = self._transport.get_client_state(client_uid).profiles.get(
            message.effective_profile_id
        )
        profile_display_name = (
            profile.display_name if profile else message.effective_profile_id
        )
        application_title = (
            active_application.window_title.strip()
            or active_application.process_name
        )

        state.pending_proactive_commentary = PendingProactiveCommentary(
            commentary_id=str(uuid4()),
            commentary_type="automatic_profile_activated",
            profile_id=message.effective_profile_id,
            profile_display_name=profile_display_name,
            application_title=application_title,
            created_at=_utcnow(),
        )

    async def _run_periodic_proactive_commentary_loop(
        self,
        client_uid: str,
    ) -> None:
        try:
            while client_uid in self._client_states:
                await asyncio.sleep(GAME_FACTOID_LOOP_INTERVAL_SECONDS)
                state = self._client_states.get(client_uid)
                if state is None:
                    return
                try:
                    queued = self._queue_periodic_game_factoid_commentary(
                        client_uid,
                        state,
                    )
                    if queued:
                        await self._emit_state(client_uid)
                    await self._maybe_trigger_pending_proactive_commentary(
                        client_uid,
                        state,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Periodic proactive commentary check failed for "
                        f"{client_uid}: {exc}"
                    )
        except asyncio.CancelledError:
            raise
        finally:
            current_task = asyncio.current_task()
            if self._proactive_tasks.get(client_uid) is current_task:
                self._proactive_tasks.pop(client_uid, None)

    def _resolve_active_game_context(
        self,
        client_uid: str,
        state: AssistantClientState,
        *,
        preferred_profile_id: Optional[str] = None,
        allow_inactive_profile: bool = False,
    ) -> Optional[ActiveGameContext]:
        transport_state = self._transport.get_client_state(client_uid)
        profile_id = (
            preferred_profile_id
            or state.runtime.state.effective_profile_id
            or state.active_profile_id
        )
        if not profile_id:
            return None

        profile = transport_state.profiles.get(profile_id)
        if profile is None:
            return None

        active_application = state.runtime.state.active_application
        process_name = active_application.process_name if active_application else ""
        profile_process_names = {
            str(name).strip().casefold()
            for name in profile.process_names
            if str(name).strip()
        }
        matches_active_process = (
            not profile_process_names
            or (
                bool(process_name)
                and process_name.strip().casefold() in profile_process_names
            )
        )
        if not allow_inactive_profile and (
            active_application is None or not matches_active_process
        ):
            return None

        application_title = (
            active_application.window_title.strip()
            if active_application and active_application.window_title.strip()
            else profile.display_name
        )
        process_title = process_name or (profile.process_names[0] if profile.process_names else "")
        return ActiveGameContext(
            profile_id=profile.profile_id,
            profile_display_name=profile.display_name or profile.profile_id,
            application_title=application_title,
            process_name=process_title,
            profile_selection_source=state.runtime.state.profile_selection_source,
        )

    def _sorted_game_factoids(
        self,
        state: AssistantClientState,
        profile_id: str,
    ) -> list[GameFactoidRecord]:
        return sorted(
            state.game_factoids.get(profile_id, []),
            key=lambda item: (
                item.selection_count,
                item.last_selected_at or datetime.min.replace(tzinfo=timezone.utc),
                item.created_at,
            ),
        )

    def _serialize_game_factoid(
        self,
        record: GameFactoidRecord,
    ) -> dict[str, Any]:
        return {
            "factoid_id": record.factoid_id,
            "profile_id": record.profile_id,
            "profile_display_name": record.profile_display_name,
            "application_title": record.application_title,
            "fact_text": record.text,
            "source_name": record.source_name,
            "source_url": record.source_url,
            "category": record.category,
            "tags": list(record.tags),
            "created_at": _isoformat(record.created_at),
            "last_selected_at": (
                _isoformat(record.last_selected_at)
                if record.last_selected_at
                else None
            ),
            "selection_count": record.selection_count,
        }

    def _game_factoid_research_needed(
        self,
        state: AssistantClientState,
        profile_id: str,
        now: Optional[datetime] = None,
    ) -> bool:
        current_time = now or _utcnow()
        factoid_count = len(state.game_factoids.get(profile_id, []))
        last_researched_at = state.game_fact_last_researched_at.get(profile_id)
        if factoid_count < GAME_FACTOID_MIN_CACHE_BEFORE_REFRESH:
            return True
        if last_researched_at is None:
            return True
        return current_time - last_researched_at >= GAME_FACTOID_RESEARCH_REFRESH

    def _build_game_factoid_prompt(
        self,
        context: ActiveGameContext,
    ) -> str:
        return (
            f"Share one brief, engaging, spoiler-safe comment about "
            f"{context.profile_display_name} to keep the conversation going. "
            "Keep it natural and under two sentences."
        )

    def _build_game_factoid_context_prompt(
        self,
        state: AssistantClientState,
        pending: PendingProactiveCommentary,
    ) -> str:
        base_context = state.runtime.build_context_prompt(
            capabilities=self._serialize_capabilities(state),
            pending_confirmations=self._serialize_pending(state),
        )
        cached_count = len(state.game_factoids.get(pending.profile_id, []))
        last_researched_at = state.game_fact_last_researched_at.get(pending.profile_id)
        research_needed = bool(
            pending.metadata.get("assistant_game_fact_research_needed")
        )
        lines = [
            base_context,
            "",
            "GAME FACTOID COMMENTARY",
            f"- Game profile: {pending.profile_display_name}",
            f"- Active application title: {pending.application_title}",
            f"- Cached spoiler-safe factoids: {cached_count}",
            (
                "- Last factoid research: "
                f"{_isoformat(last_researched_at)}"
                if last_researched_at
                else "- Last factoid research: none yet"
            ),
            f"- Fresh web research recommended: {'yes' if research_needed else 'no'}",
            "- Start by calling get_game_factoid_context.",
            "- If fresh research is recommended and a web-search MCP tool is available, use it before speaking.",
            "- Prefer safe trivia about development, soundtrack, setting flavor, release history, well-known features, or beginner-safe mechanics.",
            "- Avoid spoilers, plot twists, endings, secret bosses, hidden routes, exploit paths, walkthrough steps, puzzle answers, or late-game surprises.",
            "- After finding a good new fact, call remember_game_factoid before speaking.",
            "- Final spoken response: one or two in-character sentences that feel conversational and invite a light reply.",
        ]
        return "\n".join(lines)

    def _queue_periodic_game_factoid_commentary(
        self,
        client_uid: str,
        state: AssistantClientState,
    ) -> bool:
        current_pending = state.pending_proactive_commentary
        if current_pending and current_pending.commentary_type != "game_factoid_commentary":
            return False

        if not state.runtime.state.proactive_commentary_enabled:
            if current_pending and current_pending.commentary_type == "game_factoid_commentary":
                state.pending_proactive_commentary = None
                return True
            return False

        context = self._resolve_active_game_context(client_uid, state)
        if context is None:
            if current_pending and current_pending.commentary_type == "game_factoid_commentary":
                state.pending_proactive_commentary = None
                return True
            return False

        if (
            current_pending
            and current_pending.commentary_type == "game_factoid_commentary"
            and current_pending.profile_id == context.profile_id
        ):
            return False

        now = _utcnow()
        last_profile_transition_at = state.runtime.state.last_profile_transition_at
        if last_profile_transition_at and now - last_profile_transition_at < GAME_FACTOID_START_DELAY:
            return False

        if self._is_proactive_commentary_on_cooldown(
            state,
            "game_factoid_commentary",
            context.profile_id,
            now=now,
        ):
            return False

        if (
            state.pending_confirmations
            or state.pending_ambiguity is not None
            or state.runtime.state.current_executions
        ):
            return False

        research_needed = self._game_factoid_research_needed(
            state,
            context.profile_id,
            now,
        )
        state.pending_proactive_commentary = PendingProactiveCommentary(
            commentary_id=str(uuid4()),
            commentary_type="game_factoid_commentary",
            profile_id=context.profile_id,
            profile_display_name=context.profile_display_name,
            application_title=context.application_title,
            created_at=now,
            prompt_text=self._build_game_factoid_prompt(context),
            metadata={
                "assistant_game_fact_research_needed": research_needed,
                "assistant_game_fact_profile_selection_source": context.profile_selection_source,
                "assistant_game_fact_process_name": context.process_name,
            },
        )
        return True

    async def _maybe_trigger_pending_proactive_commentary(
        self,
        client_uid: str,
        state: AssistantClientState,
    ) -> bool:
        pending = state.pending_proactive_commentary
        if pending is None:
            return False
        if self._is_proactive_commentary_on_cooldown(
            state,
            pending.commentary_type,
            pending.profile_id,
        ):
            state.pending_proactive_commentary = None
            return False
        if state.runtime.state.vtuber_speaking or state.runtime.state.player_speaking:
            return False
        if not self._can_start_conversation(client_uid):
            return False

        prompt_text = pending.prompt_text or (
            f"{pending.application_title} just launched and the "
            f"{pending.profile_display_name} game profile is now active. "
            "Comment on that briefly in character. "
            "Keep it natural and under two sentences."
        )
        metadata = {
            "skip_history": True,
            "skip_memory": True,
            "assistant_proactive_commentary": True,
            "assistant_commentary_type": pending.commentary_type,
            "assistant_profile_id": pending.profile_id,
            "assistant_profile_display_name": pending.profile_display_name,
            "assistant_application_title": pending.application_title,
            "assistant_context_text": (
                self._build_game_factoid_context_prompt(state, pending)
                if pending.commentary_type == "game_factoid_commentary"
                else state.runtime.build_context_prompt(
                    capabilities=self._serialize_capabilities(state),
                    pending_confirmations=self._serialize_pending(state),
                )
            ),
        }
        metadata["assistant_context_text"] += (
            "\n\nPROACTIVE COMMENTARY ROUTING\n"
            "- This is a proactive commentary turn.\n"
            "- If there is nothing useful to add, call remain_silent exactly once and end the turn."
        )
        metadata.update(pending.metadata)
        self._set_remain_silent_allowed(state, allowed=True)

        try:
            started = await self._start_conversation(
                client_uid, prompt_text, metadata
            )
        except Exception as exc:
            logger.warning(
                "Failed to start proactive assistant commentary for "
                f"{client_uid}: {exc}"
            )
            return False
        if not started:
            return False

        self._mark_proactive_commentary_started(
            state,
            pending.commentary_type,
            pending.profile_id,
        )
        state.pending_proactive_commentary = None
        return True

    async def _handle_game_factoid_context_tool(
        self,
        client_uid: str,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        state = self.get_client_state(client_uid)
        preferred_profile_id = str(tool_input.get("profile_id", "")).strip() or None
        raw_limit = tool_input.get("limit")
        limit = 3
        if raw_limit is not None:
            limit = max(1, min(int(raw_limit), 5))
        context = self._resolve_active_game_context(
            client_uid,
            state,
            preferred_profile_id=preferred_profile_id,
            allow_inactive_profile=bool(preferred_profile_id),
        )
        if context is None:
            return _tool_text_result(
                {
                    "status": "no_game",
                    "message": "No active game profile is available for factoid research.",
                }
            )

        now = _utcnow()
        factoids = self._sorted_game_factoids(state, context.profile_id)
        suggested = factoids[0] if factoids else None
        if suggested is not None:
            suggested.last_selected_at = now
            suggested.selection_count += 1
        payload = {
            "status": "ok",
            "profile_id": context.profile_id,
            "game_name": context.profile_display_name,
            "application_title": context.application_title,
            "cached_factoid_count": len(factoids),
            "research_needed": self._game_factoid_research_needed(
                state,
                context.profile_id,
                now,
            ),
            "last_researched_at": (
                _isoformat(state.game_fact_last_researched_at[context.profile_id])
                if context.profile_id in state.game_fact_last_researched_at
                else None
            ),
            "suggested_factoid": (
                self._serialize_game_factoid(suggested) if suggested else None
            ),
            "factoids": [
                self._serialize_game_factoid(record)
                for record in factoids[:limit]
            ],
        }
        return _tool_text_result(payload)

    async def _handle_remember_game_factoid_tool(
        self,
        client_uid: str,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        state = self.get_client_state(client_uid)
        preferred_profile_id = str(tool_input.get("profile_id", "")).strip() or None
        context = self._resolve_active_game_context(
            client_uid,
            state,
            preferred_profile_id=preferred_profile_id,
            allow_inactive_profile=bool(preferred_profile_id),
        )
        if context is None:
            return _tool_text_result(
                {
                    "status": "no_game",
                    "message": "No active or selected game profile is available for storing a factoid.",
                },
                is_error=True,
            )

        raw_fact_text = tool_input.get("fact_text")
        fact_text = _sanitize_game_factoid_text(
            "" if raw_fact_text is None else str(raw_fact_text)
        )
        if _looks_spoilery_factoid(fact_text):
            return _tool_text_result(
                {
                    "status": "rejected",
                    "message": "That factoid looks spoilery. Store only spoiler-safe trivia.",
                },
                is_error=True,
            )

        source_name = _sanitize_short_optional_text(
            tool_input.get("source_name"),
            limit=GAME_FACTOID_MAX_SOURCE_LENGTH,
        )
        source_url = _sanitize_short_optional_text(
            tool_input.get("source_url"),
            limit=200,
        )
        category = _sanitize_short_optional_text(
            tool_input.get("category"),
            limit=GAME_FACTOID_MAX_CATEGORY_LENGTH,
        )
        tags = _sanitize_factoid_tags(tool_input.get("tags"))

        factoids = state.game_factoids.setdefault(context.profile_id, [])
        normalized_text = _normalize_voice_phrase(fact_text)
        now = _utcnow()
        for record in factoids:
            if _normalize_voice_phrase(record.text) != normalized_text:
                continue
            if source_name and not record.source_name:
                record.source_name = source_name
            if source_url and not record.source_url:
                record.source_url = source_url
            if category and not record.category:
                record.category = category
            if tags:
                merged = list(dict.fromkeys([*record.tags, *tags]))
                record.tags = merged[:GAME_FACTOID_MAX_TAGS]
            state.game_fact_last_researched_at[context.profile_id] = now
            return _tool_text_result(
                {
                    "status": "existing",
                    "profile_id": context.profile_id,
                    "game_name": context.profile_display_name,
                    "factoid": self._serialize_game_factoid(record),
                }
            )

        record = GameFactoidRecord(
            factoid_id=str(uuid4()),
            profile_id=context.profile_id,
            profile_display_name=context.profile_display_name,
            application_title=context.application_title,
            text=fact_text,
            source_name=source_name,
            source_url=source_url,
            category=category,
            tags=tags,
            created_at=now,
        )
        factoids.append(record)
        factoids.sort(key=lambda item: item.created_at, reverse=True)
        del factoids[GAME_FACTOID_MAX_PER_PROFILE:]
        state.game_fact_last_researched_at[context.profile_id] = now
        return _tool_text_result(
            {
                "status": "stored",
                "profile_id": context.profile_id,
                "game_name": context.profile_display_name,
                "factoid": self._serialize_game_factoid(record),
            }
        )

    async def _handle_list_game_factoids_tool(
        self,
        client_uid: str,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        state = self.get_client_state(client_uid)
        preferred_profile_id = str(tool_input.get("profile_id", "")).strip() or None
        raw_limit = tool_input.get("limit")
        limit = 10
        if raw_limit is not None:
            limit = max(1, min(int(raw_limit), 20))
        context = self._resolve_active_game_context(
            client_uid,
            state,
            preferred_profile_id=preferred_profile_id,
            allow_inactive_profile=bool(preferred_profile_id),
        )
        if context is None:
            return _tool_text_result(
                {
                    "status": "no_game",
                    "message": "No active or selected game profile is available.",
                }
            )

        factoids = sorted(
            state.game_factoids.get(context.profile_id, []),
            key=lambda item: item.created_at,
            reverse=True,
        )
        payload = {
            "status": "ok",
            "profile_id": context.profile_id,
            "game_name": context.profile_display_name,
            "cached_factoid_count": len(factoids),
            "last_researched_at": (
                _isoformat(state.game_fact_last_researched_at[context.profile_id])
                if context.profile_id in state.game_fact_last_researched_at
                else None
            ),
            "factoids": [
                self._serialize_game_factoid(record)
                for record in factoids[:limit]
            ],
        }
        return _tool_text_result(payload)

    async def _handle_list_tool(
        self, client_uid: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        del tool_input
        state = self.get_client_state(client_uid)
        self._expire_pending(state)
        capabilities = [
            capability
            for capability in self._serialize_capabilities(state)
            if capability["llm_access"] != "none"
        ]
        payload: dict[str, Any] = {
            "status": "ok" if capabilities else "empty",
            "active_profile_id": state.active_profile_id,
            "capabilities": capabilities,
            "pending_confirmations": self._serialize_pending(state),
            "recent_decisions": self._serialize_recent_decisions(state),
        }
        if not capabilities:
            payload["message"] = (
                "No valid automation capabilities are currently available."
            )
        return _tool_text_result(payload)

    async def _handle_suggest_tool(
        self, client_uid: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            profile_id = str(tool_input.get("profile_id", "")).strip()
            command_id = str(tool_input["command_id"]).strip()
            reason = _sanitize_reason(str(tool_input["concise_reason"]))
        except Exception as exc:
            return _tool_text_result(
                {"status": "error", "message": f"Invalid suggestion request: {exc}"},
                is_error=True,
            )

        state = self.get_client_state(client_uid)
        self._expire_pending(state)
        capability, llm_access, blocked_reason = self._validate_named_command(
            state,
            profile_id,
            command_id,
        )
        if blocked_reason:
            return await self._blocked_tool_result(
                client_uid,
                state,
                capability,
                reason,
                blocked_reason,
                "suggestion_blocked",
            )

        if llm_access == "none":
            return await self._blocked_tool_result(
                client_uid,
                state,
                capability,
                reason,
                "The assistant is not allowed to use that command.",
                "suggestion_blocked",
            )

        if not state.settings.include_command_suggestions_in_speech:
            return await self._blocked_tool_result(
                client_uid,
                state,
                capability,
                reason,
                "Command suggestions in speech are disabled.",
                "suggestion_blocked",
            )

        if self._is_repetitive_suggestion(state, capability.command_id):
            return await self._blocked_tool_result(
                client_uid,
                state,
                capability,
                reason,
                "That command was already suggested recently.",
                "suggestion_blocked",
            )

        self._record_decision(
            state,
            decision_type="suggested",
            profile_id=capability.profile_id,
            command_id=capability.command_id,
            command_label=capability.label,
            reason=reason,
            risk=capability.risk,
        )
        await self._emit_state(client_uid)
        return _tool_text_result(
            {
                "status": "suggested",
                "profile_id": capability.profile_id,
                "command_id": capability.command_id,
                "command_label": capability.label,
                "reason": reason,
                "llm_access": llm_access,
            }
        )

    async def _handle_request_tool(
        self,
        client_uid: str,
        tool_input: dict[str, Any],
        *,
        request_source: str = "vtuber",
    ) -> dict[str, Any]:
        try:
            profile_id = str(tool_input.get("profile_id", "")).strip()
            command_id = str(tool_input["command_id"]).strip()
            reason = _sanitize_reason(str(tool_input["concise_reason"]))
        except Exception as exc:
            return _tool_text_result(
                {"status": "error", "message": f"Invalid automation request: {exc}"},
                is_error=True,
            )

        state = self.get_client_state(client_uid)
        self._expire_pending(state)
        capability, llm_access, blocked_reason = self._validate_named_command(
            state,
            profile_id,
            command_id,
            source=request_source,
        )
        if blocked_reason:
            return await self._blocked_tool_result(
                client_uid,
                state,
                capability,
                reason,
                blocked_reason,
                "request_blocked",
            )

        if llm_access == "suggest":
            return await self._blocked_tool_result(
                client_uid,
                state,
                capability,
                reason,
                "That command may be suggested but not executed.",
                "request_blocked",
            )

        if llm_access == "none":
            return await self._blocked_tool_result(
                client_uid,
                state,
                capability,
                reason,
                "The assistant is not allowed to use that command.",
                "request_blocked",
            )

        if self._has_active_request_for_command(
            state,
            capability.profile_id,
            capability.command_id,
        ):
            return await self._blocked_tool_result(
                client_uid,
                state,
                capability,
                reason,
                "That command already has a pending or active assistant request.",
                "request_blocked",
            )

        if self._was_command_handled_for_current_player_request(
            state,
            capability.profile_id,
            capability.command_id,
        ):
            return await self._blocked_tool_result(
                client_uid,
                state,
                capability,
                reason,
                "That command was already handled for the current player request.",
                "request_blocked",
            )

        if llm_access == "autonomous":
            try:
                started_payload = await self._start_execution(
                    client_uid,
                    state,
                    capability,
                    reason=reason,
                    source=request_source,
                )
            except ValueError as exc:
                return await self._blocked_tool_result(
                    client_uid,
                    state,
                    capability,
                    reason,
                    str(exc),
                    "request_blocked",
                )
            await self._emit_state(client_uid)
            return _tool_text_result(started_payload)

        try:
            pending_result = self._create_pending_confirmation(
                state,
                capability,
                reason,
                source=request_source,
            )
        except ValueError as exc:
            return await self._blocked_tool_result(
                client_uid,
                state,
                capability,
                reason,
                str(exc),
                "request_blocked",
            )
        await self._emit_state(client_uid)
        return _tool_text_result(pending_result)

    async def _handle_cancel_tool(
        self, client_uid: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        state = self.get_client_state(client_uid)
        self._expire_pending(state)
        request_id = str(tool_input.get("request_id", "")).strip() or None

        if request_id and request_id in state.pending_confirmations:
            message = AutomationConfirmationResponseMessage(
                request_id=request_id,
                action="reject",
            )
            await self.handle_confirmation_response(client_uid, message)
            return _tool_text_result(
                {
                    "status": "cancelled_pending_confirmation",
                    "request_id": request_id,
                }
            )

        if request_id:
            cancelled = await self._transport.request_cancellation(
                client_uid, request_id
            )
            if cancelled:
                self._record_decision(
                    state,
                    decision_type="cancelled",
                    profile_id=None,
                    command_id=None,
                    command_label=None,
                    reason="Cancellation requested by the VTuber.",
                    risk=None,
                    request_id=request_id,
                )
                await self._emit_state(client_uid)
                return _tool_text_result(
                    {
                        "status": "cancel_requested",
                        "request_id": request_id,
                    }
                )

        latest_pending = self._get_latest_pending(state)
        if latest_pending:
            message = AutomationConfirmationResponseMessage(
                request_id=latest_pending.request_id,
                action="reject",
            )
            await self.handle_confirmation_response(client_uid, message)
            return _tool_text_result(
                {
                    "status": "cancelled_pending_confirmation",
                    "request_id": latest_pending.request_id,
                }
            )

        return _tool_text_result(
            {
                "status": "noop",
                "message": "No matching pending confirmation or running automation request was found.",
            }
        )

    async def _handle_remain_silent_tool(
        self, client_uid: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        del tool_input
        state = self.get_client_state(client_uid)
        if not state.allow_remain_silent_tool:
            blocked_reason = (
                "remain_silent is only allowed during proactive commentary turns. "
                "Reply normally to direct user requests instead."
            )
            state.last_blocked_reason = blocked_reason
            await self._emit_state(client_uid)
            return _tool_text_result(
                {
                    "status": "blocked",
                    "blocked_reason": blocked_reason,
                }
            )
        self._record_decision(
            state,
            decision_type="silent",
            profile_id=None,
            command_id=None,
            command_label=None,
            reason="The VTuber chose to remain silent.",
            risk=None,
        )
        state.runtime.mark_remained_silent("The explicit remain_silent tool was used.")
        await self._emit_state(client_uid)
        return _tool_text_result(
            {"status": "silent"},
            internal_metadata={"stop_after_tool": True},
        )

    async def _start_execution(
        self,
        client_uid: str,
        state: AssistantClientState,
        capability: AutomationCapability,
        *,
        reason: str,
        request_id: Optional[str] = None,
        source: str = "vtuber",
    ) -> dict[str, Any]:
        request_id = request_id or str(uuid4())
        message = await self._transport.request_execution(
            client_uid,
            capability.profile_id,
            capability.command_id,
            source=source,
            variables={"reason": reason},
            request_id=request_id,
        )
        state.request_records[message.request_id] = RequestRecord(
            request_id=message.request_id,
            profile_id=capability.profile_id,
            command_id=capability.command_id,
            command_label=capability.label,
            reason=reason,
            risk=capability.risk,
            status="started",
            created_at=_utcnow(),
            source=source,
        )
        self._record_decision(
            state,
            decision_type="started",
            profile_id=capability.profile_id,
            command_id=capability.command_id,
            command_label=capability.label,
            reason=reason,
            risk=capability.risk,
            request_id=message.request_id,
        )
        state.runtime.record_automation_request(
            command_label=capability.label,
            reason=reason,
            request_id=message.request_id,
            confirmation_required=False,
        )
        state.runtime.mark_acted(
            summary=f"Started {capability.label}.",
            event_type="automation.requested",
        )
        return {
            "status": "started",
            "request_id": message.request_id,
            "profile_id": capability.profile_id,
            "command_id": capability.command_id,
            "command_label": capability.label,
            "reason": reason,
        }

    def _create_pending_confirmation(
        self,
        state: AssistantClientState,
        capability: AutomationCapability,
        reason: str,
        *,
        source: str = "vtuber",
    ) -> dict[str, Any]:
        if len(state.pending_confirmations) >= state.settings.max_pending_confirmations:
            raise ValueError(
                "The maximum number of pending confirmations has been reached."
            )

        created_at = _utcnow()
        expires_at = created_at + timedelta(
            milliseconds=state.settings.confirmation_timeout_ms
        )
        request_id = str(uuid4())
        pending = PendingConfirmation(
            request_id=request_id,
            profile_id=capability.profile_id,
            command_id=capability.command_id,
            command_label=capability.label,
            concise_reason=reason,
            risk=capability.risk,
            created_at=created_at,
            expires_at=expires_at,
            source=source,
        )
        state.pending_confirmations[pending.request_id] = pending
        state.request_records[pending.request_id] = RequestRecord(
            request_id=pending.request_id,
            profile_id=capability.profile_id,
            command_id=capability.command_id,
            command_label=capability.label,
            reason=reason,
            risk=capability.risk,
            status="confirmation_pending",
            created_at=created_at,
            source=source,
        )
        self._record_decision(
            state,
            decision_type="confirmation_pending",
            profile_id=capability.profile_id,
            command_id=capability.command_id,
            command_label=capability.label,
            reason=reason,
            risk=capability.risk,
            request_id=pending.request_id,
        )
        state.runtime.set_pending_confirmation_count(len(state.pending_confirmations))
        state.runtime.record_automation_request(
            command_label=capability.label,
            reason=reason,
            request_id=pending.request_id,
            confirmation_required=True,
        )
        return {
            "status": "pending_confirmation",
            "request_id": pending.request_id,
            "profile_id": capability.profile_id,
            "command_id": capability.command_id,
            "command_label": capability.label,
            "reason": reason,
            "risk": capability.risk,
            "expires_at": _isoformat(expires_at),
            "source": source,
        }

    async def _handle_result_message(
        self,
        client_uid: str,
        state: AssistantClientState,
        message: AutomationResultMessage,
    ) -> bool:
        self._expire_pending(state)
        tracking = state.request_records.get(message.request_id)
        if tracking is None:
            return False

        tracking.status = message.status
        outcome_type = message.outcome.result_type if message.outcome else None
        safe_error = _sanitize_error_text(message.error)
        safe_outcome_reason = _sanitize_error_text(
            message.outcome.reason if message.outcome else None
        )
        normalized_status = message.status
        if outcome_type in {
            "command_sent_unconfirmed",
            "blocked_by_precondition",
            "blocked_by_policy",
            "unsupported",
            "partially_executed",
            "timed_out",
            "invalid_dependency_graph",
            "unresolved_dependency",
        }:
            normalized_status = "blocked"
        elif outcome_type == "already_satisfied":
            normalized_status = "completed"
        elif outcome_type == "failed":
            normalized_status = "failed"
        elif outcome_type == "cancelled":
            normalized_status = "cancelled"

        if normalized_status == "completed":
            state.last_blocked_reason = None
        elif normalized_status == "blocked":
            state.last_blocked_reason = (
                safe_outcome_reason or safe_error or "The automation request was blocked."
            )
        elif normalized_status == "failed":
            state.last_blocked_reason = (
                safe_outcome_reason or safe_error or "The automation request failed."
            )

        self._record_decision(
            state,
            decision_type=normalized_status,
            profile_id=tracking.profile_id,
            command_id=tracking.command_id,
            command_label=tracking.command_label,
            reason=tracking.reason,
            risk=tracking.risk,
            blocked_reason=safe_outcome_reason or safe_error
            if normalized_status in {"blocked", "failed"}
            else None,
            request_id=tracking.request_id,
        )
        if normalized_status in {"completed", "failed", "blocked", "cancelled"}:
            state.runtime.record_automation_result(
                status=normalized_status,
                command_label=tracking.command_label,
                request_id=tracking.request_id,
                error=safe_outcome_reason or safe_error,
            )
        await self._maybe_acknowledge_result(client_uid, state, tracking, message)
        return True

    async def _maybe_acknowledge_result(
        self,
        client_uid: str,
        state: AssistantClientState,
        tracking: RequestRecord,
        message: AutomationResultMessage,
    ) -> None:
        outcome = message.outcome
        outcome_type = outcome.result_type if outcome else None
        outcome_reason = _sanitize_error_text(outcome.reason if outcome else None)
        capability = state.capabilities.get(tracking.command_id)

        if outcome_type == "already_satisfied":
            if tracking.source == "player_voice":
                if state.settings.result_acknowledgements_enabled and outcome_reason:
                    await self._speak_to_client(client_uid, outcome_reason)
            elif state.settings.result_acknowledgements_enabled and outcome_reason:
                await self._speak_to_client(client_uid, outcome_reason)
            return

        if outcome_type in {
            "command_sent_unconfirmed",
            "unsupported",
            "partially_executed",
            "timed_out",
            "invalid_dependency_graph",
            "unresolved_dependency",
        }:
            acknowledgement = outcome_reason or "The action was not fully confirmed."
            if tracking.source == "player_voice":
                if state.settings.result_acknowledgements_enabled:
                    await self._speak_to_client(client_uid, acknowledgement)
            elif state.settings.result_acknowledgements_enabled:
                await self._speak_to_client(client_uid, acknowledgement)
            return

        if outcome_type in {"blocked_by_precondition", "blocked_by_policy"}:
            blocked_reason = outcome_reason or "It is blocked right now."
            if tracking.source == "player_voice":
                await self._maybe_speak_voice_command_stage(
                    client_uid,
                    state,
                    stage="blocked",
                    capability=capability,
                    blocked_reason=blocked_reason,
                )
            elif state.settings.announce_blocked_command_requests:
                await self._speak_to_client(
                    client_uid,
                    f"I couldn't run {tracking.command_label}: {blocked_reason}",
                )
            return

        if message.status == "completed":
            if tracking.source == "player_voice":
                if state.settings.result_acknowledgements_enabled:
                    await self._speak_to_client(
                        client_uid,
                        outcome_reason or f"{tracking.command_label} finished.",
                    )
            elif state.settings.result_acknowledgements_enabled:
                await self._speak_to_client(
                    client_uid,
                    outcome_reason or f"{tracking.command_label} finished.",
                )
            return

        if (
            message.status == "failed"
        ):
            if tracking.source == "player_voice":
                await self._maybe_speak_voice_command_stage(
                    client_uid,
                    state,
                    stage="failed",
                    capability=capability,
                )
            elif state.settings.result_acknowledgements_enabled:
                await self._speak_to_client(
                    client_uid,
                    f"I couldn't run {tracking.command_label}.",
                )
            return

        if message.status == "blocked":
            blocked_reason = (
                outcome_reason
                or _sanitize_error_text(message.error)
                or "It is blocked right now."
            )
            if tracking.source == "player_voice":
                await self._maybe_speak_voice_command_stage(
                    client_uid,
                    state,
                    stage="blocked",
                    capability=capability,
                    blocked_reason=blocked_reason,
                )
            elif state.settings.announce_blocked_command_requests:
                await self._speak_to_client(
                    client_uid,
                    f"I couldn't run {tracking.command_label}: {blocked_reason}",
                )

    def _validate_named_command(
        self,
        state: AssistantClientState,
        profile_id: str,
        command_id: str,
        *,
        source: str = "vtuber",
    ) -> tuple[Optional[AutomationCapability], AutomationLlmAccess, Optional[str]]:
        capability, resolution_error = self._resolve_named_command(
            state,
            profile_id,
            command_id,
        )
        if capability is None:
            return (
                None,
                "none",
                resolution_error
                or "That command is not in the current automation capability catalogue.",
            )

        llm_access, blocked_reason = self._derive_access(
            state,
            capability,
            source=source,
        )
        return capability, llm_access, blocked_reason

    def _resolve_named_command(
        self,
        state: AssistantClientState,
        profile_id: str,
        command_reference: str,
    ) -> tuple[Optional[AutomationCapability], Optional[str]]:
        requested_profile_id = profile_id.strip() or state.active_profile_id or ""
        direct = state.capabilities.get(command_reference)
        if direct is not None and (
            not requested_profile_id or direct.profile_id == requested_profile_id
        ):
            return direct, None

        scoped_capabilities = [
            capability
            for capability in state.capabilities.values()
            if not requested_profile_id or capability.profile_id == requested_profile_id
        ]
        if not scoped_capabilities:
            if requested_profile_id:
                return (
                    None,
                    f'No automation capabilities are currently available for profile "{requested_profile_id}".',
                )
            return (
                None,
                "That command is not in the current automation capability catalogue.",
            )

        lowered_reference = command_reference.strip().lower()
        casefold_matches = [
            capability
            for capability in scoped_capabilities
            if capability.command_id.lower() == lowered_reference
        ]
        if len(casefold_matches) == 1:
            return casefold_matches[0], None
        if len(casefold_matches) > 1:
            return (
                None,
                "That command name matches multiple automation commands. Use the exact command_id.",
            )

        normalized_reference = _normalize_command_reference(command_reference)
        if not normalized_reference:
            return (
                None,
                "That command is not in the current automation capability catalogue.",
            )

        normalized_matches: list[AutomationCapability] = []
        seen: set[str] = set()
        for capability in scoped_capabilities:
            phrases = [capability.command_id, capability.label, *capability.aliases]
            if any(
                _normalize_command_reference(phrase) == normalized_reference
                for phrase in phrases
            ):
                if capability.command_id in seen:
                    continue
                seen.add(capability.command_id)
                normalized_matches.append(capability)

        if len(normalized_matches) == 1:
            return normalized_matches[0], None
        if len(normalized_matches) > 1:
            labels = ", ".join(capability.label for capability in normalized_matches[:3])
            return (
                None,
                "That command name is ambiguous in the current automation capability catalogue."
                + (f" Matches: {labels}." if labels else ""),
            )
        return (
            None,
            "That command is not in the current automation capability catalogue.",
        )

    def _derive_access(
        self,
        state: AssistantClientState,
        capability: AutomationCapability,
        *,
        source: str = "vtuber",
    ) -> tuple[AutomationLlmAccess, Optional[str]]:
        advice = self._get_telemetry_advice(state, capability)
        automation_state = self._transport.get_client_state(
            self._find_client_uid(state)
        )
        if automation_state.status == "disconnected":
            return "none", "The automation transport is disconnected."
        if automation_state.emergency_stopped:
            return "none", "Automation is emergency stopped."
        if not state.active_profile_id:
            return "none", "No active automation profile is selected."
        if capability.profile_id != state.active_profile_id:
            return "none", "That command is not on the active automation profile."
        if not capability.enabled:
            return "none", "That command is disabled."
        if source == "vtuber" and not state.settings.llm_automation_enabled:
            return "none", "LLM automation is disabled."
        if not capability.available:
            return (
                "none",
                capability.blocked_reason
                or "That command is not currently available.",
            )
        if advice and advice.available_override is False:
            return (
                "none",
                advice.blocked_reason or "That command is not advisable right now.",
            )
        if capability.cooldown_remaining_ms > 0:
            return "none", "That command is still on cooldown."
        if not self._allows_assistant_execution(capability, source=source):
            if source == "player_voice":
                return "none", "That command does not allow player voice execution."
            return "none", "That command does not allow assistant execution."

        if capability.autonomy_policy == "disabled":
            if source == "player_voice":
                return "none", "That command is disabled for player voice automation."
            return "none", "That command is disabled for assistant automation."
        if capability.autonomy_policy == "suggest_only":
            return "suggest", None
        if capability.autonomy_policy == "confirmation_required":
            return "request_confirmation", None

        if capability.risk == "harmless":
            if state.settings.allow_autonomous_harmless_commands:
                return "autonomous", None
            return "request_confirmation", None
        if capability.risk == "low":
            if state.settings.allow_autonomous_low_risk_commands:
                return "autonomous", None
            return "request_confirmation", None
        return "request_confirmation", None

    def _allows_assistant_execution(
        self,
        capability: AutomationCapability,
        *,
        source: str = "vtuber",
    ) -> bool:
        allowed_sources = set(capability.allowed_trigger_sources)
        if source == "player_voice":
            return "player_voice" in allowed_sources
        return "vtuber" in allowed_sources or "player_voice" in allowed_sources

    def _get_telemetry_advice(
        self,
        state: AssistantClientState,
        capability: AutomationCapability,
    ) -> Optional[TelemetryCapabilityAdvice]:
        if state.telemetry is None:
            return None
        advice_map = state.telemetry.build_capability_advice(
            [capability.model_dump(mode="json")]
        )
        return advice_map.get(capability.command_id)

    def _find_client_uid(self, target_state: AssistantClientState) -> str:
        for client_uid, state in self._client_states.items():
            if state is target_state:
                return client_uid
        raise KeyError("Assistant client state is not registered")

    def _record_decision(
        self,
        state: AssistantClientState,
        *,
        decision_type: AutomationDecisionType,
        profile_id: Optional[str],
        command_id: Optional[str],
        command_label: Optional[str],
        reason: Optional[str],
        risk: Optional[str],
        blocked_reason: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        state.recent_decisions.insert(
            0,
            DecisionRecord(
                decision_id=str(uuid4()),
                decision_type=decision_type,
                profile_id=profile_id,
                command_id=command_id,
                command_label=command_label,
                reason=reason,
                risk=risk,
                blocked_reason=blocked_reason,
                request_id=request_id,
                created_at=_utcnow(),
            ),
        )
        if len(state.recent_decisions) > RECENT_DECISION_LIMIT:
            state.recent_decisions = state.recent_decisions[:RECENT_DECISION_LIMIT]

    def _is_repetitive_suggestion(
        self,
        state: AssistantClientState,
        command_id: str,
    ) -> bool:
        for decision in state.recent_decisions:
            if decision.command_id != command_id:
                continue
            if decision.decision_type not in {
                "suggested",
                "confirmation_pending",
                "started",
            }:
                continue
            if (
                _utcnow() - decision.created_at
            ).total_seconds() * 1000 < REPEAT_SUGGESTION_COOLDOWN_MS:
                return True
        return False

    def _has_active_request_for_command(
        self,
        state: AssistantClientState,
        profile_id: str,
        command_id: str,
    ) -> bool:
        if any(
            pending.profile_id == profile_id and pending.command_id == command_id
            for pending in state.pending_confirmations.values()
        ):
            return True

        for record in state.request_records.values():
            if (
                record.profile_id == profile_id
                and record.command_id == command_id
                and record.status in {"confirmation_pending", "started"}
            ):
                return True
        return False

    def _was_command_handled_for_current_player_request(
        self,
        state: AssistantClientState,
        profile_id: str,
        command_id: str,
    ) -> bool:
        request_started_at = state.runtime.state.last_player_request_at
        if request_started_at is None:
            return False

        handled_decision_types: set[AutomationDecisionType] = {
            "request_blocked",
            "confirmation_pending",
            "confirmed",
            "rejected",
            "expired",
            "started",
            "completed",
            "failed",
            "blocked",
            "cancelled",
        }
        for decision in state.recent_decisions:
            if decision.profile_id != profile_id or decision.command_id != command_id:
                continue
            if decision.decision_type not in handled_decision_types:
                continue
            if decision.created_at > request_started_at:
                return True
        return False

    def _expire_pending(self, state: AssistantClientState) -> bool:
        changed = self._expire_pending_ambiguity(state)
        now = _utcnow()
        expired = [
            request_id
            for request_id, pending in state.pending_confirmations.items()
            if pending.expires_at <= now
        ]
        if not expired:
            return changed

        for request_id in expired:
            pending = state.pending_confirmations.pop(request_id)
            tracking = state.request_records.get(request_id)
            if tracking:
                tracking.status = "expired"
            self._record_decision(
                state,
                decision_type="expired",
                profile_id=pending.profile_id,
                command_id=pending.command_id,
                command_label=pending.command_label,
                reason=pending.concise_reason,
                risk=pending.risk,
                request_id=request_id,
            )
        return True

    def _cancel_all_pending(
        self,
        state: AssistantClientState,
        *,
        blocked_reason: str,
        decision_type: AutomationDecisionType,
    ) -> bool:
        if not state.pending_confirmations:
            return False

        for request_id, pending in list(state.pending_confirmations.items()):
            state.pending_confirmations.pop(request_id, None)
            tracking = state.request_records.get(request_id)
            if tracking:
                tracking.status = decision_type
            self._record_decision(
                state,
                decision_type=decision_type,
                profile_id=pending.profile_id,
                command_id=pending.command_id,
                command_label=pending.command_label,
                reason=pending.concise_reason,
                risk=pending.risk,
                blocked_reason=blocked_reason,
                request_id=request_id,
            )
        state.last_blocked_reason = blocked_reason
        return True

    def _reconcile_pending_against_capabilities(
        self, state: AssistantClientState
    ) -> None:
        for request_id, pending in list(state.pending_confirmations.items()):
            capability = state.capabilities.get(pending.command_id)
            if capability is None or capability.profile_id != pending.profile_id:
                state.pending_confirmations.pop(request_id, None)
                tracking = state.request_records.get(request_id)
                if tracking:
                    tracking.status = "blocked"
                self._record_decision(
                    state,
                    decision_type="blocked",
                    profile_id=pending.profile_id,
                    command_id=pending.command_id,
                    command_label=pending.command_label,
                    reason=pending.concise_reason,
                    risk=pending.risk,
                    blocked_reason="The active profile changed and the request is no longer available.",
                    request_id=request_id,
                )
                continue

            llm_access, blocked_reason = self._derive_access(
                state,
                capability,
                source=pending.source,
            )
            if llm_access == "none":
                state.pending_confirmations.pop(request_id, None)
                tracking = state.request_records.get(request_id)
                if tracking:
                    tracking.status = "blocked"
                self._record_decision(
                    state,
                    decision_type="blocked",
                    profile_id=pending.profile_id,
                    command_id=pending.command_id,
                    command_label=pending.command_label,
                    reason=pending.concise_reason,
                    risk=pending.risk,
                    blocked_reason=blocked_reason
                    or "The pending assistant request is no longer available.",
                    request_id=request_id,
                )

    def _get_latest_pending(
        self, state: AssistantClientState
    ) -> Optional[PendingConfirmation]:
        if not state.pending_confirmations:
            return None
        return max(
            state.pending_confirmations.values(),
            key=lambda pending: pending.created_at,
        )

    async def _emit_state(self, client_uid: str) -> None:
        state = self.get_client_state(client_uid)
        self._expire_pending_ambiguity(state)
        runtime_payload = state.runtime.build_payload()
        payload = {
            "type": "automation/assistant-state",
            "active_profile_id": state.active_profile_id,
            "capability_revision": state.capability_revision,
            "settings": state.settings.model_dump(mode="json"),
            "capabilities": self._serialize_capabilities(state),
            "pending_confirmations": self._serialize_pending(state),
            "recent_decisions": self._serialize_recent_decisions(state),
            "pending_ambiguity": self._serialize_pending_ambiguity(state),
            "resolver_activity": self._serialize_resolver_activity(state),
            "last_blocked_reason": state.last_blocked_reason,
            **runtime_payload,
        }
        await self._send_to_client(client_uid, payload)

    async def send_state_snapshot(self, client_uid: str) -> None:
        await self._emit_state(client_uid)

    async def _handle_telemetry_event(
        self,
        client_uid: str,
        event: TelemetryEvent,
    ) -> None:
        state = self.get_client_state(client_uid)
        state.runtime.record_event(
            event_type=event.event_type,
            source="telemetry",
            summary=event.summary,
            novelty_key=event.novelty_key,
            confidence=event.confidence,
            data=event.data,
            importance=event.importance,
            urgency=event.urgency,
            expiry_ms=event.expiry_ms,
            cooldown_ms=event.cooldown_ms,
            conversational=event.conversational,
            high_priority=event.high_priority,
        )
        if state.telemetry is not None:
            state.runtime.set_telemetry_snapshot(state.telemetry.snapshot_payload())
        await self._emit_state(client_uid)

    async def _handle_telemetry_state_change(self, client_uid: str) -> None:
        state = self.get_client_state(client_uid)
        if state.telemetry is not None:
            state.runtime.set_telemetry_snapshot(state.telemetry.snapshot_payload())
        await self._emit_state(client_uid)

    async def _sync_telemetry_context(
        self,
        client_uid: str,
        state: AssistantClientState,
    ) -> None:
        if state.telemetry is None:
            return
        transport_state = self._transport.get_client_state(client_uid)
        effective_profile_id = (
            state.runtime.state.effective_profile_id or state.active_profile_id
        )
        process_names: list[str] = []
        profile_display_name: Optional[str] = None
        if effective_profile_id:
            profile = transport_state.profiles.get(effective_profile_id)
            if profile:
                process_names = list(profile.process_names)
                profile_display_name = profile.display_name
        detected_window_title = (
            state.runtime.state.active_application.window_title
            if state.runtime.state.active_application
            else None
        )
        detected_process_name = (
            state.runtime.state.active_application.process_name
            if state.runtime.state.active_application
            else None
        )
        await state.telemetry.update_context(
            profile_id=effective_profile_id,
            profile_display_name=profile_display_name,
            profile_process_names=process_names,
            detected_process_name=detected_process_name,
            detected_window_title=detected_window_title,
        )
        state.runtime.set_telemetry_snapshot(state.telemetry.snapshot_payload())

    def _serialize_capabilities(
        self, state: AssistantClientState
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for capability in sorted(
            state.capabilities.values(),
            key=lambda item: (item.category, item.label.lower()),
        ):
            advice = self._get_telemetry_advice(state, capability)
            llm_access, blocked_reason = self._derive_access(state, capability)
            available = capability.available
            if advice and advice.available_override is False:
                available = False
            serialized.append(
                {
                    **capability.model_dump(mode="json"),
                    "available": available,
                    "llm_access": llm_access,
                    "blocked_reason": blocked_reason,
                    "recommended": advice.recommended if advice else False,
                    "avoid_reason": advice.avoid_reason if advice else None,
                }
            )
        return serialized

    def _serialize_pending(self, state: AssistantClientState) -> list[dict[str, Any]]:
        return [
            {
                "request_id": pending.request_id,
                "profile_id": pending.profile_id,
                "command_id": pending.command_id,
                "command_label": pending.command_label,
                "concise_reason": pending.concise_reason,
                "risk": pending.risk,
                "created_at": _isoformat(pending.created_at),
                "expires_at": _isoformat(pending.expires_at),
                "source": pending.source,
            }
            for pending in sorted(
                state.pending_confirmations.values(),
                key=lambda item: item.created_at,
            )
        ]

    def _serialize_recent_decisions(
        self, state: AssistantClientState
    ) -> list[dict[str, Any]]:
        return [
            {
                "decision_id": decision.decision_id,
                "decision_type": decision.decision_type,
                "profile_id": decision.profile_id,
                "command_id": decision.command_id,
                "command_label": decision.command_label,
                "reason": decision.reason,
                "risk": decision.risk,
                "blocked_reason": decision.blocked_reason,
                "request_id": decision.request_id,
                "created_at": _isoformat(decision.created_at),
            }
            for decision in state.recent_decisions[:RECENT_DECISION_LIMIT]
        ]

    def _serialize_pending_ambiguity(
        self, state: AssistantClientState
    ) -> Optional[dict[str, Any]]:
        pending = state.pending_ambiguity
        if pending is None:
            return None
        return {
            "ambiguity_id": pending.ambiguity_id,
            "transcript": pending.transcript,
            "normalized_phrase": pending.normalized_phrase,
            "candidates": [
                {
                    "command_id": capability.command_id,
                    "label": capability.label,
                }
                for capability in (
                    state.capabilities.get(command_id)
                    for command_id in pending.candidate_command_ids
                )
                if capability is not None
            ],
            "expires_at": _isoformat(pending.expires_at),
        }

    def _serialize_resolver_activity(
        self, state: AssistantClientState
    ) -> list[dict[str, Any]]:
        return [
            {
                "activity_id": entry.activity_id,
                "transcript": entry.transcript,
                "normalized_phrase": entry.normalized_phrase,
                "activation_mode": entry.activation_mode,
                "activation_met": entry.activation_met,
                "result": entry.result,
                "matched_command_id": entry.matched_command_id,
                "matched_command_label": entry.matched_command_label,
                "candidate_command_labels": entry.candidate_command_labels,
                "blocked_reason": entry.blocked_reason,
                "request_id": entry.request_id,
                "fell_back_to_conversation": entry.fell_back_to_conversation,
                "created_at": _isoformat(entry.created_at),
            }
            for entry in state.resolver_activity[:RECENT_RESOLVER_ACTIVITY_LIMIT]
        ]

    async def _blocked_tool_result(
        self,
        client_uid: str,
        state: AssistantClientState,
        capability: Optional[AutomationCapability],
        reason: str,
        blocked_reason: str,
        decision_type: AutomationDecisionType,
    ) -> dict[str, Any]:
        profile_id = capability.profile_id if capability else None
        command_id = capability.command_id if capability else None
        command_label = capability.label if capability else None
        risk = capability.risk if capability else None
        state.last_blocked_reason = blocked_reason
        self._record_decision(
            state,
            decision_type=decision_type,
            profile_id=profile_id,
            command_id=command_id,
            command_label=command_label,
            reason=reason,
            risk=risk,
            blocked_reason=blocked_reason,
        )
        payload = {
            "status": "blocked",
            "profile_id": profile_id,
            "command_id": command_id,
            "command_label": command_label,
            "reason": reason,
            "blocked_reason": blocked_reason,
        }
        if command_label:
            state.runtime.record_automation_result(
                status="blocked",
                command_label=command_label,
                error=blocked_reason,
            )
        else:
            state.runtime.record_system_warning(blocked_reason, source="automation")
        await self._emit_state(client_uid)
        return _tool_text_result(payload, is_error=True)
