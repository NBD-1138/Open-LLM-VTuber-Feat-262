from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AutomationTriggerSource: TypeAlias = Literal[
    "manual",
    "hotkey",
    "player_voice",
    "vtuber",
    "twitch",
    "redemption",
    "game_event",
]

AutomationExecutionStatus: TypeAlias = Literal[
    "started",
    "completed",
    "cancelled",
    "blocked",
    "failed",
]

AutomationRuntimeStatus: TypeAlias = Literal[
    "ready",
    "executing",
    "cancelled",
    "emergency_stopped",
    "disconnected",
    "error",
]

AutomationAutonomyPolicy: TypeAlias = Literal[
    "disabled",
    "suggest_only",
    "confirmation_required",
    "autonomous",
]

AutomationRiskLevel: TypeAlias = Literal[
    "harmless",
    "low",
    "medium",
    "high",
    "critical",
]

AutomationLlmAccess: TypeAlias = Literal[
    "none",
    "suggest",
    "request_confirmation",
    "autonomous",
]

AssistantSpeechMode: TypeAlias = Literal["action", "exploration", "conversation"]

VoiceCommandActivationMode: TypeAlias = Literal[
    "disabled",
    "wake_phrase",
    "push_to_command",
    "always_listening",
]

VoiceCommandAcknowledgementMode: TypeAlias = Literal[
    "none",
    "fixed_original",
    "llm_in_character",
]

AutomationProfileSelectionMode: TypeAlias = Literal[
    "automatic", "manual", "disabled"
]

AutomationProfileSelectionSource: TypeAlias = Literal["automatic", "manual", "none"]

AutomationDecisionType: TypeAlias = Literal[
    "suggested",
    "request_blocked",
    "suggestion_blocked",
    "confirmation_pending",
    "confirmed",
    "rejected",
    "expired",
    "started",
    "completed",
    "failed",
    "blocked",
    "cancelled",
    "silent",
]

ConfirmationResponseSource: TypeAlias = Literal["manual", "player_voice"]

MAX_CAPABILITY_COUNT = 128
MAX_CAPABILITY_LABEL_LENGTH = 96
MAX_CAPABILITY_DESCRIPTION_LENGTH = 280
MAX_CAPABILITY_CATEGORY_LENGTH = 48
MAX_REASON_LENGTH = 160
MAX_PHRASE_COUNT = 8
MAX_PHRASE_LENGTH = 48
MAX_PENDING_CONFIRMATIONS = 5
MAX_ACTIVE_APP_PROCESS_NAME_LENGTH = 80
MAX_ACTIVE_APP_WINDOW_TITLE_LENGTH = 160


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AutomationCommandSummary(_StrictModel):
    command_id: str
    label: str
    enabled: bool = True

    @field_validator("command_id", "label")
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value


class AutomationProfileSummary(_StrictModel):
    profile_id: str
    display_name: str
    enabled: bool = True
    process_names: list[str] = Field(default_factory=list)
    commands: list[AutomationCommandSummary] = Field(default_factory=list)

    @field_validator("profile_id", "display_name")
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value

    @field_validator("commands")
    @classmethod
    def _validate_unique_command_ids(
        cls, commands: list[AutomationCommandSummary]
    ) -> list[AutomationCommandSummary]:
        command_ids = [command.command_id for command in commands]
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("Duplicate command IDs are not allowed")
        return commands

    @field_validator("process_names")
    @classmethod
    def _validate_process_names(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            cleaned = item.strip()
            if not cleaned:
                raise ValueError("process_names cannot contain empty values")
            normalized.append(cleaned)
        if len(normalized) != len(set(item.lower() for item in normalized)):
            raise ValueError("Duplicate process names are not allowed")
        return normalized


class AutomationExecuteMessage(_StrictModel):
    type: Literal["automation/execute"] = "automation/execute"
    request_id: str
    profile_id: str
    command_id: str
    source: AutomationTriggerSource = "manual"
    variables: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "profile_id", "command_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value


class AutomationResultMessage(_StrictModel):
    class TraceEntry(_StrictModel):
        command_id: str
        node_id: str
        kind: str
        status: Literal["executed", "skipped", "blocked", "dry_run", "unsupported"]
        summary: str

        @field_validator("command_id", "node_id", "kind", "summary")
        @classmethod
        def _validate_trace_strings(cls, value: str) -> str:
            value = value.strip()
            if not value:
                raise ValueError("Field cannot be empty")
            return value

    class Outcome(_StrictModel):
        command_id: str
        result_type: Literal[
            "completed",
            "already_satisfied",
            "command_sent_unconfirmed",
            "blocked_by_precondition",
            "blocked_by_policy",
            "unsupported",
            "partially_executed",
            "timed_out",
            "cancelled",
            "failed",
            "invalid_dependency_graph",
            "unresolved_dependency",
        ]
        reason: str
        physical_input_sent: bool = False
        telemetry_confirmed: bool = False
        skipped_actions: list[str] = Field(default_factory=list)
        blocking_actions: list[str] = Field(default_factory=list)
        unsupported_actions: list[str] = Field(default_factory=list)
        nested_command_trace: list[str] = Field(default_factory=list)
        duration_ms: int = 0
        final_observed_state: Optional[dict[str, Any]] = None
        trace: list["AutomationResultMessage.TraceEntry"] = Field(
            default_factory=list
        )
        semantic_capability_id: Optional[str] = None

        @field_validator("command_id", "reason")
        @classmethod
        def _validate_required_strings(cls, value: str) -> str:
            value = value.strip()
            if not value:
                raise ValueError("Field cannot be empty")
            return value

        @field_validator(
            "skipped_actions",
            "blocking_actions",
            "unsupported_actions",
            "nested_command_trace",
        )
        @classmethod
        def _validate_string_lists(cls, value: list[str]) -> list[str]:
            normalized: list[str] = []
            for item in value:
                cleaned = item.strip()
                if not cleaned:
                    raise ValueError("Lists cannot contain empty values")
                normalized.append(cleaned)
            return normalized

        @field_validator("duration_ms")
        @classmethod
        def _validate_outcome_duration(cls, value: int) -> int:
            if value < 0:
                raise ValueError("duration_ms must be non-negative")
            return value

    type: Literal["automation/result"] = "automation/result"
    request_id: str
    profile_id: str
    command_id: str
    status: AutomationExecutionStatus
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    outcome: Optional["AutomationResultMessage.Outcome"] = None

    @field_validator("request_id", "profile_id", "command_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value

    @field_validator("duration_ms")
    @classmethod
    def _validate_duration(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError("duration_ms must be non-negative")
        return value


class AutomationStatusMessage(_StrictModel):
    type: Literal["automation/status"] = "automation/status"
    status: AutomationRuntimeStatus
    emergency_stopped: bool = False
    active_profile_id: Optional[str] = None
    profiles: list[AutomationProfileSummary] = Field(default_factory=list)
    running_requests: list[str] = Field(default_factory=list)
    last_error: Optional[str] = None

    @field_validator("profiles")
    @classmethod
    def _validate_unique_profile_ids(
        cls, profiles: list[AutomationProfileSummary]
    ) -> list[AutomationProfileSummary]:
        profile_ids = [profile.profile_id for profile in profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("Duplicate profile IDs are not allowed")
        return profiles


class AutomationCancelMessage(_StrictModel):
    type: Literal["automation/cancel"] = "automation/cancel"
    request_id: str

    @field_validator("request_id")
    @classmethod
    def _validate_request_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("request_id cannot be empty")
        return value


class AutomationEmergencyStopMessage(_StrictModel):
    type: Literal["automation/emergency_stop"] = "automation/emergency_stop"
    emergency_stopped: bool
    reason: Optional[str] = None
    request_id: Optional[str] = None


class AutomationSpeakFixedMessage(_StrictModel):
    type: Literal["automation/speak-fixed"] = "automation/speak-fixed"
    request_id: str
    text: str
    interrupt_policy: Optional[str] = None

    @field_validator("request_id", "text")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value


class AutomationCapability(_StrictModel):
    profile_id: str
    command_id: str
    label: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    category: str
    enabled: bool = True
    available: bool = True
    risk: AutomationRiskLevel
    autonomy_policy: AutomationAutonomyPolicy
    allowed_trigger_sources: list[AutomationTriggerSource] = Field(default_factory=list)
    cooldown_remaining_ms: int = 0
    blocked_reason: Optional[str] = None

    @field_validator("profile_id", "command_id", "label", "category")
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value

    @field_validator("label")
    @classmethod
    def _validate_label_length(cls, value: str) -> str:
        if len(value) > MAX_CAPABILITY_LABEL_LENGTH:
            raise ValueError("label is too long")
        return value

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        value = value.strip()
        if len(value) > MAX_CAPABILITY_DESCRIPTION_LENGTH:
            raise ValueError("description is too long")
        return value

    @field_validator("aliases")
    @classmethod
    def _validate_aliases(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            cleaned = item.strip()
            if not cleaned:
                raise ValueError("aliases cannot contain empty values")
            normalized.append(cleaned)
        if len(normalized) != len(set(item.lower() for item in normalized)):
            raise ValueError("Duplicate aliases are not allowed")
        return normalized

    @field_validator("category")
    @classmethod
    def _validate_category_length(cls, value: str) -> str:
        if len(value) > MAX_CAPABILITY_CATEGORY_LENGTH:
            raise ValueError("category is too long")
        return value

    @field_validator("allowed_trigger_sources")
    @classmethod
    def _validate_unique_trigger_sources(
        cls, value: list[AutomationTriggerSource]
    ) -> list[AutomationTriggerSource]:
        if len(value) != len(set(value)):
            raise ValueError("Duplicate trigger sources are not allowed")
        return value

    @field_validator("cooldown_remaining_ms")
    @classmethod
    def _validate_cooldown(cls, value: int) -> int:
        if value < 0:
            raise ValueError("cooldown_remaining_ms must be non-negative")
        return value


class AutomationCapabilitiesMessage(_StrictModel):
    type: Literal["automation/capabilities"] = "automation/capabilities"
    active_profile_id: Optional[str] = None
    revision: int = 0
    capabilities: list[AutomationCapability] = Field(default_factory=list)

    @field_validator("active_profile_id")
    @classmethod
    def _validate_optional_profile_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("active_profile_id cannot be empty")
        return value

    @field_validator("revision")
    @classmethod
    def _validate_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("revision must be non-negative")
        return value

    @field_validator("capabilities")
    @classmethod
    def _validate_capability_count(
        cls, capabilities: list[AutomationCapability]
    ) -> list[AutomationCapability]:
        if len(capabilities) > MAX_CAPABILITY_COUNT:
            raise ValueError("Capability catalogue is too large")
        command_ids = [capability.command_id for capability in capabilities]
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("Duplicate command IDs are not allowed")
        return capabilities

    @model_validator(mode="after")
    def _validate_profile_consistency(self) -> "AutomationCapabilitiesMessage":
        if self.active_profile_id is None and self.capabilities:
            raise ValueError("active_profile_id is required when capabilities are present")
        if self.active_profile_id is not None:
            invalid = [
                capability.command_id
                for capability in self.capabilities
                if capability.profile_id != self.active_profile_id
            ]
            if invalid:
                raise ValueError(
                    "Capabilities must match the active profile ID"
                )
        return self


class AutomationAssistantSettingsMessage(_StrictModel):
    type: Literal["automation/assistant-settings"] = "automation/assistant-settings"
    llm_automation_enabled: bool = False
    allow_autonomous_harmless_commands: bool = False
    allow_autonomous_low_risk_commands: bool = False
    confirmation_timeout_ms: int = 10_000
    max_pending_confirmations: int = 1
    announce_blocked_command_requests: bool = False
    include_command_suggestions_in_speech: bool = True
    result_acknowledgements_enabled: bool = False
    confirmation_phrases: list[str] = Field(default_factory=list)
    cancellation_phrases: list[str] = Field(default_factory=list)
    default_speech_mode: AssistantSpeechMode = "action"
    proactive_commentary_enabled: bool = False
    minimum_commentary_interval_ms: int = 15_000
    max_queued_conversation_events: int = 5
    announce_application_changes: bool = False
    acknowledge_routine_automation_success: bool = False
    voice_command_activation_mode: VoiceCommandActivationMode = "wake_phrase"
    voice_command_wake_phrases: list[str] = Field(default_factory=list)
    voice_command_push_to_command_hotkey: str = ""
    voice_command_ambiguity_timeout_ms: int = 15_000
    voice_command_acknowledgement_mode: VoiceCommandAcknowledgementMode = "none"
    telemetry: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confirmation_timeout_ms")
    @classmethod
    def _validate_confirmation_timeout(cls, value: int) -> int:
        if value < 1_000 or value > 120_000:
            raise ValueError("confirmation_timeout_ms must be between 1000 and 120000")
        return value

    @field_validator("max_pending_confirmations")
    @classmethod
    def _validate_max_pending_confirmations(cls, value: int) -> int:
        if value < 1 or value > MAX_PENDING_CONFIRMATIONS:
            raise ValueError(
                f"max_pending_confirmations must be between 1 and {MAX_PENDING_CONFIRMATIONS}"
            )
        return value

    @field_validator("confirmation_phrases", "cancellation_phrases")
    @classmethod
    def _validate_phrase_list(cls, value: list[str]) -> list[str]:
        if len(value) > MAX_PHRASE_COUNT:
            raise ValueError("Too many confirmation phrases")
        normalized: list[str] = []
        for item in value:
            item = item.strip().lower()
            if not item:
                raise ValueError("Phrases cannot be empty")
            if len(item) > MAX_PHRASE_LENGTH:
                raise ValueError("Phrase is too long")
            normalized.append(item)
        if len(normalized) != len(set(normalized)):
            raise ValueError("Duplicate phrases are not allowed")
        return normalized

    @field_validator("voice_command_wake_phrases")
    @classmethod
    def _validate_wake_phrase_list(cls, value: list[str]) -> list[str]:
        if len(value) > MAX_PHRASE_COUNT:
            raise ValueError("Too many wake phrases")
        normalized: list[str] = []
        for item in value:
            item = item.strip().lower()
            if not item:
                raise ValueError("Wake phrases cannot be empty")
            if len(item) > MAX_PHRASE_LENGTH:
                raise ValueError("Wake phrase is too long")
            normalized.append(item)
        if len(normalized) != len(set(normalized)):
            raise ValueError("Duplicate wake phrases are not allowed")
        return normalized

    @field_validator("minimum_commentary_interval_ms")
    @classmethod
    def _validate_commentary_interval(cls, value: int) -> int:
        if value < 1_000 or value > 120_000:
            raise ValueError(
                "minimum_commentary_interval_ms must be between 1000 and 120000"
            )
        return value

    @field_validator("max_queued_conversation_events")
    @classmethod
    def _validate_max_queued_conversation_events(cls, value: int) -> int:
        if value < 1 or value > 20:
            raise ValueError(
                "max_queued_conversation_events must be between 1 and 20"
            )
        return value

    @field_validator("voice_command_push_to_command_hotkey")
    @classmethod
    def _validate_push_to_command_hotkey(cls, value: str) -> str:
        return value.strip()

    @field_validator("voice_command_ambiguity_timeout_ms")
    @classmethod
    def _validate_voice_command_ambiguity_timeout(cls, value: int) -> int:
        if value < 2_000 or value > 120_000:
            raise ValueError(
                "voice_command_ambiguity_timeout_ms must be between 2000 and 120000"
            )
        return value

    @field_validator("telemetry")
    @classmethod
    def _validate_telemetry(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("telemetry must be an object")
        return value


class AutomationConfirmationResponseMessage(_StrictModel):
    type: Literal["automation/confirmation-response"] = "automation/confirmation-response"
    request_id: str
    action: Literal["confirm", "reject"]
    source: ConfirmationResponseSource = "manual"

    @field_validator("request_id")
    @classmethod
    def _validate_request_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("request_id cannot be empty")
        return value


class AssistantActiveApplicationMessage(_StrictModel):
    type: Literal["assistant/active-application"] = "assistant/active-application"
    process_name: str
    window_title: str
    detected_at: str
    confidence: float = 1.0

    @field_validator("process_name")
    @classmethod
    def _validate_process_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("process_name cannot be empty")
        if len(value) > MAX_ACTIVE_APP_PROCESS_NAME_LENGTH:
            raise ValueError("process_name is too long")
        return value

    @field_validator("window_title")
    @classmethod
    def _validate_window_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("window_title cannot be empty")
        if len(value) > MAX_ACTIVE_APP_WINDOW_TITLE_LENGTH:
            raise ValueError("window_title is too long")
        return value

    @field_validator("detected_at")
    @classmethod
    def _validate_detected_at(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("detected_at cannot be empty")
        return value

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("confidence must be between 0 and 1")
        return value


class AssistantContextSyncMessage(_StrictModel):
    type: Literal["assistant/context-sync"] = "assistant/context-sync"
    version: int
    matched_profile_id: Optional[str] = None
    effective_profile_id: Optional[str] = None
    manual_profile_id: Optional[str] = None
    profile_selection_mode: AutomationProfileSelectionMode = "manual"
    profile_selection_source: AutomationProfileSelectionSource = "none"
    profile_match_confidence: Optional[float] = None
    last_profile_transition_at: Optional[str] = None
    speech_mode: AssistantSpeechMode = "action"
    proactive_commentary_enabled: bool = False
    minimum_commentary_interval_ms: int = 15_000
    max_queued_conversation_events: int = 5
    announce_application_changes: bool = False
    acknowledge_routine_automation_success: bool = False
    player_speaking: Optional[bool] = None

    @field_validator(
        "matched_profile_id",
        "effective_profile_id",
        "manual_profile_id",
        "last_profile_transition_at",
    )
    @classmethod
    def _validate_optional_trimmed_string(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value < 0:
            raise ValueError("version must be non-negative")
        return value

    @field_validator("profile_match_confidence")
    @classmethod
    def _validate_profile_match_confidence(
        cls, value: Optional[float]
    ) -> Optional[float]:
        if value is None:
            return value
        if value < 0 or value > 1:
            raise ValueError("profile_match_confidence must be between 0 and 1")
        return value

    @field_validator("minimum_commentary_interval_ms")
    @classmethod
    def _validate_commentary_interval(cls, value: int) -> int:
        if value < 1000 or value > 120_000:
            raise ValueError(
                "minimum_commentary_interval_ms must be between 1000 and 120000"
            )
        return value

    @field_validator("max_queued_conversation_events")
    @classmethod
    def _validate_max_queued_conversation_events(cls, value: int) -> int:
        if value < 1 or value > 20:
            raise ValueError(
                "max_queued_conversation_events must be between 1 and 20"
            )
        return value


class AssistantPlayerStateMessage(_StrictModel):
    type: Literal["assistant/player-state"] = "assistant/player-state"
    speaking: bool
    timestamp: Optional[str] = None
    confidence: Optional[float] = None

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("timestamp cannot be empty")
        return value

    @field_validator("confidence")
    @classmethod
    def _validate_player_confidence(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if value < 0 or value > 1:
            raise ValueError("confidence must be between 0 and 1")
        return value


class AssistantVoiceCommandStateMessage(_StrictModel):
    type: Literal["assistant/voice-command-state"] = "assistant/voice-command-state"
    listening_active: bool


class AutomationVoiceCommandAmbiguityResponseMessage(_StrictModel):
    type: Literal["automation/voice-command-ambiguity-response"] = (
        "automation/voice-command-ambiguity-response"
    )
    ambiguity_id: str
    action: Literal["select", "cancel"]
    command_id: Optional[str] = None

    @field_validator("ambiguity_id")
    @classmethod
    def _validate_ambiguity_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ambiguity_id cannot be empty")
        return value

    @field_validator("command_id")
    @classmethod
    def _validate_command_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("command_id cannot be empty")
        return value

    @model_validator(mode="after")
    def _validate_selection(self) -> "AutomationVoiceCommandAmbiguityResponseMessage":
        if self.action == "select" and not self.command_id:
            raise ValueError("command_id is required when selecting a command")
        return self


class AutomationVoiceCommandResolveTestMessage(_StrictModel):
    type: Literal["automation/voice-command-resolve-test"] = (
        "automation/voice-command-resolve-test"
    )
    text: str

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text cannot be empty")
        return value


AutomationInboundMessage: TypeAlias = (
    AutomationExecuteMessage
    | AutomationStatusMessage
    | AutomationResultMessage
    | AutomationCancelMessage
    | AutomationEmergencyStopMessage
    | AutomationSpeakFixedMessage
    | AutomationCapabilitiesMessage
    | AutomationAssistantSettingsMessage
    | AutomationConfirmationResponseMessage
    | AssistantActiveApplicationMessage
    | AssistantContextSyncMessage
    | AssistantPlayerStateMessage
    | AssistantVoiceCommandStateMessage
    | AutomationVoiceCommandAmbiguityResponseMessage
    | AutomationVoiceCommandResolveTestMessage
)


def parse_automation_message(data: dict[str, Any]) -> AutomationInboundMessage:
    message_type = data.get("type")
    model_map: dict[str, type[AutomationInboundMessage]] = {
        "automation/execute": AutomationExecuteMessage,
        "automation/status": AutomationStatusMessage,
        "automation/result": AutomationResultMessage,
        "automation/cancel": AutomationCancelMessage,
        "automation/emergency_stop": AutomationEmergencyStopMessage,
        "automation/speak-fixed": AutomationSpeakFixedMessage,
        "automation/capabilities": AutomationCapabilitiesMessage,
        "automation/assistant-settings": AutomationAssistantSettingsMessage,
        "automation/confirmation-response": AutomationConfirmationResponseMessage,
        "assistant/active-application": AssistantActiveApplicationMessage,
        "assistant/context-sync": AssistantContextSyncMessage,
        "assistant/player-state": AssistantPlayerStateMessage,
        "assistant/voice-command-state": AssistantVoiceCommandStateMessage,
        "automation/voice-command-ambiguity-response": AutomationVoiceCommandAmbiguityResponseMessage,
        "automation/voice-command-resolve-test": AutomationVoiceCommandResolveTestMessage,
    }

    model = model_map.get(message_type)
    if model is None:
        raise ValueError(f"Unsupported automation message type: {message_type}")
    return model.model_validate(data)


@dataclass
class ClientAutomationState:
    status: AutomationRuntimeStatus = "disconnected"
    emergency_stopped: bool = False
    active_profile_id: Optional[str] = None
    profiles: dict[str, AutomationProfileSummary] = field(default_factory=dict)
    pending_request_ids: set[str] = field(default_factory=set)
    running_request_ids: set[str] = field(default_factory=set)
    last_error: Optional[str] = None
    results: dict[str, AutomationResultMessage] = field(default_factory=dict)

    def get_command(
        self, profile_id: str, command_id: str
    ) -> Optional[AutomationCommandSummary]:
        profile = self.profiles.get(profile_id)
        if not profile:
            return None
        for command in profile.commands:
            if command.command_id == command_id:
                return command
        return None


class AutomationTransport:
    def __init__(
        self, send_to_client: Callable[[str, dict[str, Any]], Awaitable[None]]
    ) -> None:
        self._send_to_client = send_to_client
        self._client_states: dict[str, ClientAutomationState] = {}

    def register_client(self, client_uid: str) -> None:
        self._client_states.setdefault(client_uid, ClientAutomationState())

    def unregister_client(self, client_uid: str) -> None:
        self._client_states.pop(client_uid, None)

    def get_client_state(self, client_uid: str) -> ClientAutomationState:
        self.register_client(client_uid)
        return self._client_states[client_uid]

    async def handle_incoming_message(
        self, client_uid: str, data: dict[str, Any]
    ) -> AutomationInboundMessage:
        message = parse_automation_message(data)
        state = self.get_client_state(client_uid)

        if isinstance(message, AutomationStatusMessage):
            state.status = message.status
            state.emergency_stopped = message.emergency_stopped
            state.active_profile_id = message.active_profile_id
            state.last_error = message.last_error
            state.running_request_ids = set(message.running_requests)
            state.profiles = {
                profile.profile_id: profile for profile in message.profiles
            }
        elif isinstance(message, AutomationResultMessage):
            state.pending_request_ids.discard(message.request_id)
            state.running_request_ids.discard(message.request_id)
            state.results[message.request_id] = message
            if message.status == "failed":
                state.last_error = message.error
            elif message.status == "completed":
                state.last_error = None
            if message.status in {"cancelled", "failed", "blocked"}:
                state.status = "cancelled" if message.status == "cancelled" else "error"
            elif not state.running_request_ids:
                state.status = "ready"
        elif isinstance(message, AutomationCancelMessage):
            state.pending_request_ids.discard(message.request_id)
            state.running_request_ids.discard(message.request_id)
            if not state.running_request_ids and state.status == "executing":
                state.status = "cancelled"
        elif isinstance(message, AutomationEmergencyStopMessage):
            state.emergency_stopped = message.emergency_stopped
            if message.emergency_stopped:
                state.status = "emergency_stopped"
                state.running_request_ids.clear()
                state.pending_request_ids.clear()
            elif not state.running_request_ids:
                state.status = "ready"

        return message

    async def request_execution(
        self,
        client_uid: str,
        profile_id: str,
        command_id: str,
        source: AutomationTriggerSource = "manual",
        variables: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> AutomationExecuteMessage:
        state = self.get_client_state(client_uid)

        if state.emergency_stopped:
            raise ValueError("Automation is emergency stopped")

        profile = state.profiles.get(profile_id)
        if not profile:
            raise ValueError(f"Unknown automation profile: {profile_id}")
        if not profile.enabled:
            raise ValueError(f"Automation profile is disabled: {profile_id}")

        command = state.get_command(profile_id, command_id)
        if not command:
            raise ValueError(f"Unknown automation command: {command_id}")
        if not command.enabled:
            raise ValueError(f"Automation command is disabled: {command_id}")

        request_id = (request_id or str(uuid4())).strip()
        if not request_id:
            raise ValueError("request_id cannot be empty")
        if request_id in state.pending_request_ids:
            raise ValueError(f"Duplicate automation request ID: {request_id}")

        message = AutomationExecuteMessage(
            request_id=request_id,
            profile_id=profile_id,
            command_id=command_id,
            source=source,
            variables=variables or {},
        )

        state.pending_request_ids.add(request_id)
        state.running_request_ids.add(request_id)
        state.status = "executing"

        await self._send_to_client(client_uid, message.model_dump(mode="json"))
        return message

    async def request_cancellation(self, client_uid: str, request_id: str) -> bool:
        state = self.get_client_state(client_uid)
        request_id = request_id.strip()
        if not request_id:
            raise ValueError("request_id cannot be empty")
        if (
            request_id not in state.pending_request_ids
            and request_id not in state.running_request_ids
        ):
            return False

        message = AutomationCancelMessage(request_id=request_id)
        await self._send_to_client(client_uid, message.model_dump(mode="json"))
        return True
