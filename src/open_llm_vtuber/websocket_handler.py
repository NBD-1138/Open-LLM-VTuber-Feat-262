from typing import Any, Dict, List, Optional, Callable, TypedDict, TYPE_CHECKING
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from enum import Enum
from datetime import datetime
from uuid import uuid4
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from .live.publisher import LivePublisher

from .service_context import ServiceContext
from .agent.output_types import DisplayText
from .automation import (
    AutomationAssistantCoordinator,
    AutomationConfirmationResponseMessage,
    AutomationSpeakFixedMessage,
    AutomationTransport,
    parse_automation_message,
)
from .chat_group import (
    ChatGroupManager,
    handle_group_operation,
    handle_client_disconnect,
    broadcast_to_group,
)
from .message_handler import message_handler
from .utils.stream_audio import prepare_audio_payload
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
)
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .conversations.conversation_handler import (
    handle_conversation_trigger,
    handle_group_interrupt,
    handle_individual_interrupt,
)


class MessageType(Enum):
    """Enum for WebSocket message types"""

    GROUP = ["add-client-to-group", "remove-client-from-group"]
    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
    ]
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal", "audio-play-start"]
    DATA = ["mic-audio-data"]


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    images: Optional[List[str]]
    files: Optional[list[dict[str, str]]]
    history_uid: Optional[str]
    file: Optional[str]
    display_text: Optional[dict]
    request_id: Optional[str]
    profile_id: Optional[str]
    command_id: Optional[str]
    source: Optional[str]
    variables: Optional[dict]
    status: Optional[str]
    duration_ms: Optional[int]
    error: Optional[str]
    emergency_stopped: Optional[bool]
    profiles: Optional[list[dict]]
    active_profile_id: Optional[str]
    interrupt_policy: Optional[str]
    revision: Optional[int]
    capabilities: Optional[list[dict]]
    action: Optional[str]
    llm_automation_enabled: Optional[bool]
    allow_autonomous_harmless_commands: Optional[bool]
    allow_autonomous_low_risk_commands: Optional[bool]
    confirmation_timeout_ms: Optional[int]
    max_pending_confirmations: Optional[int]
    announce_blocked_command_requests: Optional[bool]
    include_command_suggestions_in_speech: Optional[bool]
    result_acknowledgements_enabled: Optional[bool]
    confirmation_phrases: Optional[list[str]]
    cancellation_phrases: Optional[list[str]]
    metadata: Optional[dict[str, Any]]
    forwarded: Optional[bool]
    speaking: Optional[bool]
    timestamp: Optional[str]
    confidence: Optional[float]


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(self, default_context_cache: ServiceContext):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, WebSocket] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.chat_group_manager = ChatGroupManager()
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}
        self.automation_transport = AutomationTransport(self._send_client_message)
        self.automation_assistant = AutomationAssistantCoordinator(
            self.automation_transport,
            self._send_client_message,
            self._send_fixed_speech_text,
            self._can_start_assistant_conversation,
            self._start_assistant_conversation,
        )
        self.live_publisher: Optional["LivePublisher"] = None
        self.status_snapshot_provider: Optional[Callable[[], Any]] = None

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "interrupt-signal": self._handle_interrupt,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-end": self._handle_conversation_trigger,
            "raw-audio-data": self._handle_raw_audio_data,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "audio-play-start": self._handle_audio_play_start,
            "request-init-config": self._handle_init_config_request,
            "heartbeat": self._handle_heartbeat,
            "update-talkback-tts": self._handle_update_talkback_tts,
            "automation/status": self._handle_automation_transport_message,
            "automation/result": self._handle_automation_transport_message,
            "automation/cancel": self._handle_automation_transport_message,
            "automation/emergency_stop": self._handle_automation_transport_message,
            "automation/capabilities": self._handle_automation_assistant_message,
            "automation/assistant-settings": self._handle_automation_assistant_message,
            "automation/confirmation-response": self._handle_automation_confirmation_response,
            "automation/voice-command-ambiguity-response": self._handle_automation_assistant_message,
            "automation/voice-command-resolve-test": self._handle_automation_assistant_message,
            "automation/speak-fixed": self._handle_automation_speak_fixed,
            "assistant/active-application": self._handle_automation_assistant_message,
            "assistant/context-sync": self._handle_automation_assistant_message,
            "assistant/player-state": self._handle_automation_assistant_message,
            "assistant/voice-command-state": self._handle_automation_assistant_message,
            "frontend-playback-complete": self._handle_frontend_playback_complete,
        }

    def register_live_publisher(self, publisher: "LivePublisher") -> None:
        """Register the live publisher so live integrations can use websocket broadcasts."""
        self.live_publisher = publisher

    def register_status_snapshot_provider(
        self,
        provider: Callable[[], Any],
    ) -> None:
        """Register a provider that returns status messages for newly connected clients."""
        self.status_snapshot_provider = provider

    def _can_start_assistant_conversation(self, client_uid: str) -> bool:
        websocket = self.client_connections.get(client_uid)
        context = self.client_contexts.get(client_uid)
        if websocket is None or context is None:
            return False

        group = self.chat_group_manager.get_client_group(client_uid)
        task_key = group.group_id if group and len(group.members) > 1 else client_uid
        task = self.current_conversation_tasks.get(task_key)
        return task is None or task.done()

    async def _start_assistant_conversation(
        self,
        client_uid: str,
        text: str,
        metadata: dict[str, Any],
    ) -> bool:
        websocket = self.client_connections.get(client_uid)
        context = self.client_contexts.get(client_uid)
        if websocket is None or context is None:
            return False
        if not self._can_start_assistant_conversation(client_uid):
            return False

        await handle_conversation_trigger(
            msg_type="text-input",
            data={
                "type": "text-input",
                "text": text,
                "metadata": metadata,
            },
            client_uid=client_uid,
            context=context,
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
        )
        return True

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        try:
            session_service_context = await self._init_service_context(
                websocket.send_text, client_uid
            )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )

            logger.info(f"Connection established for client {client_uid}")

        except Exception as e:
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e}"
            )
            await self._cleanup_failed_connection(client_uid)
            raise

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store client data and initialize group status"""
        self.client_connections[client_uid] = websocket
        self.client_contexts[client_uid] = session_service_context
        self.received_data_buffers[client_uid] = np.array([])

        self.chat_group_manager.client_group_map[client_uid] = ""
        self.automation_transport.register_client(client_uid)
        self.automation_assistant.register_client(client_uid)
        await self.send_group_update(websocket, client_uid)

    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": session_service_context.live2d_model.model_info,
                    "conf_name": session_service_context.character_config.conf_name,
                    "conf_uid": session_service_context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

        # Send initial group status
        await self.send_group_update(websocket, client_uid)
        await self._send_status_snapshots(websocket)
        await self.automation_assistant.send_state_snapshot(client_uid)

        # Start microphone
        await websocket.send_text(json.dumps({"type": "control", "text": "start-mic"}))

    async def _send_status_snapshots(self, websocket: WebSocket) -> None:
        provider = self.status_snapshot_provider
        if not provider:
            return

        try:
            snapshots = provider()
            if asyncio.iscoroutine(snapshots):
                snapshots = await snapshots
        except Exception as exc:
            logger.warning(f"Failed to build live status snapshots: {exc}")
            return

        if snapshots is None:
            return

        messages = snapshots if isinstance(snapshots, list) else [snapshots]
        for message in messages:
            if not message:
                continue
            await websocket.send_text(json.dumps(message))

    async def _init_service_context(
        self, send_text: Callable, client_uid: str
    ) -> ServiceContext:
        """Initialize service context for a new session by cloning the default context"""
        session_service_context = ServiceContext()
        await session_service_context.load_cache(
            config=self.default_context_cache.config.model_copy(deep=True),
            system_config=self.default_context_cache.system_config.model_copy(
                deep=True
            ),
            character_config=self.default_context_cache.character_config.model_copy(
                deep=True
            ),
            live2d_model=self.default_context_cache.live2d_model,
            asr_engine=self.default_context_cache.asr_engine,
            tts_engine=self.default_context_cache.tts_engine,
            vad_engine=self.default_context_cache.vad_engine,
            agent_engine=self.default_context_cache.agent_engine,
            translate_engine=self.default_context_cache.translate_engine,
            mcp_server_registery=self.default_context_cache.mcp_server_registery,
            tool_adapter=self.default_context_cache.tool_adapter,
            send_text=send_text,
            client_uid=client_uid,
            local_tools_factory=self._build_local_tools_for_context,
        )
        return session_service_context

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    try:
                        await websocket.send_text(
                            json.dumps({"type": "error", "message": str(e)})
                        )
                    except Exception as send_error:
                        logger.warning(
                            f"Failed to send websocket error response to {client_uid}: {send_error!r}"
                        )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """
        msg_type = data.get("type")
        if not msg_type:
            logger.warning("Message received without type")
            return

        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(websocket, client_uid, data)
        else:
            if msg_type != "frontend-playback-complete":
                logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_group_operation(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle group-related operations"""
        operation = data.get("type")
        target_uid = data.get(
            "invitee_uid" if operation == "add-client-to-group" else "target_uid"
        )

        await handle_group_operation(
            operation=operation,
            client_uid=client_uid,
            target_uid=target_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

    async def handle_disconnect(self, client_uid: str) -> None:
        """Handle client disconnection"""
        group = self.chat_group_manager.get_client_group(client_uid)
        context = self.client_contexts.get(client_uid)
        if group:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response="",
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )

        await handle_client_disconnect(
            client_uid=client_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

        # Clean up other client data
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.automation_transport.unregister_client(client_uid)
        self.automation_assistant.unregister_client(client_uid)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # Call context close to clean up resources (e.g., MCPClient)
        if context:
            await context.close()

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Clean up failed connection data"""
        context = self.client_contexts.get(client_uid)
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.chat_group_manager.client_group_map.pop(client_uid, None)
        self.automation_transport.unregister_client(client_uid)
        self.automation_assistant.unregister_client(client_uid)

        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        if context:
            await context.close()
        message_handler.cleanup_client(client_uid)

    async def broadcast_to_group(
        self, group_members: list[str], message: dict, exclude_uid: str = None
    ) -> None:
        """Broadcasts a message to group members"""
        await broadcast_to_group(
            group_members=group_members,
            message=message,
            client_connections=self.client_connections,
            exclude_uid=exclude_uid,
        )

    async def _send_client_message(self, client_uid: str, message: dict) -> None:
        websocket = self.client_connections.get(client_uid)
        if websocket is None:
            logger.debug(f"Skipping send to disconnected client: {client_uid}")
            return
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send client message to {client_uid}: {e!r}")

    async def send_group_update(self, websocket: WebSocket, client_uid: str):
        """Sends group information to a client"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            current_members = self.chat_group_manager.get_group_members(client_uid)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": current_members,
                        "is_owner": group.owner_uid == client_uid,
                    }
                )
            )
        else:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": [],
                        "is_owner": False,
                    }
                )
            )

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption"""
        heard_response = data.get("text", "")
        context = self.client_contexts[client_uid]
        group = self.chat_group_manager.get_client_group(client_uid)

        if group and len(group.members) > 1:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response=heard_response,
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )
        else:
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
            )

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        # Update history_uid in service context
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )

        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            context.history_uid = history_uid
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        audio_data = data.get("audio", [])
        if audio_data:
            self.received_data_buffers[client_uid] = np.append(
                self.received_data_buffers[client_uid],
                np.array(audio_data, dtype=np.float32),
            )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming raw audio data for VAD processing"""
        context = self.client_contexts[client_uid]
        chunk = data.get("audio", [])
        if chunk:
            for audio_bytes in context.vad_engine.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "interrupt"})
                    )
                elif audio_bytes == b"<|RESUME|>":
                    pass
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    self.received_data_buffers[client_uid] = np.append(
                        self.received_data_buffers[client_uid],
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "mic-audio-end"})
                    )

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        async def finish_suppressed_interaction() -> None:
            await websocket.send_text(json.dumps({"type": "force-new-message"}))
            await websocket.send_text(
                json.dumps({"type": "control", "text": "conversation-chain-end"})
            )

        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        input_source = str(metadata.get("source") or "player").lower()

        if data.get("type") == "text-input":
            input_text = str(data.get("text") or "").strip()
            if (
                input_text
                and input_source != "twitch"
                and await self.automation_assistant.maybe_handle_confirmation_phrase(
                client_uid, input_text
                )
            ):
                await finish_suppressed_interaction()
                return
            if input_text:
                suppress, next_metadata = (
                    await self.automation_assistant.prepare_conversation_metadata(
                        client_uid,
                        input_text,
                        metadata,
                        source=input_source or "player",
                    )
                )
                if suppress:
                    if input_source != "twitch":
                        await finish_suppressed_interaction()
                    return
                data = dict(data)
                data["metadata"] = next_metadata

        if (
            data.get("type") == "mic-audio-end"
            and self.automation_assistant.has_pending_confirmations(client_uid)
        ):
            audio_buffer = self.received_data_buffers.get(client_uid)
            if audio_buffer is not None and audio_buffer.size > 0:
                context = self.client_contexts[client_uid]
                try:
                    input_text = await context.asr_engine.async_transcribe_np(audio_buffer)
                    self.received_data_buffers[client_uid] = np.array([])
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "user-input-transcription",
                                "text": input_text,
                            }
                        )
                    )
                    if await self.automation_assistant.maybe_handle_confirmation_phrase(
                        client_uid, input_text
                    ):
                        await finish_suppressed_interaction()
                        return
                    suppress, next_metadata = (
                        await self.automation_assistant.prepare_conversation_metadata(
                            client_uid,
                            input_text,
                            metadata,
                            source="player_voice",
                        )
                    )
                    if suppress:
                        await finish_suppressed_interaction()
                        return
                    data = dict(data)
                    data["type"] = "text-input"
                    data["text"] = input_text
                    data["metadata"] = next_metadata
                except Exception as exc:
                    logger.warning(
                        f"Failed to pre-process pending confirmation audio for {client_uid}: {exc}"
                    )
        elif data.get("type") == "mic-audio-end":
            audio_buffer = self.received_data_buffers.get(client_uid)
            if audio_buffer is not None and audio_buffer.size > 0:
                context = self.client_contexts[client_uid]
                try:
                    input_text = await context.asr_engine.async_transcribe_np(audio_buffer)
                    self.received_data_buffers[client_uid] = np.array([])
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "user-input-transcription",
                                "text": input_text,
                            }
                        )
                    )
                    suppress, next_metadata = (
                        await self.automation_assistant.prepare_conversation_metadata(
                            client_uid,
                            input_text,
                            metadata,
                            source="player_voice",
                        )
                    )
                    if suppress:
                        await finish_suppressed_interaction()
                        return
                    data = dict(data)
                    data["type"] = "text-input"
                    data["text"] = input_text
                    data["metadata"] = next_metadata
                except Exception as exc:
                    logger.warning(
                        f"Failed to prepare conversation context for audio input {client_uid}: {exc}"
                    )

        await handle_conversation_trigger(
            msg_type=data.get("type", ""),
            data=data,
            client_uid=client_uid,
            context=self.client_contexts[client_uid],
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
        )

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            context = self.client_contexts[client_uid]
            await context.handle_config_switch(websocket, config_file_name)

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_audio_play_start(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Handle audio playback start notification
        """
        if not data.get("forwarded"):
            await self.automation_assistant.set_vtuber_speaking(client_uid, True)
        group_members = self.chat_group_manager.get_group_members(client_uid)
        if len(group_members) > 1:
            display_text = data.get("display_text")
            if display_text:
                silent_payload = prepare_audio_payload(
                    audio_path=None,
                    display_text=display_text,
                    actions=None,
                    forwarded=True,
                )
                await self.broadcast_to_group(
                    group_members, silent_payload, exclude_uid=client_uid
                )

    async def _handle_frontend_playback_complete(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        del websocket, data
        await self.automation_assistant.set_vtuber_speaking(client_uid, False)

    async def _handle_group_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle group info request"""
        await self.send_group_update(websocket, client_uid)

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        context = self.client_contexts.get(client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": context.live2d_model.model_info,
                    "conf_name": context.character_config.conf_name,
                    "conf_uid": context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")

    async def _handle_update_talkback_tts(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Update talkback TTS settings for live integrations."""
        publisher = self.live_publisher
        if not publisher:
            logger.warning(
                "Talkback TTS update received but no live publisher is registered."
            )
            return

        enabled = data.get("enabled")
        voice = data.get("voice")
        engine = data.get("engine")
        publisher.update_talkback_tts(
            enabled=bool(enabled) if isinstance(enabled, bool) else None,
            voice=str(voice).strip() if isinstance(voice, str) else None,
            engine=str(engine).strip() if isinstance(engine, str) else None,
        )

        provider = self.status_snapshot_provider
        if not provider:
            return

        snapshots = provider()
        if asyncio.iscoroutine(snapshots):
            snapshots = await snapshots
        if snapshots is None:
            return

        for message in snapshots if isinstance(snapshots, list) else [snapshots]:
            if message:
                await self.broadcast_json(message)

    async def _handle_automation_transport_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle general automation transport messages."""
        message = await self.automation_transport.handle_incoming_message(
            client_uid, data
        )
        await self.automation_assistant.handle_transport_message(client_uid, message)

    async def _handle_automation_assistant_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle assistant capability and settings synchronization messages."""
        del websocket
        message = parse_automation_message(data)
        await self.automation_assistant.handle_transport_message(client_uid, message)

    async def _handle_automation_confirmation_response(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle frontend confirmation and rejection actions."""
        del websocket
        message = AutomationConfirmationResponseMessage.model_validate(data)
        await self.automation_assistant.handle_confirmation_response(
            client_uid, message
        )

    async def _handle_automation_speak_fixed(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Generate fixed TTS speech for automation without going through the LLM."""
        message = AutomationSpeakFixedMessage.model_validate(data)
        del websocket
        await self._send_fixed_speech_text(
            client_uid,
            message.text,
            request_id=message.request_id,
        )

    def _build_local_tools_for_context(self, context: ServiceContext):
        return self.automation_assistant.build_local_tools(context.client_uid)

    async def _send_fixed_speech_text(
        self,
        client_uid: str,
        text: str,
        request_id: Optional[str] = None,
    ) -> None:
        websocket = self.client_connections.get(client_uid)
        context = self.client_contexts.get(client_uid)
        if websocket is None or context is None:
            raise ValueError(f"Client connection not found: {client_uid}")

        character_name = (
            context.character_config.character_name
            or context.character_config.conf_name
        )
        display_text = DisplayText(
            text=text,
            name=character_name,
            avatar=context.character_config.avatar or None,
        )

        audio_path = None
        try:
            file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
            audio_path = await context.tts_engine.async_generate_audio(
                text=text,
                file_name_no_ext=file_name,
            )
            payload = prepare_audio_payload(
                audio_path=audio_path,
                display_text=display_text,
                actions=None,
            )
            if request_id:
                payload["automation_request_id"] = request_id
            await websocket.send_text(json.dumps(payload))
        except Exception as e:
            logger.error(f"Automation speak_fixed failed: {e}")
            if request_id:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "automation/speak-fixed-error",
                            "request_id": request_id,
                            "message": "Unable to generate speech for automation.",
                        }
                    )
                )
        finally:
            if audio_path:
                context.tts_engine.remove_file(audio_path)

    async def broadcast_json(self, message: dict) -> None:
        """Broadcast a JSON-serializable payload to every connected websocket client."""
        payload = json.dumps(message)
        disconnect: list[str] = []
        for client_uid, connection in list(self.client_connections.items()):
            try:
                await connection.send_text(payload)
            except Exception as exc:
                logger.warning(
                    f"Failed to send broadcast to client {client_uid}: {exc}"
                )
                disconnect.append(client_uid)

        for uid in disconnect:
            await self.handle_disconnect(uid)

    async def handle_system_notification(
        self,
        text: str,
        category: str,
        source: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Broadcast a live notification as external chat so the normal LLM path can react."""
        display_text = f"[{source.capitalize()}] {text}"
        meta: Dict[str, Any] = {
            "category": category,
            "source": source,
        }
        if metadata:
            meta.update(metadata)

        if source == "twitch":
            await self.automation_assistant.record_twitch_event_for_all_clients(
                display_text,
                meta,
            )
        else:
            await self.automation_assistant.record_system_warning_for_all_clients(
                display_text,
                source=source,
            )

        await self.broadcast_json(
            {
                "type": "external-chat",
                "source": source,
                "user": source,
                "text": display_text,
                "metadata": meta,
            }
        )
