import os
import json
from html import escape
from uuid import uuid4
import numpy as np
from datetime import datetime
from typing import Any, Callable, Optional
from fastapi import APIRouter, Query, WebSocket, UploadFile, File, Response
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.websockets import WebSocketDisconnect
from loguru import logger
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler
from .proxy_handler import ProxyHandler


def init_client_ws_route(ws_handler: WebSocketHandler) -> APIRouter:
    """
    Create and return API routes for handling the `/client-ws` WebSocket connections.

    Args:
        ws_handler: Shared websocket handler instance for all client sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()

    @router.websocket("/client-ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for client connections"""
        await websocket.accept()
        client_uid = str(uuid4())

        try:
            await ws_handler.handle_new_connection(websocket, client_uid)
            await ws_handler.handle_websocket_communication(websocket, client_uid)
        except WebSocketDisconnect:
            await ws_handler.handle_disconnect(client_uid)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            await ws_handler.handle_disconnect(client_uid)
            raise

    return router


def init_proxy_route(server_url: str) -> APIRouter:
    """
    Create and return API routes for handling proxy connections.

    Args:
        server_url: The WebSocket URL of the actual server

    Returns:
        APIRouter: Configured router with proxy WebSocket endpoint
    """
    router = APIRouter()
    proxy_handler = ProxyHandler(server_url)

    @router.websocket("/proxy-ws")
    async def proxy_endpoint(websocket: WebSocket):
        """WebSocket endpoint for proxy connections"""
        try:
            await proxy_handler.handle_client_connection(websocket)
        except Exception as e:
            logger.error(f"Error in proxy connection: {e}")
            raise

    return router


def init_webtool_routes(
    default_context_cache: ServiceContext,
    live_runtime_provider: Optional[Callable[[], Any]] = None,
) -> APIRouter:
    """
    Create and return API routes for handling web tool interactions.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()

    def _build_twitch_auth_page(
        *,
        title: str,
        message: str,
        success: bool,
    ) -> HTMLResponse:
        title_html = escape(title)
        message_html = escape(message)
        accent = "#38a169" if success else "#e53e3e"
        html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title_html}</title>
    <style>
      body {{
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background: #111827;
        color: #f9fafb;
        font-family: Segoe UI, sans-serif;
      }}
      main {{
        width: min(36rem, calc(100vw - 2rem));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.5rem;
        background: rgba(17, 24, 39, 0.92);
        box-shadow: 0 20px 40px rgba(0,0,0,0.35);
      }}
      h1 {{
        margin: 0 0 0.75rem;
        font-size: 1.4rem;
        color: {accent};
      }}
      p {{
        margin: 0.5rem 0;
        line-height: 1.5;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>{title_html}</h1>
      <p>{message_html}</p>
      <p>You can return to Open-LLM-VTuber now.</p>
    </main>
  </body>
</html>"""
        return HTMLResponse(content=html)

    def _get_live_runtime():
        runtime = live_runtime_provider() if live_runtime_provider else None
        if runtime is None:
            raise RuntimeError("Twitch live runtime is unavailable.")
        return runtime

    @router.get("/web-tool")
    async def web_tool_redirect():
        """Redirect /web-tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/web_tool")
    async def web_tool_redirect_alt():
        """Redirect /web_tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/auth/twitch/start")
    @router.get("/twitch/start")
    async def start_twitch_auth(
        force_verify: bool = Query(
            default=True,
            description="Force Twitch to show the authorization prompt again.",
        ),
    ):
        try:
            runtime = _get_live_runtime()
            authorize_url = await runtime.begin_twitch_authorization(
                force_verify=force_verify
            )
        except Exception as exc:
            logger.warning(f"Unable to start Twitch OAuth flow: {exc}")
            return _build_twitch_auth_page(
                title="Twitch Authorization Unavailable",
                message=str(exc),
                success=False,
            )
        return RedirectResponse(authorize_url, status_code=307)

    @router.get("/auth/twitch/callback")
    @router.get("/twitch/callback")
    async def complete_twitch_auth(
        code: Optional[str] = None,
        state: Optional[str] = None,
        error: Optional[str] = None,
        error_description: Optional[str] = None,
    ):
        runtime = None
        try:
            runtime = _get_live_runtime()
        except Exception as exc:
            return _build_twitch_auth_page(
                title="Twitch Authorization Failed",
                message=str(exc),
                success=False,
            )

        if error:
            detail = error_description or error
            await runtime.record_twitch_authorization_failure(
                f"Twitch authorization was not completed: {detail}"
            )
            return _build_twitch_auth_page(
                title="Twitch Authorization Cancelled",
                message=detail,
                success=False,
            )

        if not code or not state:
            detail = "Twitch did not return the expected authorization code."
            await runtime.record_twitch_authorization_failure(detail)
            return _build_twitch_auth_page(
                title="Twitch Authorization Failed",
                message=detail,
                success=False,
            )

        try:
            await runtime.complete_twitch_authorization(code=code, state=state)
        except Exception as exc:
            await runtime.record_twitch_authorization_failure(str(exc))
            logger.warning(f"Twitch OAuth callback failed: {exc}")
            return _build_twitch_auth_page(
                title="Twitch Authorization Failed",
                message=str(exc),
                success=False,
            )

        return _build_twitch_auth_page(
            title="Twitch Authorization Complete",
            message="Your Twitch account is connected and the new tokens were stored locally.",
            success=True,
        )

    @router.get("/live2d-models/info")
    async def get_live2d_folder_info():
        """Get information about available Live2D models"""
        live2d_dir = "live2d-models"
        if not os.path.exists(live2d_dir):
            return JSONResponse(
                {"error": "Live2D models directory not found"}, status_code=404
            )

        valid_characters = []
        supported_extensions = [".png", ".jpg", ".jpeg"]

        for entry in os.scandir(live2d_dir):
            if entry.is_dir():
                folder_name = entry.name.replace("\\", "/")
                model3_file = os.path.join(
                    live2d_dir, folder_name, f"{folder_name}.model3.json"
                ).replace("\\", "/")

                if os.path.isfile(model3_file):
                    # Find avatar file if it exists
                    avatar_file = None
                    for ext in supported_extensions:
                        avatar_path = os.path.join(
                            live2d_dir, folder_name, f"{folder_name}{ext}"
                        )
                        if os.path.isfile(avatar_path):
                            avatar_file = avatar_path.replace("\\", "/")
                            break

                    valid_characters.append(
                        {
                            "name": folder_name,
                            "avatar": avatar_file,
                            "model_path": model3_file,
                        }
                    )
        return JSONResponse(
            {
                "type": "live2d-models/info",
                "count": len(valid_characters),
                "characters": valid_characters,
            }
        )

    @router.post("/asr")
    async def transcribe_audio(file: UploadFile = File(...)):
        """
        Endpoint for transcribing audio using the ASR engine
        """
        logger.info(f"Received audio file for transcription: {file.filename}")

        try:
            contents = await file.read()

            # Validate minimum file size
            if len(contents) < 44:  # Minimum WAV header size
                raise ValueError("Invalid WAV file: File too small")

            # Decode the WAV header and get actual audio data
            wav_header_size = 44  # Standard WAV header size
            audio_data = contents[wav_header_size:]

            # Validate audio data size
            if len(audio_data) % 2 != 0:
                raise ValueError("Invalid audio data: Buffer size must be even")

            # Convert to 16-bit PCM samples to float32
            try:
                audio_array = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            except ValueError as e:
                raise ValueError(
                    f"Audio format error: {str(e)}. Please ensure the file is 16-bit PCM WAV format."
                )

            # Validate audio data
            if len(audio_array) == 0:
                raise ValueError("Empty audio data")

            text = await default_context_cache.asr_engine.async_transcribe_np(
                audio_array
            )
            logger.info(f"Transcription result: {text}")
            return {"text": text}

        except ValueError as e:
            logger.error(f"Audio format error: {e}")
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return Response(
                content=json.dumps(
                    {"error": "Internal server error during transcription"}
                ),
                status_code=500,
                media_type="application/json",
            )

    @router.websocket("/tts-ws")
    async def tts_endpoint(websocket: WebSocket):
        """WebSocket endpoint for TTS generation"""
        await websocket.accept()
        logger.info("TTS WebSocket connection established")

        try:
            while True:
                data = await websocket.receive_json()
                text = data.get("text")
                if not text:
                    continue

                logger.info(f"Received text for TTS: {text}")

                # Split text into sentences
                sentences = [s.strip() for s in text.split(".") if s.strip()]

                try:
                    # Generate and send audio for each sentence
                    for sentence in sentences:
                        sentence = sentence + "."  # Add back the period
                        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
                        audio_path = (
                            await default_context_cache.tts_engine.async_generate_audio(
                                text=sentence, file_name_no_ext=file_name
                            )
                        )
                        logger.info(
                            f"Generated audio for sentence: {sentence} at: {audio_path}"
                        )

                        await websocket.send_json(
                            {
                                "status": "partial",
                                "audioPath": audio_path,
                                "text": sentence,
                            }
                        )

                    # Send completion signal
                    await websocket.send_json({"status": "complete"})

                except Exception as e:
                    logger.error(f"Error generating TTS: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})

        except WebSocketDisconnect:
            logger.info("TTS WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in TTS WebSocket connection: {e}")
            await websocket.close()

    return router
