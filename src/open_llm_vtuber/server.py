"""
Open-LLM-VTuber Server
========================
This module contains the WebSocket server for Open-LLM-VTuber, which handles
the WebSocket connections, serves static files, and manages the web tool.
It uses FastAPI for the server and Starlette for static file serving.
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

from .config_manager.utils import Config
from .items_catalog import build_items_catalog
from .live.app import load_live_runtime
from .routes import init_client_ws_route, init_webtool_routes, init_proxy_route
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler


# Create a custom StaticFiles class that adds CORS headers
class CORSStaticFiles(StarletteStaticFiles):
    """
    Static files handler that adds CORS headers to all responses.
    Needed because Starlette StaticFiles might bypass standard middleware.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)

        # Add CORS headers to all responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"

        if path.endswith(".js"):
            response.headers["Content-Type"] = "application/javascript"

        return response


class AvatarStaticFiles(CORSStaticFiles):
    """
    Avatar files handler with security restrictions and CORS headers
    """

    async def get_response(self, path: str, scope):
        allowed_extensions = (".jpg", ".jpeg", ".png", ".gif", ".svg")
        if not any(path.lower().endswith(ext) for ext in allowed_extensions):
            return Response("Forbidden file type", status_code=403)
        response = await super().get_response(path, scope)
        return response


class WebSocketServer:
    """
    API server for Open-LLM-VTuber. This contains the websocket endpoint for the client, hosts the web tool, and serves static files.

    Creates and configures a FastAPI app, registers all routes
    (WebSocket, web tools, proxy) and mounts static assets with CORS.

    Args:
        config (Config): Application configuration containing system settings.
        default_context_cache (ServiceContext, optional):
            Pre‑initialized service context for sessions' service context to reference to.
            **If omitted, `initialize()` method needs to be called to load service context.**

    Notes:
        - If default_context_cache is omitted, call `await initialize()` to load service context cache.
        - Use `clean_cache()` to clear and recreate the local cache directory.
    """

    def __init__(self, config: Config, default_context_cache: ServiceContext = None):
        self.app = FastAPI(title="Open-LLM-VTuber Server")  # Added title for clarity
        self.config = config
        self.default_context_cache = (
            default_context_cache or ServiceContext()
        )  # Use provided context or initialize a new empty one waiting to be loaded
        self.ws_handler = WebSocketHandler(self.default_context_cache)
        self.live_runtime = None
        # It will be populated during the initialize method call

        # Add global CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Include routes, passing the context instance
        # The context will be populated during the initialize step
        self.app.include_router(
            init_client_ws_route(ws_handler=self.ws_handler),
        )
        self.app.include_router(
            init_webtool_routes(
                default_context_cache=self.default_context_cache,
                live_runtime_provider=lambda: self.live_runtime,
            ),
        )

        # Initialize and include proxy routes if proxy is enabled
        system_config = config.system_config
        if hasattr(system_config, "enable_proxy") and system_config.enable_proxy:
            # Construct the server URL for the proxy
            host = system_config.host
            port = system_config.port
            server_url = f"ws://{host}:{port}/client-ws"
            self.app.include_router(
                init_proxy_route(server_url=server_url),
            )

        # Mount cache directory first (to ensure audio file access)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        self.app.mount(
            "/cache",
            CORSStaticFiles(directory="cache"),
            name="cache",
        )

        # Mount static files with CORS-enabled handlers
        self.app.mount(
            "/live2d-models",
            CORSStaticFiles(directory="live2d-models"),
            name="live2d-models",
        )
        self.app.mount(
            "/bg",
            CORSStaticFiles(directory="backgrounds"),
            name="backgrounds",
        )
        self.app.mount(
            "/avatars",
            AvatarStaticFiles(directory="avatars"),
            name="avatars",
        )

        # Mount web tool directory separately from frontend
        self.app.mount(
            "/web-tool",
            CORSStaticFiles(directory="web_tool", html=True),
            name="web_tool",
        )

        # Mount main frontend last (as catch-all)
        self.app.mount(
            "/",
            CORSStaticFiles(directory="frontend", html=True),
            name="frontend",
        )

        @self.app.on_event("startup")
        async def _startup_live_runtime():
            if self.live_runtime:
                await self.live_runtime.start()

        @self.app.on_event("shutdown")
        async def _shutdown_live_runtime():
            if self.live_runtime:
                await self.live_runtime.stop()
            await self.default_context_cache.close()

    async def initialize(self):
        """Asynchronously load the service context from config.
        Calling this function is needed if default_context_cache was not provided to the constructor."""
        await self.default_context_cache.load_from_config(self.config)
        self._build_items_catalog()
        self._load_live_runtime()

    def _build_items_catalog(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        items_dir = project_root / "live2d-models" / "items"

        try:
            built_items = build_items_catalog(
                base_dir=str(items_dir),
                url_prefix="/live2d-models/items",
            )
            logger.info(
                f"[ItemsCatalog] Generated {len(built_items)} item entries from {items_dir}"
            )
        except Exception as exc:
            logger.warning(f"[ItemsCatalog] Unable to build item catalog: {exc}")

    def _load_live_runtime(self) -> None:
        config_path = Path("conf.yaml")
        if not config_path.exists():
            logger.warning("Live runtime disabled because conf.yaml was not found.")
            return

        try:
            host = self.config.system_config.host or "127.0.0.1"
            if host in {"0.0.0.0", "::"}:
                host = "127.0.0.1"
            backend_base_url = f"http://{host}:{self.config.system_config.port}"
            self.live_runtime = load_live_runtime(
                str(config_path),
                self.ws_handler,
                backend_base_url,
            )
            self.ws_handler.register_status_snapshot_provider(
                self.live_runtime.get_status_messages
            )
            logger.info("Live runtime configuration loaded successfully.")
        except Exception as exc:
            self.live_runtime = None
            logger.warning(f"Live runtime disabled: {exc}")

    @staticmethod
    def clean_cache():
        """Clean the cache directory by removing and recreating it."""
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
