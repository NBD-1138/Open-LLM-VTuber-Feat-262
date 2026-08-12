# ruff: noqa: E402

from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from open_llm_vtuber.live.oauth import (
    TwitchOAuthManager,
    TwitchTokenBundle,
    TwitchTokenStore,
)
from open_llm_vtuber.live.twitch_chat import ChatBot


def test_sync_twitchio_token_state_updates_cached_clients():
    bot = ChatBot.__new__(ChatBot)
    bot.cfg = SimpleNamespace(oauth_token="oauth:new-access-token")
    bot._http = SimpleNamespace(token="old-access-token")
    bot._connection = SimpleNamespace(_token="old-access-token")

    bot._sync_twitchio_token_state()

    assert bot._http.token == "new-access-token"
    assert bot._connection._token == "new-access-token"


def test_twitch_token_store_round_trips_tokens(tmp_path):
    store = TwitchTokenStore(tmp_path / "private" / "twitch_tokens.json")
    store.save(
        TwitchTokenBundle(
            access_token="access-token",
            refresh_token="refresh-token",
            expires_in=1234,
            scope=["chat:read", "chat:edit"],
            token_type="bearer",
        )
    )

    loaded = store.load()

    assert loaded is not None
    assert loaded.access_token == "access-token"
    assert loaded.refresh_token == "refresh-token"
    assert loaded.expires_in == 1234
    assert loaded.scope == ["chat:read", "chat:edit"]
    assert loaded.token_type == "bearer"


def test_twitch_oauth_manager_builds_authorize_url():
    cfg = SimpleNamespace(
        client_id="client-id",
        client_secret="client-secret",
        oauth_token="",
        refresh_token="",
        scopes=["chat:read", "chat:edit"],
    )
    manager = TwitchOAuthManager(
        cfg,
        TwitchTokenStore(Path("private") / "test_tokens.json"),
        "http://127.0.0.1:12393",
    )

    authorize_url = manager.build_authorize_url(force_verify=True)
    parsed = urlparse(authorize_url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "id.twitch.tv"
    assert query["client_id"] == ["client-id"]
    assert query["response_type"] == ["code"]
    assert query["redirect_uri"] == ["http://127.0.0.1:12393/twitch/callback"]
    assert query["scope"] == ["chat:read chat:edit"]
    assert query["force_verify"] == ["true"]
    assert query["state"]
    assert manager.auth_flow_pending is True
