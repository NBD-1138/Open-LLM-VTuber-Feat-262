from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_llm_vtuber.config_manager.live import LiveConfig
from open_llm_vtuber.live.config import load_conf


def test_load_conf_accepts_legacy_talkback_llm_block(tmp_path):
    config_path = tmp_path / "legacy-talkback.yaml"
    config_path.write_text(
        """
live_config:
  talkback:
    enabled: true
    llm:
      provider: openai
      model: gpt-4o-mini
      api_key_env: OPENAI_API_KEY
      max_chars: 111
""".strip(),
        encoding="utf-8",
    )

    cfg = load_conf(config_path)

    assert cfg.talkback.enabled is True
    assert cfg.talkback.max_chars == 111


def test_live_config_normalizes_legacy_talkback_llm_block():
    config = LiveConfig.model_validate(
        {
            "talkback": {
                "enabled": True,
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "max_chars": 222,
                },
            }
        }
    )

    assert config.talkback["enabled"] is True
    assert config.talkback["max_chars"] == 222
    assert "llm" not in config.talkback
