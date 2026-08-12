import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_llm_vtuber.agent.stateless_llm.ollama_llm import OllamaLLM


def test_resolve_native_base_url_strips_v1_suffix():
    assert OllamaLLM._resolve_native_base_url("http://localhost:11434/v1") == (
        "http://localhost:11434"
    )
    assert OllamaLLM._resolve_native_base_url("http://localhost:11434/v1/") == (
        "http://localhost:11434"
    )
    assert OllamaLLM._resolve_native_base_url("http://localhost:11434") == (
        "http://localhost:11434"
    )


def test_convert_messages_to_native_format_extracts_images():
    native_messages = OllamaLLM._convert_messages_to_native_format(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Sure."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this photo?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,ZmFrZS1pbWFnZS1ieXRlcw==",
                            "detail": "auto",
                        },
                    },
                ],
            },
        ]
    )

    assert native_messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "Sure."},
        {
            "role": "user",
            "content": "What is this photo?",
            "images": ["ZmFrZS1pbWFnZS1ieXRlcw=="],
        },
    ]


def test_infer_capabilities_from_model_name_marks_common_vision_models():
    assert "vision" in OllamaLLM._infer_capabilities_from_model_name("gemma3:12b")
    assert "vision" in OllamaLLM._infer_capabilities_from_model_name("qwen2.5-vl")
    assert "vision" not in OllamaLLM._infer_capabilities_from_model_name("llama3.1")
