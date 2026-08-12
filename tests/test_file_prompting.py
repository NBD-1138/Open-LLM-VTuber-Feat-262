import base64

from open_llm_vtuber.agent.file_prompting import render_file_attachments_for_prompt
from open_llm_vtuber.agent.input_types import FileData


def test_render_file_attachments_includes_text_contents():
    encoded = base64.b64encode(b"Tenno, stay mobile.").decode("utf-8")
    prompt = render_file_attachments_for_prompt(
        [
            FileData(
                name="tips.txt",
                data=encoded,
                mime_type="text/plain",
            )
        ]
    )

    assert "tips.txt" in prompt
    assert "Tenno, stay mobile." in prompt
    assert "Attached file content begins" in prompt


def test_render_file_attachments_marks_binary_content():
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("utf-8")
    prompt = render_file_attachments_for_prompt(
        [
            FileData(
                name="preview.bin",
                data=encoded,
                mime_type="application/octet-stream",
            )
        ]
    )

    assert "preview.bin" in prompt
    assert "Binary attachment received" in prompt
