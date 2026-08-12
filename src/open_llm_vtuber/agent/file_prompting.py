import base64
from typing import Iterable

from .input_types import FileData

TEXT_MIME_TYPES = {
    "application/json",
    "application/ld+json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/javascript",
    "application/x-javascript",
}

TEXT_EXTENSIONS = {
    ".cfg",
    ".conf",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".log",
    ".md",
    ".py",
    ".txt",
    ".tsx",
    ".ts",
    ".xml",
    ".yaml",
    ".yml",
}

MAX_FILE_PROMPT_CHARS = 6000


def _decode_file_bytes(data: str) -> bytes:
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def _looks_textual(file_data: FileData) -> bool:
    mime_type = (file_data.mime_type or "").lower()
    if mime_type.startswith("text/") or mime_type in TEXT_MIME_TYPES:
        return True

    file_name = (file_data.name or "").lower()
    return any(file_name.endswith(extension) for extension in TEXT_EXTENSIONS)


def _decode_textual_file(file_data: FileData) -> str | None:
    try:
        decoded_bytes = _decode_file_bytes(file_data.data)
    except Exception:
        return None

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = decoded_bytes.decode(encoding)
            return text.replace("\x00", "").strip()
        except UnicodeDecodeError:
            continue
    return None


def render_file_attachments_for_prompt(files: Iterable[FileData] | None) -> str:
    rendered_sections: list[str] = []
    for file_data in files or []:
        header = f"[User attached file: {file_data.name} ({file_data.mime_type})]"
        if _looks_textual(file_data):
            decoded_text = _decode_textual_file(file_data)
            if decoded_text:
                truncated = decoded_text[:MAX_FILE_PROMPT_CHARS]
                suffix = (
                    "\n[Attached file content truncated for brevity.]"
                    if len(decoded_text) > MAX_FILE_PROMPT_CHARS
                    else ""
                )
                rendered_sections.append(
                    f"{header}\n[Attached file content begins]\n"
                    f"{truncated}{suffix}\n[Attached file content ends]"
                )
                continue

            rendered_sections.append(
                f"{header}\n[The file looked textual but could not be decoded.]"
            )
            continue

        rendered_sections.append(
            f"{header}\n[Binary attachment received. The file contents are not "
            "directly readable in this chat.]"
        )

    return "\n\n".join(section for section in rendered_sections if section.strip())
