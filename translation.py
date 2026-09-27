"""Translate natural-language spans without sending code or URLs to a translator."""

import re
from collections.abc import Awaitable, Callable

import httpx


DEEPL_ENDPOINTS = {
    "free": "https://api-free.deepl.com/v2/translate",
    "pro": "https://api.deepl.com/v2/translate",
}
PROTECTED = re.compile(
    r"```[\s\S]*?```|```[\s\S]*$|`[^`\n]+`|https?://[^\s<>()]+|[\u3400-\u9fff]+",
    re.IGNORECASE,
)
ENGLISH = re.compile(r"[A-Za-z]{2,}")


class TranslationError(Exception):
    """A translation provider failed or returned no usable text."""


async def translate_spans(
    text: str, translate: Callable[[str], Awaitable[str]]
) -> str:
    if not text or not ENGLISH.search(text):
        return text

    pieces: list[str] = []
    start = 0
    for match in PROTECTED.finditer(text):
        pieces.append(await _translate_piece(text[start : match.start()], translate))
        pieces.append(match.group())
        start = match.end()
    pieces.append(await _translate_piece(text[start:], translate))
    return "".join(pieces)


async def _translate_piece(
    text: str, translate: Callable[[str], Awaitable[str]]
) -> str:
    if not ENGLISH.search(text):
        return text
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()) :]
    translated = await translate(text.strip())
    if not isinstance(translated, str) or not translated.strip():
        raise TranslationError("empty translation")
    return leading + translated.strip() + trailing


async def translate_with_deepl(text: str, api_key: str, plan: str, timeout: int) -> str:
    if not api_key.strip():
        raise TranslationError("DeepL API key is missing")
    endpoint = DEEPL_ENDPOINTS.get(plan)
    if endpoint is None:
        raise TranslationError("invalid DeepL plan")

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            endpoint,
            headers={"Authorization": f"DeepL-Auth-Key {api_key.strip()}"},
            data={"text": text, "source_lang": "EN", "target_lang": "ZH-HANS"},
        )
        response.raise_for_status()
        payload = response.json()
    try:
        translated = payload["translations"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationError("invalid DeepL response") from exc
    if not isinstance(translated, str) or not translated.strip():
        raise TranslationError("empty DeepL response")
    return translated
