"""AstrBot plugin: translate final English chat replies before sending."""

import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star
import astrbot.api.message_components as Comp

from .translation import TranslationError, translate_spans, translate_with_deepl


class ReplyTranslate(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._pending: dict[int, tuple[AstrMessageEvent, str, float]] = {}
        self._translating_events: set[int] = set()

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("翻译")
    async def translate_command(self, event: AstrMessageEvent, action: str = ""):
        """开启当前会话翻译；使用 /翻译 关闭 关闭。"""
        if action not in ("", "关闭"):
            yield event.plain_result("用法：/翻译 或 /翻译 关闭")
            return
        enabled = action != "关闭"
        await self.put_kv_data(self._state_key(event), enabled)
        yield event.plain_result("当前会话翻译已开启。" if enabled else "当前会话翻译已关闭。")

    @filter.on_llm_response()
    async def mark_llm_reply(self, event: AstrMessageEvent, resp: LLMResponse):
        if id(event) in self._translating_events:
            return
        if not await self.get_kv_data(self._state_key(event), False):
            return
        original = getattr(resp, "completion_text", "")
        if not isinstance(original, str) or not original.strip():
            return
        self._prune_pending()
        self._pending[id(event)] = (event, original.strip(), time.monotonic())

    @filter.on_decorating_result()
    async def translate_before_send(self, event: AstrMessageEvent):
        pending = self._pending.pop(id(event), None)
        if pending is None or pending[0] is not event:
            return
        if not await self.get_kv_data(self._state_key(event), False):
            return
        result = event.get_result()
        if result is None:
            return
        chain = result.chain
        plain_parts = [part for part in chain if isinstance(part, Comp.Plain)]
        if not plain_parts:
            return
        original = pending[1]
        if original not in "".join(part.text for part in plain_parts):
            return

        # Commit only after all pieces succeed, so a failed request keeps the full original reply.
        event_id = id(event)
        self._translating_events.add(event_id)
        try:
            translated = [await translate_spans(part.text, self._translate) for part in plain_parts]
        except Exception as exc:
            logger.error("回复翻译失败，已发送原文：%s", type(exc).__name__)
            return
        finally:
            self._translating_events.discard(event_id)
        for part, text in zip(plain_parts, translated):
            part.text = text

    @filter.after_message_sent()
    async def clear_pending(self, event: AstrMessageEvent):
        self._pending.pop(id(event), None)

    @staticmethod
    def _state_key(event: AstrMessageEvent) -> str:
        return f"translation_enabled:{event.unified_msg_origin}"

    def _prune_pending(self):
        now = time.monotonic()
        for key, (_, _, created) in list(self._pending.items()):
            if now - created > 600:
                del self._pending[key]

    async def _translate(self, text: str) -> str:
        timeout = max(1, int(self.config.get("timeout_seconds", 20)))
        backend = self.config.get("backend", "deepl")
        if backend == "deepl":
            return await translate_with_deepl(
                text,
                self.config.get("deepl_api_key", ""),
                self.config.get("deepl_plan", "free"),
                timeout,
            )
        if backend == "astrbot_llm":
            provider_id = self.config.get("translation_provider_id", "").strip()
            if not provider_id:
                raise TranslationError("translation provider is missing")
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=(
                    "Translate the following English text into Simplified Chinese. "
                    "Return only the translation. Preserve Markdown formatting, "
                    "names, numbers, and punctuation. Do not answer the text.\n\n"
                    f"{text}"
                ),
            )
            translated = getattr(response, "completion_text", "")
            if not isinstance(translated, str) or not translated.strip():
                raise TranslationError("empty model response")
            return translated
        raise TranslationError("unknown translation backend")
