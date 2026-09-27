import importlib
import logging
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))


class Plain:
    def __init__(self, text):
        self.text = text


class Image:
    pass


class FakeStar:
    data = {}

    def __init__(self, context):
        self.context = context

    async def put_kv_data(self, key, value):
        self.data[key] = value

    async def get_kv_data(self, key, default=None):
        return self.data.get(key, default)


def install_astrbot_stubs():
    astrbot = types.ModuleType("astrbot")
    astrbot.__path__ = []
    api = types.ModuleType("astrbot.api")
    api.__path__ = []
    api.AstrBotConfig = dict
    api.logger = logging.getLogger("reply_translate_test")
    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = object

    def identity(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate

    def permission(permission_type):
        def decorate(fn):
            fn.permission_type = permission_type
            return fn
        return decorate

    event.filter = SimpleNamespace(
        PermissionType=SimpleNamespace(ADMIN="admin"),
        permission_type=permission,
        command=identity,
        on_llm_response=identity,
        on_decorating_result=identity,
        after_message_sent=identity,
    )
    provider = types.ModuleType("astrbot.api.provider")
    provider.LLMResponse = object
    star = types.ModuleType("astrbot.api.star")
    star.Context = object
    star.Star = FakeStar
    components = types.ModuleType("astrbot.api.message_components")
    components.Plain = Plain
    sys.modules.update({
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event,
        "astrbot.api.provider": provider,
        "astrbot.api.star": star,
        "astrbot.api.message_components": components,
    })


install_astrbot_stubs()
plugin = importlib.import_module(f"{ROOT.name}.main")
translation = importlib.import_module(f"{ROOT.name}.translation")


class FakeEvent:
    def __init__(self, origin, chain=None):
        self.unified_msg_origin = origin
        self.result = SimpleNamespace(chain=chain or [])

    def get_result(self):
        return self.result

    def plain_result(self, text):
        return text


async def collect(async_generator):
    return [item async for item in async_generator]


class TranslationTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_chinese_code_and_urls(self):
        source = "你好 Hello world. `print('hi')` https://example.com/a ```py\nhello\n``` Goodbye!"
        calls = []

        async def translator(text):
            calls.append(text)
            return "译文"

        result = await translation.translate_spans(source, translator)
        self.assertIn("你好", result)
        self.assertIn("`print('hi')`", result)
        self.assertIn("https://example.com/a", result)
        self.assertIn("```py\nhello\n```", result)
        self.assertNotIn("hello\n", " ".join(calls))

    async def test_deepl_request(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"translations": [{"text": "你好"}]})

        real_client = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        with patch.object(translation.httpx, "AsyncClient", side_effect=client_factory):
            result = await translation.translate_with_deepl("Hello", "secret", "free", 10)
        self.assertEqual(result, "你好")
        self.assertEqual(requests[0].url.host, "api-free.deepl.com")
        self.assertEqual(requests[0].headers["Authorization"], "DeepL-Auth-Key secret")
        self.assertIn(b"target_lang=ZH-HANS", requests[0].content)


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        FakeStar.data = {}
        self.context = SimpleNamespace(llm_generate=AsyncMock())
        self.instance = plugin.ReplyTranslate(self.context, {"backend": "astrbot_llm", "translation_provider_id": "translator"})

    async def test_admin_command_and_session_persistence(self):
        self.assertEqual(plugin.ReplyTranslate.translate_command.permission_type, "admin")
        first = FakeEvent("group:first")
        second = FakeEvent("group:second")
        self.assertEqual(await collect(self.instance.translate_command(first)), ["用法：/翻译 开启 或 /翻译 关闭"])
        self.assertTrue(await self.instance.get_kv_data(self.instance._state_key(first), True))
        self.assertEqual(await collect(self.instance.translate_command(first, "开启")), ["当前会话翻译已开启。"])
        restarted = plugin.ReplyTranslate(self.context, {})
        self.assertTrue(await restarted.get_kv_data(restarted._state_key(first), True))
        self.assertTrue(await restarted.get_kv_data(restarted._state_key(second), True))
        self.assertEqual(await collect(restarted.translate_command(first, "关闭")), ["当前会话翻译已关闭。"])
        self.assertFalse(await restarted.get_kv_data(restarted._state_key(first), True))

    async def test_only_matching_llm_reply_is_translated(self):
        event = FakeEvent("group:first", [Plain("Hello world!"), Image()])
        self.context.llm_generate.return_value = SimpleNamespace(completion_text="你好，世界！")
        response = SimpleNamespace(completion_text="Hello world!")
        await self.instance.mark_llm_reply(event, response)
        await self.instance.translate_before_send(event)
        self.assertEqual(event.result.chain[0].text, "你好，世界！")
        self.assertIsInstance(event.result.chain[1], Image)
        self.assertEqual(response.completion_text, "Hello world!")

        other = FakeEvent("group:first", [Plain("Plugin says hello")])
        await self.instance.translate_before_send(other)
        self.assertEqual(other.result.chain[0].text, "Plugin says hello")

    async def test_failure_keeps_entire_reply_and_hides_key(self):
        event = FakeEvent("group:first", [Plain("Hello world."), Plain("Goodbye world.")])
        await self.instance.mark_llm_reply(event, SimpleNamespace(completion_text="Hello world.Goodbye world."))
        self.instance._translate = AsyncMock(side_effect=["你好。", RuntimeError("secret-key")])
        with self.assertLogs("reply_translate_test", level="ERROR") as logs:
            await self.instance.translate_before_send(event)
        self.assertEqual([part.text for part in event.result.chain], ["Hello world.", "Goodbye world."])
        self.assertNotIn("secret-key", " ".join(logs.output))

    async def test_unconfigured_and_disabled_replies(self):
        event = FakeEvent("group:first", [Plain("Hello world!")])
        await self.instance.mark_llm_reply(event, SimpleNamespace(completion_text="Hello world!"))
        self.assertIn(id(event), self.instance._pending)
        await collect(self.instance.translate_command(event, "关闭"))
        await self.instance.translate_before_send(event)
        self.context.llm_generate.assert_not_awaited()
        self.assertEqual(event.result.chain[0].text, "Hello world!")
        await collect(self.instance.translate_command(event, "开启"))
        await self.instance.mark_llm_reply(event, SimpleNamespace(completion_text="Hello world!"))
        self.instance.config["translation_provider_id"] = ""
        with self.assertLogs("reply_translate_test", level="ERROR"):
            await self.instance.translate_before_send(event)
        self.assertEqual(event.result.chain[0].text, "Hello world!")

    async def test_translation_model_reply_does_not_mark_event_again(self):
        event = FakeEvent("group:first", [Plain("Hello world!")])
        async def generate(**kwargs):
            await self.instance.mark_llm_reply(event, SimpleNamespace(completion_text="你好，世界！"))
            return SimpleNamespace(completion_text="你好，世界！")

        self.context.llm_generate.side_effect = generate
        await self.instance.mark_llm_reply(event, SimpleNamespace(completion_text="Hello world!"))
        await self.instance.translate_before_send(event)
        self.assertEqual(event.result.chain[0].text, "你好，世界！")
        self.assertNotIn(id(event), self.instance._pending)


if __name__ == "__main__":
    unittest.main()
