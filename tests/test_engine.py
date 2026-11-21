import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.engine.model import ChatModel, Model
from src.utils.bridge import BridgeError, OpenAICompatibleBridge
from src.utils.config import Config, Paths
from src.utils.logger import setup_logging, shutdown_logging

REPO = Path(__file__).resolve().parents[1]


def ok_body(text="olá", **extra):
    return json.dumps({"model": "m", "choices": [{"message": {"content": text},
                       "finish_reason": "stop"}], "usage": {"total_tokens": 7}, **extra}).encode()


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, dict(headers), json.loads(body), timeout))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class LoggedTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        setup_logging(Path(self._tmp.name))

    def tearDown(self):
        shutdown_logging()
        self._tmp.cleanup()


class BridgeTest(LoggedTestCase):
    def make(self, *responses, **kw):
        t = FakeTransport(*responses)
        return OpenAICompatibleBridge("http://x/v1/", "KEY", backoff_s=0, transport=t, **kw), t

    async def test_success_builds_request_and_parses(self):
        bridge, t = self.make((200, ok_body("oi")))
        r = await bridge.chat("m1", [{"role": "user", "content": "q"}], temperature=0.3,
                              max_tokens=None)
        url, headers, payload, _ = t.calls[0]
        self.assertEqual(url, "http://x/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer KEY")
        self.assertEqual(payload["temperature"], 0.3)
        self.assertNotIn("max_tokens", payload)  # None não é enviado
        self.assertEqual((r.text, r.finish_reason, r.usage), ("oi", "stop", {"total_tokens": 7}))

    async def test_no_auth_header_without_key(self):
        t = FakeTransport((200, ok_body()))
        bridge = OpenAICompatibleBridge("http://x", None, transport=t)
        await bridge.chat("m", [])
        self.assertNotIn("Authorization", t.calls[0][1])

    async def test_retries_retryable_then_succeeds(self):
        bridge, t = self.make((429, b"slow down"), BridgeError("rede", retryable=True),
                              (200, ok_body("ok")))
        self.assertEqual((await bridge.chat("m", [])).text, "ok")
        self.assertEqual(len(t.calls), 3)

    async def test_non_retryable_fails_immediately(self):
        bridge, t = self.make((400, b"bad request"))
        with self.assertRaises(BridgeError) as ctx:
            await bridge.chat("m", [])
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(len(t.calls), 1)

    async def test_gives_up_after_max_retries(self):
        bridge, t = self.make((503, b"a"), (503, b"b"), (503, b"c"), max_retries=2)
        with self.assertRaises(BridgeError):
            await bridge.chat("m", [])
        self.assertEqual(len(t.calls), 3)

    async def test_malformed_response(self):
        bridge, _ = self.make((200, b'{"choices": []}'))
        with self.assertRaises(BridgeError):
            await bridge.chat("m", [])

    def test_from_provider_config_reads_key_from_env(self):
        cfg = {"base_url": "http://h", "api_key_env": "TEST_KEY_XYZ", "timeout_s": 5}
        with mock.patch.dict(os.environ, {"TEST_KEY_XYZ": "secret"}):
            b = OpenAICompatibleBridge.from_provider_config(cfg)
        self.assertEqual((b.api_key, b.timeout_s), ("secret", 5.0))
        with self.assertRaises(BridgeError):
            OpenAICompatibleBridge.from_provider_config({})


class ModelTest(LoggedTestCase):
    async def test_defaults_overrides_and_ask(self):
        t = FakeTransport((200, ok_body("resp")), (200, ok_body("resp2")))
        bridge = OpenAICompatibleBridge("http://x", transport=t)
        model = ChatModel("actor", "mid", bridge, {"temperature": 0.7, "max_tokens": 100,
                                                   "provider": "ignorado"})
        self.assertEqual(await model.ask("pergunta", system="seja breve"), "resp")
        payload = t.calls[0][2]
        self.assertEqual(payload["model"], "mid")
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])
        self.assertEqual((payload["temperature"], payload["max_tokens"]), (0.7, 100))
        self.assertNotIn("provider", payload)
        await model.ask("outra", temperature=0.1, foo="x")
        self.assertEqual(t.calls[1][2]["temperature"], 0.1)  # override por chamada
        self.assertNotIn("foo", t.calls[1][2])

    async def test_from_repo_configs(self):
        cfg = Config(Paths(REPO))
        t = FakeTransport((200, ok_body()))
        bridge = OpenAICompatibleBridge.from_provider_config(cfg.provider("example"), transport=t)
        model = ChatModel.from_configs("actor", cfg.engine("actor"), cfg.provider("example"), bridge)
        await model.ask("oi")
        self.assertEqual(t.calls[0][2]["model"], cfg.engine("actor")["model"])

    def test_missing_model_field_and_abstract(self):
        with self.assertRaises(ValueError):
            ChatModel.from_configs("x", {}, {"base_url": "http://h"})
        with self.assertRaises(TypeError):
            Model("x")  # abstrata


if __name__ == "__main__":
    unittest.main()
