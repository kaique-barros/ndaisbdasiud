import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.core.interpreter import (Interpreter, InterpreterContext, InterpreterPromptBuilder,
                                  fallback_interpretation, parse_interpretation)
from src.data.database import Database
from src.engine.main.primary_interpreter import InterpreterError, PrimaryInterpreter
from src.memory.flash import FlashMemory
from src.utils.bridge import ChatResponse
from src.engine.model import Model
from src.utils.config import Config, Paths
from src.utils.logger import setup_logging, shutdown_logging

REPO = Path(__file__).resolve().parents[1]
STORES = ["events", "knowledge", "notes"]


class ParseTest(unittest.TestCase):
    def parse(self, raw, user=("oi",)):
        return parse_interpretation(raw, STORES, user)

    def test_plain_fenced_and_surrounded_json(self):
        body = '{"queries": {"events": ["a"]}, "tone": "calmo", "emotion_delta": {"joy": 0.2}}'
        for raw in (body, f"```json\n{body}\n```", f"Claro! Aqui está:\n{body}\nEspero ajudar."):
            r = self.parse(raw)
            self.assertEqual((r.queries, r.tone, r.emotion_delta, r.fallback),
                             ({"events": ["a"]}, "calmo", {"joy": 0.2}, False))

    def test_queries_are_cleaned_deduplicated_and_limited(self):
        long = "x" * 500
        raw = json.dumps({"queries": {
            "events": ["  a   b ", "a b", "c", "d", "e", 5, None, ""],
            "knowledge": "uma só",  # string solta vira lista
            "notes": [long], "desconhecida": ["z"]}})
        q = self.parse(raw).queries
        self.assertEqual(q["events"], ["a b", "c", "d"])  # limpa, sem duplicata, máx. 3
        self.assertEqual(q["knowledge"], ["uma só"])
        self.assertEqual(len(q["notes"][0]), 300)
        self.assertNotIn("desconhecida", q)

    def test_empty_queries_are_respected_but_missing_key_falls_back(self):
        self.assertEqual(self.parse('{"queries": {}, "tone": "x"}').queries, {})
        r = self.parse('{"tone": "x"}', user=("chuva?",))
        self.assertEqual(r.queries, {n: ["chuva?"] for n in STORES})

    def test_tone_and_delta_are_sanitized(self):
        raw = json.dumps({"tone": "  muito   longo " + "y" * 400,
                          "emotion_delta": {"Joy": 0.5, "anger": "alto", "fear": True, "trust": -1}})
        r = self.parse(raw)
        self.assertEqual(len(r.tone), 200)
        self.assertEqual(r.emotion_delta, {"joy": 0.5, "trust": -1.0})  # sem str/bool, chave em minúsculas
        self.assertEqual(self.parse('{"tone": 5, "emotion_delta": [1]}').tone, "")

    def test_unusable_output_raises(self):
        for raw in ("sem json", "[1, 2]", "{quebrado", ""):
            with self.assertRaises(ValueError):
                self.parse(raw)

    def test_fallback(self):
        f = fallback_interpretation(["  a ", "", "b"], STORES)
        self.assertEqual((f.queries["notes"], f.fallback), (["a\nb"], True))
        self.assertEqual(fallback_interpretation(["  "], STORES).queries, {})


class FakeEngine(PrimaryInterpreter):
    def __init__(self, result):
        self.result, self.calls = result, 0

    async def interpret(self, messages, **overrides):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class InterpreterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        setup_logging(Path(self.tmp.name))
        self.config = Config(Paths(REPO), load_env=False)
        self.db = Database(Path(self.tmp.name) / "m.db")
        self.flash = FlashMemory(self.db)
        for i in range(10):
            self.flash.add("user", f"msg{i}")

    def tearDown(self):
        shutdown_logging()
        self.db.close()
        self.tmp.cleanup()

    def builder(self, max_chars=12000, flash_k=3):
        labels = {"joy": "alegria", "anger": "raiva"}
        return InterpreterPromptBuilder(self.config.prompt("primary_interpreter"), self.flash,
                                        STORES, labels, max_chars, flash_k)

    def ctx(self, **kw):
        return InterpreterContext("Ana", ["olá"], persona="curiosa", emotions="calma", **kw)

    def test_prompt_contents(self):
        system, user = self.builder().build(self.ctx())
        self.assertEqual(user, {"role": "user", "content": "olá"})
        text = system["content"]
        self.assertIn("interpretador de Ana", text)
        self.assertIn("- knowledge: o que Ana aprendeu", text)  # {{name}} resolvido nas descrições
        self.assertIn("joy (alegria), anger (raiva)", text)
        self.assertIn("msg9", text)
        self.assertIn("msg7", text)
        self.assertNotIn("msg6", text)  # flash_k = 3
        self.assertNotIn("{{", text)

    def test_flash_respects_char_budget(self):
        no_flash = self.builder(flash_k=0).build(self.ctx())[0]["content"]  # só a parte fixa
        budget = 15  # cabe uma linha ("user: msg9" = 10 chars + quebra), não duas
        small = self.builder(max_chars=len(no_flash) + len("olá") + budget)
        text = small.build(self.ctx())[0]["content"]
        self.assertIn("msg9", text)
        self.assertNotIn("msg8", text)  # cortou as mais antigas primeiro

    def test_requires_user_message(self):
        with self.assertRaises(ValueError):
            self.builder().build(InterpreterContext("Ana", ["  "]))

    async def test_success_path(self):
        eng = FakeEngine('{"queries": {"notes": ["x"]}, "tone": "t", "emotion_delta": {"joy": 0.1}}')
        r = await Interpreter(self.builder(), eng, STORES).interpret(self.ctx())
        self.assertEqual((r.queries, r.tone, r.fallback), ({"notes": ["x"]}, "t", False))

    async def test_never_raises_and_falls_back(self):
        for result in (RuntimeError("rede"), InterpreterError("vazio"), "lixo"):
            r = await Interpreter(self.builder(), FakeEngine(result), STORES).interpret(self.ctx())
            self.assertTrue(r.fallback)
            self.assertEqual(r.queries["events"], ["olá"])

    async def test_disabled_skips_the_model(self):
        eng = FakeEngine("{}")
        r = await Interpreter(self.builder(), eng, STORES, enabled=False).interpret(self.ctx())
        self.assertEqual((eng.calls, r.fallback), (0, True))


class PrimaryInterpreterEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_text_and_rejects_empty(self):
        class M(Model):
            def __init__(self, text):
                super().__init__("m")
                self.text = text

            async def _complete(self, messages, params):
                return ChatResponse(text=self.text)

        with TemporaryDirectory() as tmp:
            setup_logging(Path(tmp))
            self.assertEqual(await PrimaryInterpreter(M(" {} ")).interpret([]), "{}")
            with self.assertRaises(InterpreterError):
                await PrimaryInterpreter(M("  ")).interpret([])
            shutdown_logging()


if __name__ == "__main__":
    unittest.main()
