import asyncio
import logging
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.core.memories import MemoryManager
from src.data.database import Database
from src.data.personas import ARCHIVED, PersonaRegistry
from src.memory.flash import FlashMemory
from src.utils.config import Config, ConfigError, Paths
from src.utils.logger import get_logger, get_prompt_logger, setup_logging, shutdown_logging

REPO = Path(__file__).resolve().parents[1]


class FlashTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "messages.db")
        self.flash = FlashMemory(self.db, k=3)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_last_k_in_chronological_order(self):
        for i in range(5):
            self.flash.add("user", f"m{i}")
        self.assertEqual([m.content for m in self.flash.last()], ["m2", "m3", "m4"])

    def test_format_and_char_budget(self):
        self.flash.add("user", "a" * 10)
        self.flash.add("persona", "b" * 10)
        self.assertEqual(self.flash.format(), "user: " + "a" * 10 + "\npersona: " + "b" * 10)
        # orçamento só comporta a última linha: descarta a mais antiga
        self.assertEqual(self.flash.format(max_chars=20), "persona: " + "b" * 10)

    def test_filter_by_conversation(self):
        self.flash.add("user", "x", "c1")
        self.flash.add("user", "y", "c2")
        self.assertEqual([m.content for m in self.flash.last(conversation_id="c1")], ["x"])


class RegistryTest(unittest.TestCase):
    def test_create_list_status_and_order(self):
        with TemporaryDirectory() as tmp, Database(Path(tmp) / "global.db") as db:
            reg = PersonaRegistry(db)
            a = reg.create("Ana", {"cor": "azul"})
            b = reg.create("Bia")
            self.assertEqual(len(a.id), 36)
            reg.touch_conversation(b.id)
            self.assertEqual([p.name for p in reg.list()], ["Bia", "Ana"])
            reg.set_status(a.id, ARCHIVED)
            self.assertEqual([p.name for p in reg.list(status="active")], ["Bia"])
            self.assertEqual([p.name for p in reg.list(search="an")], ["Ana"])
            reg.update_metadata(a.id, idioma="pt")
            self.assertEqual(reg.get(a.id).metadata, {"cor": "azul", "idioma": "pt"})
            with self.assertRaises(ValueError):
                reg.set_status(a.id, "xyz")


class ConfigTest(unittest.TestCase):
    def test_loads_repo_configs(self):
        cfg = Config(Paths(REPO))
        self.assertEqual(cfg.engine("flash_memory")["k"], 20)
        self.assertEqual(cfg.provider("groq")["api_key_env"], "GROQ_API_KEY")
        self.assertEqual(cfg.provider("ollama")["base_url"], "http://localhost:11434/v1")
        self.assertEqual(len(cfg.engine("emotions_description")["wheel"]), 8)
        self.assertEqual(cfg.engine("summary")["every_messages"], 8)
        self.assertEqual(cfg.persona_parameters("persona_example")["language"], "pt-BR")
        with self.assertRaises(ConfigError):
            cfg.engine("nao_existe")


class LoggerTest(unittest.TestCase):
    def test_general_class_and_prompt_logs(self):
        with TemporaryDirectory() as tmp:
            setup_logging(Path(tmp))
            get_logger("flash", "memory").info("evento")
            get_prompt_logger("actor", "core").info("CONTEUDO DO PROMPT")
            for h in logging.getLogger("persona").handlers:
                h.flush()
            general = (Path(tmp) / "general.log").read_text()
            self.assertIn("evento", general)
            self.assertNotIn("CONTEUDO DO PROMPT", general)  # prompt não polui o geral
            self.assertIn("evento", (Path(tmp) / "memory" / "flash.log").read_text())
            self.assertIn("CONTEUDO DO PROMPT",
                          (Path(tmp) / "core" / "actor.prompt.log").read_text())
            shutdown_logging()


class FakeStore:
    def __init__(self, name, delay=0.0, items=None, fail=False):
        self.name, self.delay, self.items, self.fail = name, delay, items or [], fail

    async def search(self, query, top_k):
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        return [f"{self.name}:{query}:{i}" for i in self.items][:top_k]


class MemoryManagerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        setup_logging(Path(self._tmp.name))

    def tearDown(self):
        shutdown_logging()
        self._tmp.cleanup()

    async def test_searches_run_in_parallel(self):
        mgr = MemoryManager([FakeStore("events", 0.3, ["a"]), FakeStore("knowledge", 0.3, ["b"]),
                             FakeStore("notes", 0.3, ["c"])])
        t0 = time.perf_counter()
        res = await mgr.search_all({"events": ["q1", "q2"], "knowledge": ["q1"], "notes": ["q1"]})
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.6)  # sequencial levaria ~1.2s
        self.assertEqual(res["events"].items, ["events:q1:a", "events:q2:a"])

    async def test_failure_and_timeout_are_isolated(self):
        mgr = MemoryManager([FakeStore("events", items=["a"]), FakeStore("knowledge", fail=True),
                             FakeStore("notes", delay=1.0, items=["c"])], timeout_s=0.1)
        res = await mgr.search_all({"events": ["q"], "knowledge": ["q"], "notes": ["q"]})
        self.assertEqual(res["events"].items, ["events:q:a"])
        self.assertIn("RuntimeError", res["knowledge"].error)
        self.assertIn("timeout", res["notes"].error)
        prompt = mgr.build_prompt(res)
        self.assertIn("## events", prompt)
        self.assertNotIn("## knowledge", prompt)

    async def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            MemoryManager([FakeStore("x"), FakeStore("x")])


if __name__ == "__main__":
    unittest.main()
