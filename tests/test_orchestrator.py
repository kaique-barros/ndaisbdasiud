import asyncio
import contextlib
import io
import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src import cli
from src.core.actor import ActorContext, ActorPromptBuilder, render
from src.data.database import Database
from src.data.personas import PersonaRegistry, create_persona
from src.engine.main.actor import Actor, ActorError
from src.engine.model import Model
from src.memory.flash import FlashMemory
from src.orchestrator import Orchestrator
from src.utils.bridge import ChatResponse
from src.utils.config import Config, ConfigError, Paths
from src.utils.logger import shutdown_logging

REPO = Path(__file__).resolve().parents[1]


class FakeModel(Model):
    def __init__(self, replies=("Oi!",)):
        super().__init__("fake")
        self.replies, self.calls = list(replies), []

    async def _complete(self, messages, params):
        self.calls.append([dict(m) for m in messages])
        return ChatResponse(text=self.replies.pop(0))


class FakeStore:
    name = "events"

    def __init__(self):
        self.queries = []

    async def search(self, query, top_k):
        self.queries.append(query)
        return ["Ontem choveu."]


def make_root(tmp: str) -> Paths:
    root = Path(tmp)
    shutil.copytree(REPO / ".configs", root / ".configs")
    shutil.copytree(REPO / ".data" / "personas" / "persona_example",
                    root / ".data" / "personas" / "persona_example")
    return Paths(root)


class RenderTest(unittest.TestCase):
    def test_single_pass_empty_and_unknown(self):
        out = render("{{a}}|{{b}}|{{c}}", {"a": "{{b}}", "b": "  "})
        self.assertEqual(out, "{{b}}|(nenhum)|{{c}}")  # sem reexpansão; vazio vira (nenhum)


class PromptBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.paths = make_root(self.tmp.name)
        from src.utils.logger import setup_logging
        setup_logging(self.paths.logs)
        self.db = Database(self.tmp.name + "/m.db")
        self.flash = FlashMemory(self.db, k=50)
        for i in range(10):
            self.flash.add("user", f"mensagem antiga número {i}")

    def tearDown(self):
        shutdown_logging()
        self.db.close()
        self.tmp.cleanup()

    def ctx(self, **kw):
        return ActorContext(persona_name="Ana", user_messages=["olá", " ", "tudo bem?"], **kw)

    def test_fills_sections_and_joins_user_messages(self):
        builder = ActorPromptBuilder("N={{name}} S={{state}} F={{flash}}", self.flash, 10_000)
        system, user = builder.build(self.ctx(state="cansada"))
        self.assertTrue(system["content"].startswith("N=Ana S=cansada F=user: mensagem antiga"))
        self.assertEqual(user, {"role": "user", "content": "olá\ntudo bem?"})

    def test_flash_shrinks_to_budget_keeping_most_recent(self):
        small = ActorPromptBuilder("F={{flash}}", self.flash, 80)
        system = small.build(self.ctx())[0]["content"]
        self.assertLessEqual(len(system) + len("olá\ntudo bem?"), 80)
        self.assertIn("número 9", system)
        self.assertNotIn("número 0", system)

    def test_prompt_goes_to_prompt_log_not_general(self):
        ActorPromptBuilder("X={{memory}}", self.flash).build(self.ctx(memory="SEGREDO-NO-PROMPT"))
        for h in __import__("logging").getLogger("persona").handlers:
            h.flush()
        for h in __import__("logging").getLogger("persona.core.actor.prompt").handlers:
            h.flush()
        self.assertIn("SEGREDO-NO-PROMPT",
                      (self.paths.logs / "core" / "actor.prompt.log").read_text())
        self.assertNotIn("SEGREDO-NO-PROMPT", (self.paths.logs / "general.log").read_text())

    def test_requires_user_message(self):
        with self.assertRaises(ValueError):
            ActorPromptBuilder("x", self.flash).build(
                ActorContext(persona_name="Ana", user_messages=["  "]))


class ActorTest(unittest.IsolatedAsyncioTestCase):
    async def test_empty_reply_raises(self):
        with TemporaryDirectory() as tmp:
            from src.utils.logger import setup_logging
            setup_logging(Path(tmp))
            with self.assertRaises(ActorError):
                await Actor(FakeModel(["   "])).respond([{"role": "user", "content": "x"}])
            self.assertEqual(await Actor(FakeModel([" fala "])).respond([]), "fala")
            shutdown_logging()


class OrchestratorTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.paths = make_root(self.tmp.name)
        self.config = Config(self.paths, load_env=False)
        reg_db = Database(self.paths.global_db)
        self.record = create_persona(PersonaRegistry(reg_db), self.paths, "Ana")
        reg_db.close()

    def tearDown(self):
        shutdown_logging()
        self.tmp.cleanup()

    def build(self, model, stores=()):
        return Orchestrator.build(self.record.id, self.config, model, stores)

    async def test_two_turns_flash_registry_and_prompt(self):
        model = FakeModel(["Oi, tudo bem!", "Que bom."])
        orch = self.build(model)
        self.assertEqual(await orch.chat(["Olá"]), "Oi, tudo bem!")
        self.assertEqual(await orch.chat(["Como vai?"]), "Que bom.")

        first, second = model.calls
        self.assertNotIn("Olá", first[0]["content"])  # 1º turno: flash vazia
        self.assertIn("Você é Ana", first[0]["content"])
        self.assertIn("Curiosa, calorosa", first[0]["content"])  # descrições da persona
        self.assertIn("user: Olá", second[0]["content"])  # 2º turno enxerga o 1º
        self.assertIn("persona: Oi, tudo bem!", second[0]["content"])
        self.assertEqual(second[1]["content"], "Como vai?")
        self.assertEqual([m.role for m in orch.flash.last()], ["user", "persona"] * 2)
        self.assertIsNotNone(orch.registry.get(self.record.id).last_conversation_at)
        orch.close()

    async def test_memory_searches_feed_prompt(self):
        store, model = FakeStore(), FakeModel()
        orch = self.build(model, [store])
        await orch.chat(["chuva?"])
        self.assertEqual(store.queries, ["chuva?"])
        self.assertIn("## events\n- Ontem choveu.", model.calls[0][0]["content"])
        orch.close()

    async def test_failed_turn_is_not_saved(self):
        orch = self.build(FakeModel(["  "]))
        with self.assertRaises(ActorError):
            await orch.chat(["oi"])
        self.assertEqual(orch.flash.last(), [])
        orch.close()

    async def test_empty_message_and_unknown_persona(self):
        orch = self.build(FakeModel())
        with self.assertRaises(ValueError):
            await orch.chat(["  "])
        orch.close()
        with self.assertRaises(ConfigError):
            Orchestrator.build("nao-existe", self.config, FakeModel())


class PersonaCreationAndCliTest(unittest.TestCase):
    def test_create_persona_copies_template_and_sets_name(self):
        with TemporaryDirectory() as tmp:
            paths = make_root(tmp)
            with Database(paths.global_db) as db:
                rec = create_persona(PersonaRegistry(db), paths, "Bia")
            target = paths.persona_dir(rec.id)
            self.assertTrue((target / "descriptions" / "personality.md").is_file())
            self.assertEqual(Config(paths, load_env=False).persona_parameters(rec.id)["name"], "Bia")

    def test_cli_new_and_list(self):
        with TemporaryDirectory() as tmp:
            make_root(tmp)
            with mock.patch.dict(os.environ, {"PERSONA_ROOT": tmp}):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(cli.main(["--new", "Carla"]), 0)
                    self.assertEqual(cli.main(["--list"]), 0)
                text = out.getvalue()
                self.assertIn("persona criada: Carla", text)
                self.assertIn("Carla  [active]", text)
                self.assertEqual(cli.main(["id-inexistente"]), 1)


if __name__ == "__main__":
    unittest.main()
