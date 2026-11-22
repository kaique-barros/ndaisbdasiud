import asyncio
import contextlib
import json
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
from src.memory.events import EventsMemory
from src.memory.flash import FlashMemory
from src.memory.knowledge import KnowledgeMemory
from src.memory.notes import NotesMemory
from src.orchestrator import Orchestrator
from src.utils.bridge import ChatResponse
from src.utils.config import Config, ConfigError, Paths
from src.utils.logger import shutdown_logging
from helpers import INTERP_OK, REPO, DownModel, FakeModel, make_root



class FakeStore:
    name = "events"

    def __init__(self):
        self.queries = []

    async def search(self, query, top_k):
        self.queries.append(query)
        return ["Ontem choveu."]


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


class SlowModel(FakeModel):
    """Registra início/fim de cada chamada para provar que turnos não se intercalam."""

    def __init__(self, replies, log):
        super().__init__(replies)
        self.events = log

    async def _complete(self, messages, params):
        self.events.append("start")
        await asyncio.sleep(0.05)
        self.events.append("end")
        return await super()._complete(messages, params)


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

    def build(self, actor_replies=("Oi!",), interp_replies=None, stores=(), actor=None,
              extra=None, background=False):
        actor = actor or FakeModel(actor_replies)
        interp = FakeModel([INTERP_OK] * 10 if interp_replies is None else interp_replies)
        models = {"actor": actor, "primary_interpreter": interp, **(extra or {})}
        orch = Orchestrator.build(self.record.id, self.config, models, stores,
                                  background=background)
        return orch, actor, interp

    def marker(self, name):
        with Database(self.paths.messages_db(self.record.id)) as db:
            return FlashMemory(db).get_meta(name)

    async def test_two_turns_flash_emotions_registry_and_prompt(self):
        orch, actor, interp = self.build(["Oi, tudo bem!", "Que bom."])
        self.assertEqual(await orch.chat(["Olá"]), "Oi, tudo bem!")
        self.assertEqual(await orch.chat(["Como vai?"]), "Que bom.")

        first, second = actor.calls
        self.assertNotIn("Olá", first[0]["content"])  # 1º turno: flash vazia
        self.assertIn("Você é Ana", first[0]["content"])
        self.assertIn("Curiosa, calorosa", first[0]["content"])  # descrições da persona
        self.assertIn("caloroso", first[0]["content"])  # tom vindo do interpretador
        self.assertIn("alegria", first[0]["content"])  # emoção já atualizada antes do actor
        self.assertIn("user: Olá", second[0]["content"])  # 2º turno enxerga o 1º
        self.assertIn("persona: Oi, tudo bem!", second[0]["content"])
        self.assertEqual(second[1]["content"], "Como vai?")
        self.assertEqual(len(interp.calls), 2)
        self.assertIn("Emoções atuais", interp.calls[0][0]["content"])

        saved = orch.flash.last()
        self.assertEqual([m.role for m in saved], ["user", "persona"] * 2)
        self.assertEqual(saved[0].emotion_delta, {"joy": 0.3})  # variação gravada com a mensagem
        self.assertIsNone(saved[1].emotion_delta or None)
        persisted = json.loads((self.paths.persona_dir(self.record.id) / "emotions.json").read_text())
        self.assertGreater(persisted["state"]["joy"], 0.6)  # 0.15 + 0.3 + 0.3 (menos decaimento mínimo)
        self.assertIsNotNone(orch.registry.get(self.record.id).last_conversation_at)
        orch.close()

    async def test_interpreter_queries_feed_memory_search(self):
        store = FakeStore()
        reply = json.dumps({"queries": {"events": ["chuva ontem", "chuva ontem"], "outra": ["x"]},
                            "tone": "", "emotion_delta": {}})
        orch, actor, _ = self.build(interp_replies=[reply], stores=[store])
        await orch.chat(["será que choveu?"])
        self.assertEqual(store.queries, ["chuva ontem"])  # deduplicada; memória inexistente ignorada
        self.assertIn("## events\n- Ontem choveu.", actor.calls[0][0]["content"])
        orch.close()

    async def test_bad_interpreter_output_falls_back_and_chat_survives(self):
        store = FakeStore()
        orch, actor, _ = self.build(interp_replies=["isto não é json"], stores=[store])
        self.assertEqual(await orch.chat(["chuva?"]), "Oi!")
        self.assertEqual(store.queries, ["chuva?"])  # fallback: texto do usuário
        self.assertAlmostEqual(orch.emotions.values["joy"], 0.15, places=2)  # sem variação
        orch.close()

    async def test_disabled_interpreter_is_never_called(self):
        cfg_path = self.paths.configs / "engine" / "primary_interpreter.json"
        cfg_path.write_text('{"enabled": false}')
        store = FakeStore()
        orch, _, interp = self.build(interp_replies=[], stores=[store])
        await orch.chat(["oi"])
        self.assertEqual((interp.calls, store.queries), ([], ["oi"]))
        orch.close()

    async def test_failed_turn_is_not_saved_and_emotions_roll_back(self):
        orch, _, _ = self.build(["  "])
        with self.assertRaises(ActorError):
            await orch.chat(["oi"])
        self.assertEqual(orch.flash.last(), [])
        self.assertAlmostEqual(orch.emotions.values["joy"], 0.15, places=2)  # alegria +0.3 desfeita
        orch.close()

    async def test_concurrent_turns_are_serialized(self):
        events = []
        orch, _, _ = self.build(actor=SlowModel(["a", "b"], events))
        await asyncio.gather(orch.chat(["1"]), orch.chat(["2"]))
        self.assertEqual(events, ["start", "end", "start", "end"])
        self.assertEqual([m.content for m in orch.flash.last()], ["1", "a", "2", "b"])
        orch.close()

    async def test_default_stores_are_real_one_db_each(self):
        orch = Orchestrator.build(self.record.id, self.config,
                                  {"actor": FakeModel(["ok"]), "primary_interpreter": FakeModel([INTERP_OK])},
                                  stores=None)
        self.assertEqual(list(orch.memory.stores), ["events", "knowledge", "notes"])
        await orch.chat(["oi"])
        names = {p.name for p in (self.paths.persona_dir(self.record.id) / "memories").glob("*.db")}
        self.assertEqual(names, {"messages.db", "events.db", "knowledge.db", "notes.db"})
        orch.close()

    def background_models(self, notes_reply=None):
        notes = notes_reply or json.dumps({"notes": [{"block": 1, "text": "Fiquei muito feliz.",
                                                      "subject": "o usuário", "importance": 0.8}]})
        return {
            "notetaker": FakeModel([notes] * 5),
            "summary": FakeModel([json.dumps({"summary": "Resumo X", "state": "Feliz",
                                              "scene": {"local": "sala"}})] * 5),
            "consolidator": FakeModel([json.dumps({
                "events": [{"text": "O usuário conversou com Ana em 2026-11-22."}],
                "knowledge": [{"subject": "usuário", "relation": "conversou com", "object": "Ana"}],
                "obsolete": []})] * 5)}

    async def test_background_jobs_feed_memories_and_next_prompt(self):
        extra = self.background_models()
        orch, actor, _ = self.build(["r"] * 10, stores=None, extra=extra, background=True)
        for i in range(3):  # 6 mensagens: só o notetaker (a cada 6) está devido
            await orch.chat([f"oi {i}"])
        await orch.pipeline.wait()
        self.assertEqual((len(extra["notetaker"].calls), len(extra["summary"].calls)), (1, 0))
        self.assertEqual(self.marker("notes_up_to"), 6)

        await orch.chat(["oi 3"])  # 8 mensagens: o resumo (a cada 8) fica devido
        await orch.pipeline.wait()
        self.assertEqual(orch.files.summary, "Resumo X")
        self.assertEqual((orch.files.state, orch.files.scene()), ("Feliz", "- local: sala"))

        await orch.chat(["oi 4"])  # o próximo turno já enxerga resumo, estado e cenário
        system = actor.calls[-1][0]["content"]
        self.assertIn("Resumo X", system)
        self.assertIn("Feliz", system)
        self.assertIn("- local: sala", system)

        await orch.finalize()  # fim da conversa: processa o que faltava
        self.assertGreaterEqual(len(extra["consolidator"].calls), 1)
        for name in ("notes_up_to", "summary_up_to", "consolidated_up_to"):
            self.assertEqual(self.marker(name), 10, name)
        with Database(self.paths.memory_db(self.record.id, "notes")) as db:
            note = NotesMemory(db).list()[0]
        self.assertEqual((note.text, note.meta["subject"], note.meta["importance"]),
                         ("Fiquei muito feliz.", "o usuário", 0.8))
        self.assertIn("joy", note.meta["emotions"])
        with Database(self.paths.memory_db(self.record.id, "events")) as db:
            self.assertEqual(EventsMemory(db).list()[0].text, "O usuário conversou com Ana em 2026-11-22.")
        with Database(self.paths.memory_db(self.record.id, "knowledge")) as db:
            self.assertEqual(KnowledgeMemory(db).list()[0].relation, "conversou com")

    async def test_background_failure_never_breaks_the_chat(self):
        extra = self.background_models()
        down = DownModel()
        extra["notetaker"] = down
        orch, _, _ = self.build(["r"] * 10, stores=None, extra=extra, background=True)
        for i in range(3):  # 6 mensagens: o notetaker é chamado e falha
            self.assertEqual(await orch.chat([f"oi {i}"]), "r")  # o chat segue normal
        await orch.pipeline.wait()
        self.assertEqual(len(down.calls), 1)  # foi chamado de verdade...
        self.assertEqual(self.marker("notes_up_to"), 0)  # ...falhou e não avançou o marcador
        self.assertEqual(len(orch.flash.last()), 6)
        orch.close()

    async def test_background_can_be_disabled(self):
        orch, _, _ = self.build(["r"], background=False)
        self.assertIsNone(orch.pipeline)
        await orch.chat(["oi"])
        await orch.finalize()  # sem pipeline: só fecha

    async def test_empty_message_and_unknown_persona(self):
        orch, _, _ = self.build()
        with self.assertRaises(ValueError):
            await orch.chat(["  "])
        orch.close()
        with self.assertRaises(ConfigError):
            Orchestrator.build("nao-existe", self.config, {"actor": FakeModel()})


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
