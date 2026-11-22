import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.core.consolidator import Consolidator, parse_consolidation
from src.core.persona_files import PersonaFiles
from src.core.pipeline import BackgroundPipeline, Job
from src.data.database import Database
from src.engine.base.consolidator import ConsolidatorEngine
from src.engine.base.notetaker import Notetaker, format_blocks, parse_notes, split_blocks
from src.engine.base.summary import Summarizer
from src.engine.model import Model
from src.memory.events import EventsMemory
from src.memory.flash import FlashMemory, Message
from src.memory.knowledge import KnowledgeMemory
from src.utils.bridge import ChatResponse
from src.utils.config import Config, Paths
from src.utils.jsonx import clip, extract_json_object
from src.utils.logger import setup_logging, shutdown_logging

REPO = Path(__file__).resolve().parents[1]
LABELS = {"joy": "alegria", "anger": "raiva", "fear": "medo"}


class ScriptedModel(Model):
    def __init__(self, *replies):
        super().__init__("scripted")
        self.replies, self.calls = list(replies), []

    async def _complete(self, messages, params):
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return ChatResponse(text=reply)


def msg(id, role="user", content="x", conv="c1", at="2026-11-21T10:00:00.000Z", delta=None):
    return Message(id, conv, role, content, at, delta or {})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        setup_logging(self.root / "logs")
        self.config = Config(Paths(REPO), load_env=False)

    def tearDown(self):
        shutdown_logging()
        self.tmp.cleanup()


class JsonxTest(unittest.TestCase):
    def test_extract_and_clip(self):
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json_object('ok {"a": {"b": 2}} fim'), {"a": {"b": 2}})
        for bad in ("nada", "[1]", ""):
            with self.assertRaises(ValueError):
                extract_json_object(bad)
        self.assertEqual(clip("uma frase bem comprida aqui", 14), "uma frase bem…")
        self.assertEqual(clip(" curto ", 14), "curto")


class NotetakerTest(Base):
    def test_blocks_split_by_conversation_and_gap_and_sum_deltas(self):
        ms = [msg(1, delta={"joy": 0.3}), msg(2, "persona"),
              msg(3, delta={"joy": 0.2, "anger": -0.1}, at="2026-11-21T10:05:00.000Z"),
              msg(4, at="2026-11-21T12:00:00.000Z"),             # pausa de 1h55: novo bloco
              msg(5, conv="c2", at="2026-11-21T12:01:00.000Z")]  # outra conversa: novo bloco
        blocks = split_blocks(ms, gap_minutes=30)
        self.assertEqual([[m.id for m in b.messages] for b in blocks], [[1, 2, 3], [4], [5]])
        self.assertEqual(blocks[0].total, {"joy": 0.5, "anger": -0.1})
        self.assertEqual(blocks[0].significance, 0.6)  # soma dos módulos
        self.assertEqual(blocks[1].significance, 0)
        # data inválida (bancos antigos) não quebra: fica no mesmo bloco
        self.assertEqual(len(split_blocks([msg(1, at="x"), msg(2, at="y")])), 1)

    def test_format_keeps_individual_and_total_deltas(self):
        blocks = split_blocks([msg(1, "user", "perdi o emprego", delta={"fear": 0.4}),
                               msg(2, "persona", "sinto muito")])
        text = format_blocks(blocks, "Ana", LABELS)
        self.assertIn("## Trecho 1 (variação emocional total: medo +0.40)", text)
        self.assertIn("usuário: perdi o emprego  [medo +0.40]", text)
        self.assertIn("Ana: sinto muito", text)

    def test_parse_notes_validation(self):
        blocks = split_blocks([msg(1, delta={"joy": 0.5})])
        raw = json.dumps({"notes": [
            {"block": 1, "text": "Fiquei tão feliz!", "subject": "o usuário", "importance": 9},
            {"block": 1, "text": "duplicada (mesmo trecho)"},
            {"block": 2, "text": "trecho inexistente"}, {"block": "1", "text": "tipo errado"},
            {"block": 1, "text": "  "}, "lixo"]})
        notes = parse_notes(raw, blocks)
        self.assertEqual(len(notes), 1)
        self.assertEqual((notes[0].subject, notes[0].importance, notes[0].emotions),
                         ("o usuário", 1.0, {"joy": 0.5}))  # importance limitada a 1
        no_importance = parse_notes('{"notes": [{"block": 1, "text": "ok"}]}', blocks)[0]
        self.assertEqual(no_importance.importance, 0.5)  # padrão: significância do trecho
        with self.assertRaises(ValueError):
            parse_notes('{"outra": []}', blocks)

    def make(self, model, **kw):
        return Notetaker(model, self.config.prompt("notetaker"), lambda: "Ana",
                         lambda: "curiosa", LABELS, **kw)

    def test_insignificant_batch_never_calls_model(self):
        model = ScriptedModel('{"notes": []}')
        notes = asyncio.run(self.make(model).take_notes([msg(1, delta={"joy": 0.05})]))
        self.assertEqual((notes, model.calls), ([], []))

    def test_significant_blocks_go_in_one_call(self):
        model = ScriptedModel('{"notes": [{"block": 1, "text": "A"}, {"block": 2, "text": "B"}]}')
        ms = [msg(1, delta={"anger": 0.6}), msg(2, conv="c2", delta={"joy": 0.5}),
              msg(3, conv="c3", delta={"joy": 0.01})]  # o 3º é banal
        notes = asyncio.run(self.make(model).take_notes(ms))
        self.assertEqual([n.text for n in notes], ["A", "B"])
        self.assertEqual(len(model.calls), 1)  # um único lote
        prompt = model.calls[0][0]["content"]
        self.assertIn("raiva +0.60", prompt)
        self.assertNotIn("Trecho 3", prompt)
        self.assertIn("curiosa", prompt)


class SummarizerTest(Base):
    def setUp(self):
        super().setUp()
        self.files = PersonaFiles(self.root / "persona")
        self.files.write_text("summary.md", "Resumo antigo.")
        self.files.write_text("state.md", "Estado antigo.")

    def make(self, model, max_chars=2000):
        return Summarizer(model, self.config.prompt("summary"), self.files, lambda: "Ana",
                          lambda: "calma", max_chars)

    def test_updates_three_files_and_uses_previous_ones(self):
        reply = json.dumps({"summary": "Novo resumo.", "state": "Animada.",
                            "scene": {"local": "café", "pessoas": 2, "lista": [1], "ok": True}})
        model = ScriptedModel(reply)
        asyncio.run(self.make(model).update([msg(1, content="vamos ao café?")]))
        self.assertEqual((self.files.summary, self.files.state), ("Novo resumo.", "Animada."))
        self.assertEqual(self.files.scene_data(), {"local": "café", "pessoas": 2})  # só str/número
        prompt = model.calls[0][0]["content"]
        self.assertIn("Resumo antigo.", prompt)
        self.assertIn("usuário: vamos ao café?", prompt)
        self.assertIn("No máximo 2000 caracteres", prompt)

    def test_invalid_output_leaves_files_untouched(self):
        for bad in ("sem json", '{"state": "só estado"}', '{"summary": "  "}'):
            with self.assertRaises(ValueError):
                asyncio.run(self.make(ScriptedModel(bad)).update([msg(1)]))
        self.assertEqual((self.files.summary, self.files.state), ("Resumo antigo.", "Estado antigo."))

    def test_summary_is_capped(self):
        reply = json.dumps({"summary": "palavra " * 100})
        asyncio.run(self.make(ScriptedModel(reply), max_chars=50).update([msg(1)]))
        self.assertLessEqual(len(self.files.summary), 51)
        self.assertTrue(self.files.summary.endswith("…"))
        self.assertEqual(self.files.state, "Estado antigo.")  # sem "state": mantém o anterior


class ConsolidatorTest(Base):
    def setUp(self):
        super().setUp()
        self.db = Database(self.root / "k.db")
        self.events, self.knowledge = EventsMemory(self.db), KnowledgeMemory(self.db)
        self.old_event = self.events.add("Ana mora em Lisboa desde 2020.")
        self.old_fact = self.knowledge.add_fact("Ana", "mora em", "Lisboa")

    def tearDown(self):
        self.db.close()
        super().tearDown()

    def make(self, model, **kw):
        return Consolidator(ConsolidatorEngine(model), self.config.prompt("consolidator"),
                            self.events, self.knowledge, lambda: "Bia", **kw)

    def test_parse_drops_invalid_and_foreign_ids(self):
        raw = json.dumps({"events": [{"text": "a"}, "b", {"text": ""}, 7, {"text": "a"}],
                          "knowledge": [{"subject": "A", "relation": "r", "object": "B"},
                                        {"subject": "A", "relation": "", "object": "B"}, "x"],
                          "obsolete": ["e1", "F99", "E5", 3, "E1"]})
        plan = parse_consolidation(raw, {"E1", "F2"})
        self.assertEqual((plan.events, len(plan.facts), plan.obsolete), (["a", "b"], 1, ["E1"]))
        with self.assertRaises(ValueError):
            parse_consolidation('{"nada": 1}', set())

    def test_applies_events_facts_and_obsolete(self):
        reply = json.dumps({
            "events": [{"text": "Em 2026-11-21 o usuário contou que Ana se mudou para o Porto."}],
            "knowledge": [{"subject": "Ana", "relation": "mora em", "object": "Porto",
                           "subject_type": "pessoa", "object_type": "lugar"}],
            "obsolete": [f"E{self.old_event}", f"F{self.old_fact.id}", "E999"]})
        model = ScriptedModel(reply)
        msgs = [msg(1, content="A Ana mora em Lisboa mas vai se mudar para o Porto",
                    at="2026-11-21T10:00:00.000Z")]
        result = asyncio.run(self.make(model).consolidate(msgs))
        self.assertEqual((result.events_added, result.facts_added, result.removed), (1, 1, 2))
        self.assertEqual([h.text for h in self.events.list()],
                         ["Em 2026-11-21 o usuário contou que Ana se mudou para o Porto."])
        self.assertEqual([self.knowledge.format(f) for f in self.knowledge.list()],
                         ["Ana — mora em — Porto"])
        self.assertEqual(self.events.list()[0].meta["when"], "2026-11-21")
        prompt = model.calls[0][0]["content"]  # o modelo viu as memórias parecidas, com ids
        self.assertIn(f"[E{self.old_event}]", prompt)
        self.assertIn(f"[F{self.old_fact.id}] Ana — mora em — Lisboa", prompt)

    def test_lookup_does_not_count_as_access(self):
        old = "2000-01-01T00:00:00+00:00"  # valor-sentinela: qualquer "touch" o sobrescreveria
        self.db.execute("UPDATE edges SET last_accessed_at = ?", (old,))
        asyncio.run(self.make(ScriptedModel('{"events": []}')).consolidate(
            [msg(1, content="Ana mora em Lisboa")]))
        self.assertIsNone(self.events.index.get(self.old_event).last_accessed_at)
        self.assertEqual(self.knowledge.list()[0].last_accessed_at, old)

    def test_reinforced_fact_is_not_deleted_by_same_batch_obsolete(self):
        reply = json.dumps({"knowledge": [{"subject": "Ana", "relation": "mora em", "object": "Lisboa"}],
                            "obsolete": [f"F{self.old_fact.id}"]})
        result = asyncio.run(self.make(ScriptedModel(reply)).consolidate(
            [msg(1, content="Ana mora em Lisboa")]))
        self.assertEqual(result.removed, 0)
        self.assertEqual(self.knowledge.list()[0].weight, 2)  # reforçado, não apagado


class PipelineTest(Base):
    def setUp(self):
        super().setUp()
        self.db = Database(self.root / "m.db")
        self.flash = FlashMemory(self.db)
        self.seen = []

    def tearDown(self):
        self.db.close()
        super().tearDown()

    def add(self, n):
        for i in range(n):
            self.flash.add("user", f"m{i}")

    def job(self, run=None, **kw):
        async def default(batch):
            self.seen.append([m.content for m in batch])
        return Job("j", "j_up_to", run or default, **kw)

    def test_runs_only_at_threshold_and_advances_marker(self):
        pipe = BackgroundPipeline(self.flash, [self.job(threshold=3)])
        self.add(2)
        self.assertFalse(asyncio.run(pipe.run_due()))
        self.add(1)
        self.assertTrue(asyncio.run(pipe.run_due()))
        self.assertEqual(self.seen, [["m0", "m1", "m0"]])
        self.assertEqual(self.flash.get_meta("j_up_to"), 3)
        self.assertFalse(asyncio.run(pipe.run_due()))  # nada novo

    def test_flush_forces_remaining_in_batches(self):
        pipe = BackgroundPipeline(self.flash, [self.job(threshold=100, max_batch=2)])
        self.add(5)
        asyncio.run(pipe.flush())
        self.assertEqual([len(b) for b in self.seen], [2, 2, 1])

    def test_failure_keeps_marker_and_is_retried(self):
        calls = []

        async def flaky(batch):
            calls.append(len(batch))
            if len(calls) == 1:
                raise ConnectionError("ollama fora do ar")

        pipe = BackgroundPipeline(self.flash, [self.job(flaky, threshold=1)])
        self.add(2)
        self.assertFalse(asyncio.run(pipe.run_due()))
        self.assertEqual(self.flash.get_meta("j_up_to"), 0)  # não avançou
        self.add(1)
        self.assertTrue(asyncio.run(pipe.run_due()))
        self.assertEqual(calls, [2, 3])  # o lote antigo foi refeito, junto com o novo
        self.assertEqual(self.flash.get_meta("j_up_to"), 3)

    def test_poison_batch_is_dropped_after_three_bad_outputs(self):
        async def bad(batch):
            raise ValueError("JSON inválido")

        job = self.job(bad, threshold=1)
        pipe = BackgroundPipeline(self.flash, [job])
        self.add(2)
        for _ in range(2):
            asyncio.run(pipe.run_due())
        self.assertEqual(self.flash.get_meta("j_up_to"), 0)
        asyncio.run(pipe.run_due())  # 3ª saída inválida: descarta o lote
        self.assertEqual((self.flash.get_meta("j_up_to"), job.failures), (2, 0))

    def test_kick_runs_in_background_and_rerun_is_coalesced(self):
        starts = []

        async def slow(batch):
            starts.append(len(batch))
            await asyncio.sleep(0.05)

        async def scenario():
            pipe = BackgroundPipeline(self.flash, [self.job(slow, threshold=1)])
            self.add(1)
            pipe.kick()
            await asyncio.sleep(0.01)
            self.add(1)
            pipe.kick()
            pipe.kick()  # várias durante a rodada viram uma só repetição
            await pipe.wait()
            return pipe

        asyncio.run(scenario())
        self.assertEqual(starts, [1, 1])  # nunca duas rodadas sobrepostas
        self.assertEqual(self.flash.get_meta("j_up_to"), 2)

    def test_independent_jobs_have_independent_markers(self):
        a = Job("a", "a_up_to", lambda b: asyncio.sleep(0), threshold=1)
        b = Job("b", "b_up_to", lambda b: asyncio.sleep(0), threshold=5)
        pipe = BackgroundPipeline(self.flash, [a, b])
        self.add(2)
        asyncio.run(pipe.run_due())
        self.assertEqual((self.flash.get_meta("a_up_to"), self.flash.get_meta("b_up_to")), (2, 0))


if __name__ == "__main__":
    unittest.main()
