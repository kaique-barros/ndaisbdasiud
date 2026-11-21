import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.database import Database
from src.data.graphRag import GraphRag
from src.data.rag import RagIndex, cosine, fts_query
from src.memory.base import MemoryStore
from src.memory.events import EventsMemory
from src.memory.flash import FlashMemory
from src.memory.knowledge import KnowledgeMemory
from src.memory.notes import NotesMemory
from src.utils.bridge import BridgeError, OpenAICompatibleBridge
from src.utils.logger import setup_logging, shutdown_logging


class DbCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        setup_logging(Path(self.tmp.name) / "logs")
        self.db = Database(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        shutdown_logging()
        self.db.close()
        self.tmp.cleanup()


class RagTest(DbCase):
    def test_fts_query_is_safe(self):
        self.assertEqual(fts_query('Ação "x" OR a coração!'), '"ação" OR "or" OR "coração"')
        self.assertEqual(fts_query("a ! ?"), "")

    def test_lexical_search_ignores_accents_and_ranks(self):
        rag = RagIndex(self.db, "docs")
        rag.add("A Maria adora café com leite", {"k": 1})
        rag.add("O João joga futebol aos domingos")
        rag.add("Café é a bebida favorita da Maria")
        hits = rag.search("quem gosta de CAFE?", top_k=5)
        self.assertEqual({h.text for h in hits},
                         {"A Maria adora café com leite", "Café é a bebida favorita da Maria"})
        self.assertEqual(rag.search("zzz inexistente", 5), [])
        self.assertEqual(rag.search("!!", 5), [])

    def test_touch_delete_list_count(self):
        rag = RagIndex(self.db, "docs")
        a = rag.add("gato preto")
        rag.add("cachorro branco")
        self.assertIsNone(rag.get(a).last_accessed_at)
        rag.search("gato", 3)
        self.assertIsNotNone(rag.get(a).last_accessed_at)
        self.assertEqual([h.text for h in rag.list()], ["cachorro branco", "gato preto"])
        self.assertTrue(rag.delete(a))
        self.assertFalse(rag.delete(a))
        self.assertEqual((rag.count(), rag.search("gato", 3)), (1, []))  # FTS acompanha o delete
        with self.assertRaises(ValueError):
            rag.add("   ")
        with self.assertRaises(ValueError):
            RagIndex(self.db, "nome inválido; drop")

    def test_hybrid_with_embedder_finds_semantic_match(self):
        vocab = {"carro": [1, 0], "automóvel": [0.95, 0.1], "banana": [0, 1]}

        def embed(texts):
            return [next((v for w, v in vocab.items() if w in t.lower()), [0.1, 0.1])
                    for t in texts]

        rag = RagIndex(self.db, "docs", embed)
        rag.add("comprei um automóvel novo")
        rag.add("comi uma banana")
        hits = rag.search("meu carro", 2)  # sem nenhuma palavra em comum: só o vetor acha
        self.assertEqual([h.text for h in hits], ["comprei um automóvel novo"])
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertEqual(cosine([0, 0], [1, 0]), 0.0)

    def test_embedder_failure_falls_back_to_lexical(self):
        def broken(texts):
            raise RuntimeError("sem rede")

        rag = RagIndex(self.db, "docs", broken)
        rag.add("a lua é bonita")  # não pode falhar
        self.assertEqual([h.text for h in rag.search("lua", 3)], ["a lua é bonita"])
        self.assertTrue(rag._embed_failed)


class GraphTest(DbCase):
    def test_add_reinforces_and_is_case_insensitive(self):
        g = GraphRag(self.db)
        f1 = g.add_fact("Maria", "É Irmã De", "João", "pessoa", "pessoa")
        f2 = g.add_fact("maria", "é irmã de", "joão")
        self.assertEqual((f1.id, f1.relation, f1.weight, f2.weight), (f1.id, "é irmã de", 1, 2))
        self.assertEqual(g.count(), 1)
        with self.assertRaises(ValueError):
            g.add_fact("", "x", "y")

    def test_search_expands_neighbors_by_depth(self):
        g = GraphRag(self.db)
        g.add_fact("Maria", "mora em", "Lisboa")
        g.add_fact("Lisboa", "fica em", "Portugal")
        g.add_fact("Portugal", "faz parte da", "Europa")
        g.add_fact("Pedro", "trabalha com", "Carlos")
        facts = g.search("o que sabe da Maria?", top_k=10, depth=2)
        self.assertEqual([(f.subject, f.object) for f in facts],
                         [("Maria", "Lisboa"), ("Lisboa", "Portugal")])
        self.assertEqual(len(g.search("Maria", depth=3)), 3)
        self.assertEqual(g.search("ninguém conhecido", 5), [])
        top1 = g.search("Maria", top_k=1)
        self.assertEqual(top1[0].object, "Lisboa")
        self.assertIsNotNone(g.list_facts()[-1].last_accessed_at)  # acesso registrado

    def test_delete_fact_and_list(self):
        g = GraphRag(self.db)
        f = g.add_fact("A", "gosta de", "B")
        self.assertEqual(len(g.list_facts()), 1)
        self.assertTrue(g.delete_fact(f.id))
        self.assertEqual(g.list_facts(), [])


class StoresTest(DbCase):
    def test_stores_follow_memory_protocol_and_format(self):
        ev, kn, nt = EventsMemory(self.db), KnowledgeMemory(self.db), NotesMemory(self.db)
        for store in (ev, kn, nt):
            self.assertIsInstance(store, MemoryStore)
        ev.add("Ana viajou para o Rio", when="2026-11-20")
        kn.add_fact("Ana", "mora em", "Belo Horizonte")
        nt.add("Fiquei feliz com a visita da Ana", subject="Ana", importance=0.8)

        import asyncio
        out = asyncio.run(_gather(ev, kn, nt))
        self.assertEqual(out[0], ["[2026-11-20] Ana viajou para o Rio"])
        self.assertEqual(out[1], ["Ana — mora em — Belo Horizonte"])
        self.assertEqual(out[2], ["(sobre Ana) Fiquei feliz com a visita da Ana"])
        self.assertEqual(nt.list()[0].meta["importance"], 0.8)
        self.assertTrue(ev.delete(ev.list()[0].id))


async def _gather(*stores):
    import asyncio
    return await asyncio.gather(*(s.search("Ana", 5) for s in stores))


class FlashExtensionsTest(DbCase):
    def test_emotion_delta_since_meta_and_delete(self):
        flash = FlashMemory(self.db, k=5)
        a = flash.add("user", "oi", emotion_delta={"joy": 0.2})
        b = flash.add("persona", "olá")
        c = flash.add("user", "tchau")
        self.assertEqual(flash.last()[0].emotion_delta, {"joy": 0.2})
        self.assertEqual([m.id for m in flash.since(a)], [b, c])
        self.assertEqual(flash.count_since(b), 1)
        self.assertEqual(flash.get_meta("notes_up_to"), 0)
        flash.set_meta("notes_up_to", b)
        flash.set_meta("notes_up_to", c)
        self.assertEqual(flash.get_meta("notes_up_to"), c)
        self.assertTrue(flash.delete(b))
        self.assertFalse(flash.delete(b))

    def test_migrates_old_schema(self):
        self.db.script("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                       "conversation_id TEXT, role TEXT NOT NULL, content TEXT NOT NULL, "
                       "created_at TEXT NOT NULL DEFAULT 'x'); "
                       "INSERT INTO messages (role, content) VALUES ('user', 'antigo');")
        flash = FlashMemory(self.db)
        flash.add("user", "novo", emotion_delta={"anger": 0.1})
        self.assertEqual([m.content for m in flash.last()], ["antigo", "novo"])


class EmbedBridgeTest(unittest.TestCase):
    def make(self, body, status=200):
        calls = []

        def transport(url, headers, payload, timeout):
            calls.append((url, json.loads(payload)))
            return status, body

        return OpenAICompatibleBridge("http://h/v1", transport=transport, max_retries=0), calls

    def test_embed_orders_by_index(self):
        body = json.dumps({"data": [{"index": 1, "embedding": [2.0]},
                                    {"index": 0, "embedding": [1.0]}]}).encode()
        bridge, calls = self.make(body)
        self.assertEqual(bridge.embed_sync("m", ["a", "b"]), [[1.0], [2.0]])
        self.assertEqual(calls[0][0], "http://h/v1/embeddings")
        self.assertEqual(calls[0][1], {"model": "m", "input": ["a", "b"]})

    def test_embed_errors(self):
        bridge, _ = self.make(b'{"data": [{"index": 0, "embedding": [1]}]}')
        with self.assertRaises(BridgeError):
            bridge.embed_sync("m", ["a", "b"])  # quantidade diferente
        bridge, _ = self.make(b"not json")
        with self.assertRaises(BridgeError):
            bridge.embed_sync("m", ["a"])


if __name__ == "__main__":
    unittest.main()
