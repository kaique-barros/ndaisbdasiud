import json
import unittest
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from helpers import INTERP_OK, DownModel, FakeModel, RepeatModel, make_root
from src.api.app import create_app
from src.api.v1.persona.config.assets import MAX_BYTES, sniff
from src.utils.bridge import BridgeError
from src.utils.config import Config
from src.utils.logger import setup_logging, shutdown_logging

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
BG_SUMMARY = json.dumps({"summary": "Resumo da API", "state": "Tranquila", "scene": {"local": "sala"}})


class ApiCase(unittest.TestCase):
    background = False

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.paths = make_root(self.tmp.name)
        setup_logging(self.paths.logs)  # os modelos falsos criam loggers já no construtor
        self.actor = RepeatModel(["Olá, tudo bem!"])
        self.interp = RepeatModel([INTERP_OK])
        models = {"actor": self.actor, "primary_interpreter": self.interp}
        if self.background:
            models.update(summary=RepeatModel([BG_SUMMARY]), notetaker=RepeatModel(['{"notes": []}']),
                          consolidator=RepeatModel(['{"events": [], "knowledge": []}']))
        self.models = models
        self.client = TestClient(create_app(Config(self.paths, load_env=False), models,
                                            background=self.background))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        shutdown_logging()
        self.tmp.cleanup()

    def new(self, name="Ana"):
        r = self.client.post("/v1/personas", json={"name": name})
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()["id"]

    def url(self, pid, suffix=""):
        return f"/v1/personas/{pid}{suffix}"


class PersonasTest(ApiCase):
    def test_health_create_list_search_get(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        a, b = self.new("Ana"), self.new("Beatriz")
        self.assertEqual(len(a), 36)
        names = lambda r: sorted(p["name"] for p in r.json())
        self.assertEqual(names(self.client.get("/v1/personas")), ["Ana", "Beatriz"])
        self.assertEqual(names(self.client.get("/v1/personas", params={"search": "bea"})), ["Beatriz"])
        self.assertEqual(self.client.get(self.url(b)).json()["status"], "active")

    def test_validation_and_unknown_ids(self):
        self.assertEqual(self.client.post("/v1/personas", json={"name": "  "}).status_code, 422)
        self.assertEqual(self.client.post("/v1/personas", json={}).status_code, 422)
        for bad in ("nao-e-uuid", "..%2F..%2Fetc", "00000000-0000-0000-0000-000000000000"):
            self.assertEqual(self.client.get(f"/v1/personas/{bad}").status_code, 404, bad)
            self.assertEqual(self.client.get(f"/v1/personas/{bad}/config/memory/events").status_code,
                             404, bad)

    def test_patch_name_syncs_file_registry_and_archive_blocks_chat(self):
        pid = self.new("Ana")
        r = self.client.patch(self.url(pid), json={"name": "Nova"})
        self.assertEqual(r.json()["name"], "Nova")
        params = json.loads((self.paths.persona_dir(pid) / "parameters.json").read_text())
        self.assertEqual(params["name"], "Nova")
        self.assertEqual(params["language"], "pt-BR")  # os outros parâmetros ficam
        self.client.patch(self.url(pid), json={"status": "archived"})
        self.assertEqual(self.client.post(self.url(pid, "/chat"), json={"messages": ["oi"]}).status_code, 409)
        self.assertEqual(len(self.client.get("/v1/personas", params={"status": "active"}).json()), 0)
        self.client.patch(self.url(pid), json={"status": "active"})
        self.assertEqual(self.client.post(self.url(pid, "/chat"), json={"messages": ["oi"]}).status_code, 200)
        self.assertEqual(self.client.patch(self.url(pid), json={"status": "xyz"}).status_code, 422)


class ChatTest(ApiCase):
    def test_chat_flash_status_and_registry(self):
        pid = self.new()
        r = self.client.post(self.url(pid, "/chat"), json={"messages": ["Oi!", "Tudo bem?"]})
        self.assertEqual(r.json(), {"reply": "Olá, tudo bem!", "persona": "Ana"})
        msgs = self.client.get(self.url(pid, "/config/memory/messages")).json()
        self.assertEqual([m["role"] for m in msgs], ["user", "user", "persona"])
        self.assertEqual(msgs[1]["emotion_delta"], {"joy": 0.3})  # na última mensagem do turno
        status = self.client.get(self.url(pid, "/status")).json()
        self.assertIn("joy", {a["emotion"] for a in status["emotions"]["active"]})
        self.assertIsNotNone(self.client.get(self.url(pid)).json()["last_conversation_at"])
        # o menu lista quem conversou primeiro
        other = self.new("Zé")
        self.assertEqual([p["id"] for p in self.client.get("/v1/personas").json()], [pid, other])

    def test_chat_validation(self):
        pid = self.new()
        for body in ({"messages": []}, {"messages": ["  ", ""]}, {"messages": ["x" * 8001]},
                     {"messages": ["a"] * 21}, {}):
            self.assertEqual(self.client.post(self.url(pid, "/chat"), json=body).status_code, 422, body)

    def test_model_failure_is_a_502_and_nothing_is_saved(self):
        pid = self.new()
        self.client.post(self.url(pid, "/chat"), json={"messages": ["primeira"]})  # abre o orchestrator
        orch = self.client.app.state.ctx.orchestrator(pid)
        orch.actor.model = DownModel(BridgeError("HTTP 503", status=503))
        r = self.client.post(self.url(pid, "/chat"), json={"messages": ["segunda"]})
        self.assertEqual(r.status_code, 502)
        self.assertIn("falha ao falar com o modelo", r.json()["detail"])
        saved = self.client.get(self.url(pid, "/config/memory/messages")).json()
        self.assertEqual([m["content"] for m in saved], ["primeira", "Olá, tudo bem!"])


class ConfigTest(ApiCase):
    def test_parameters_put_syncs_name_everywhere(self):
        pid = self.new("Ana")
        self.client.post(self.url(pid, "/chat"), json={"messages": ["oi"]})  # orchestrator aberto
        r = self.client.put(self.url(pid, "/config/parameters"), json={"name": " Luz ", "tema": "azul"})
        self.assertEqual(r.json(), {"name": "Luz", "tema": "azul"})
        self.assertEqual(self.client.get(self.url(pid, "/config/parameters")).json()["tema"], "azul")
        self.assertEqual(self.client.get(self.url(pid)).json()["name"], "Luz")
        self.client.post(self.url(pid, "/chat"), json={"messages": ["oi de novo"]})
        self.assertIn("Você é Luz", self.actor.calls[-1][0]["content"])  # valeu sem reiniciar
        kept = self.client.put(self.url(pid, "/config/parameters"), json={"x": 1}).json()
        self.assertEqual(kept["name"], "Luz")  # sem 'name': mantém o atual

    def test_parameters_rejects_bad_input(self):
        pid = self.new()
        for body in ({"name": 5}, {"name": ""}, {"name": "x" * 101}):
            self.assertEqual(self.client.put(self.url(pid, "/config/parameters"), json=body).status_code,
                             422, body)
        big = {"name": "ok", "lixo": "x" * 60_000}
        self.assertEqual(self.client.put(self.url(pid, "/config/parameters"), json=big).status_code, 413)
        self.assertEqual(self.client.put(self.url(pid, "/config/parameters"), json=[1]).status_code, 422)

    def test_descriptions_partial_update_reaches_the_prompt(self):
        pid = self.new()
        d = self.client.get(self.url(pid, "/config/descriptions")).json()
        self.assertIn("Curiosa", d["personality"])
        r = self.client.put(self.url(pid, "/config/descriptions"),
                            json={"personality": "Misteriosa e sarcástica.", "appearance": ""})
        self.assertEqual((r.json()["personality"], r.json()["appearance"], r.json()["traits"] == d["traits"]),
                         ("Misteriosa e sarcástica.", "", True))  # vazio limpa; omitido não muda
        self.client.post(self.url(pid, "/chat"), json={"messages": ["oi"]})
        self.assertIn("Misteriosa e sarcástica.", self.actor.calls[-1][0]["content"])


class AssetsTest(ApiCase):
    def upload(self, pid, data, filename="a.png"):
        return self.client.post(self.url(pid, "/config/assets"), files={"file": (filename, data)})

    def test_sniff_by_content(self):
        self.assertEqual([sniff(b) for b in (PNG, b"\xff\xd8\xff\xe0x", b"GIF89a..", b"RIFF1234WEBPxx",
                                              b"<svg onload=alert(1)>", b"")],
                         ["png", "jpg", "gif", "webp", None, None])

    def test_upload_list_get_delete(self):
        pid = self.new()
        r = self.upload(pid, PNG, "meu avatar.exe")  # a extensão enviada é ignorada
        self.assertEqual(r.status_code, 201, r.text)
        name = r.json()["name"]
        self.assertRegex(name, r"^[0-9a-f-]{36}\.png$")
        self.assertEqual([a["name"] for a in self.client.get(self.url(pid, "/config/assets")).json()], [name])
        got = self.client.get(self.url(pid, f"/config/assets/{name}"))
        self.assertEqual((got.content, got.headers["content-type"], got.headers["x-content-type-options"]),
                         (PNG, "image/png", "nosniff"))
        self.assertEqual(self.client.delete(self.url(pid, f"/config/assets/{name}")).status_code, 204)
        self.assertEqual(self.client.get(self.url(pid, f"/config/assets/{name}")).status_code, 404)

    def test_hostile_uploads_and_names(self):
        pid = self.new()
        self.assertEqual(self.upload(pid, b"<svg onload=alert(1)/>", "x.png").status_code, 415)
        self.assertEqual(self.upload(pid, b"#!/bin/sh\nrm -rf /", "x.png").status_code, 415)
        self.assertEqual(self.upload(pid, PNG + b"\x00" * MAX_BYTES).status_code, 413)
        self.assertEqual(self.client.get(self.url(pid, "/config/assets")).json(), [])

    def test_only_generated_names_are_served_or_deleted(self):
        """Arquivo que não veio do upload (nome fora do padrão) é invisível para a API.
        Sem o guard de nome, o GET dava 500 (tipo desconhecido) e o DELETE apagava o arquivo."""
        pid = self.new()
        folder = self.paths.persona_dir(pid) / "assets"
        folder.mkdir()
        (folder / "segredo.txt").write_text("conteúdo")
        self.assertEqual(self.client.get(self.url(pid, "/config/assets/segredo.txt")).status_code, 404)
        self.assertEqual(self.client.delete(self.url(pid, "/config/assets/segredo.txt")).status_code, 404)
        self.assertTrue((folder / "segredo.txt").is_file())
        self.assertEqual(self.client.get(self.url(pid, "/config/assets")).json(), [])  # nem listado

    def test_path_guard_rejects_traversal_names(self):
        """O roteador já não aceita '/' no parâmetro, então o guard é testado direto:
        '../parameters.json' EXISTE, e sem o guard a função devolveria esse caminho."""
        from fastapi import HTTPException
        from src.api.v1.persona.config.assets import _path
        pid = self.new()
        state = self.client.app.state.ctx
        self.assertTrue((self.paths.persona_dir(pid) / "assets" / ".." / "parameters.json").resolve().is_file())
        (self.paths.persona_dir(pid) / "assets").mkdir()
        for name in ("../parameters.json", "..", "a/b.png", "/etc/passwd", "x.png",
                     "00000000-0000-0000-0000-000000000000.png/../../parameters.json"):
            with self.assertRaises(HTTPException, msg=name) as ctx:
                _path(state, pid, name)
            self.assertEqual(ctx.exception.status_code, 404, name)


class MemoryCrudTest(ApiCase):
    def test_events_crud_and_search(self):
        pid = self.new()
        base = self.url(pid, "/config/memory/events")
        a = self.client.post(base, json={"text": "Ana viajou para o Rio", "when": "2026-11-20"}).json()["id"]
        self.client.post(base, json={"text": "Beto jogou futebol"})
        self.assertEqual([e["text"] for e in self.client.get(base).json()],
                         ["Beto jogou futebol", "Ana viajou para o Rio"])
        found = self.client.get(base, params={"q": "rio viagem"}).json()
        self.assertEqual([(e["text"], e["when"]) for e in found], [("Ana viajou para o Rio", "2026-11-20")])
        self.assertIsNone(found[0]["last_accessed_at"])  # buscar pela API não conta como acesso
        self.assertEqual(self.client.delete(f"{base}/{a}").status_code, 204)
        self.assertEqual(self.client.delete(f"{base}/{a}").status_code, 404)
        self.assertEqual(self.client.post(base, json={"text": ""}).status_code, 422)
        self.assertEqual(self.client.get(base, params={"limit": 0}).status_code, 422)

    def test_knowledge_crud_and_graph_search(self):
        pid = self.new()
        base = self.url(pid, "/config/memory/knowledge")
        fact = self.client.post(base, json={"subject": "Ana", "relation": "Mora em", "object": "Lisboa"}).json()
        self.assertEqual((fact["relation"], fact["weight"]), ("mora em", 1))
        self.client.post(base, json={"subject": "Lisboa", "relation": "fica em", "object": "Portugal"})
        hits = self.client.get(base, params={"q": "Ana"}).json()
        self.assertEqual([(f["subject"], f["object"]) for f in hits],
                         [("Ana", "Lisboa"), ("Lisboa", "Portugal")])
        self.assertEqual(self.client.delete(f"{base}/{fact['id']}").status_code, 204)
        self.assertEqual(self.client.delete(f"{base}/{fact['id']}").status_code, 404)
        self.assertEqual(self.client.post(base, json={"subject": "A", "relation": "", "object": "B"}).status_code, 422)

    def test_notes_crud(self):
        pid = self.new()
        base = self.url(pid, "/config/memory/notes")
        nid = self.client.post(base, json={"text": "Gostei muito da visita", "subject": "Ana",
                                           "importance": 0.9}).json()["id"]
        note = self.client.get(base, params={"q": "visita"}).json()[0]
        self.assertEqual((note["subject"], note["importance"]), ("Ana", 0.9))
        self.assertEqual(self.client.post(base, json={"text": "x", "importance": 2}).status_code, 422)
        self.assertEqual(self.client.delete(f"{base}/{nid}").status_code, 204)
        self.assertEqual(self.client.get(base).json(), [])

    def test_messages_delete(self):
        pid = self.new()
        self.client.post(self.url(pid, "/chat"), json={"messages": ["oi"]})
        base = self.url(pid, "/config/memory/messages")
        first = self.client.get(base).json()[0]["id"]
        self.assertEqual(self.client.delete(f"{base}/{first}").status_code, 204)
        self.assertEqual(self.client.delete(f"{base}/{first}").status_code, 404)
        self.assertEqual(len(self.client.get(base, params={"limit": 1}).json()), 1)


class BackgroundApiTest(ApiCase):
    background = True

    def test_finalize_runs_pending_background_work(self):
        pid = self.new()
        self.client.post(self.url(pid, "/chat"), json={"messages": ["oi"]})  # 2 msgs: nada devido ainda
        before = self.client.get(self.url(pid, "/status")).json()
        self.assertEqual(before["summary"], "")
        self.assertEqual(self.client.post(self.url(pid, "/finalize")).status_code, 204)
        after = self.client.get(self.url(pid, "/status")).json()
        self.assertEqual((after["summary"], after["state"], after["scene"]),
                         ("Resumo da API", "Tranquila", {"local": "sala"}))

    def test_shutdown_flushes_background(self):
        pid = self.new()
        self.client.post(self.url(pid, "/chat"), json={"messages": ["oi"]})
        self.client.__exit__(None, None, None)  # fecha a app: o lifespan faz o flush
        self.assertEqual((self.paths.persona_dir(pid) / "summary.md").read_text().strip(), "Resumo da API")
        self.client.__enter__()  # o tearDown espera a app aberta


if __name__ == "__main__":
    unittest.main()
