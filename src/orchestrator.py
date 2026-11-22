# O orquestrador da persona: carrega as memórias, instancia as emoções, monta o contexto,
# envia/recebe os prompts. Não é a main e não tem rotas: só a lógica da persona
# (a API e o CLI chamam `chat`).
#
# Um turno (chat):
#   1. o interpretador lê a mensagem + emoções e gera consultas de memória, tom e variação emocional
#   2. as emoções são atualizadas; todas as buscas de memória rodam em paralelo
#   3. core/actor monta o prompt e o actor responde
#   4. o turno vai para a flash (com a variação emocional) e o registro de personas é atualizado
# Turnos da mesma persona são serializados (lock). Se o turno falha, as emoções voltam ao que eram.
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Mapping, Sequence

from src.core.actor import ActorContext, ActorPromptBuilder
from src.core.emotions import EmotionDictionary, EmotionParams, EmotionState, EmotionStore
from src.core.interpreter import Interpreter, InterpreterContext, InterpreterPromptBuilder
from src.core.memories import MemoryManager
from src.data.database import Database
from src.data.personas import PersonaRegistry
from src.engine.main.actor import Actor
from src.engine.main.primary_interpreter import PrimaryInterpreter
from src.engine.model import ChatModel, Model, make_embedder
from src.memory.base import MemoryStore
from src.memory.events import EventsMemory
from src.memory.flash import FlashMemory
from src.memory.knowledge import KnowledgeMemory
from src.memory.notes import NotesMemory
from src.utils.config import Config, ConfigError
from src.utils.logger import get_logger, setup_logging


class Orchestrator:
    def __init__(self, persona_id: str, persona_dir: Path, params: dict, flash: FlashMemory,
                 builder: ActorPromptBuilder, actor: Actor, interpreter: Interpreter,
                 memory: MemoryManager, emotions: EmotionState, emotion_store: EmotionStore,
                 registry: PersonaRegistry | None = None,
                 databases: Sequence[Database] = ()) -> None:
        self.persona_id = persona_id
        self.persona_dir = persona_dir
        self.params = params
        self.flash = flash
        self.builder = builder
        self.actor = actor
        self.interpreter = interpreter
        self.memory = memory
        self.emotions = emotions
        self.emotion_store = emotion_store
        self.registry = registry
        self._databases = list(databases)
        self._lock = asyncio.Lock()
        self.log = get_logger("orchestrator", "core")

    @classmethod
    def build(cls, persona_id: str, config: Config | None = None,
              models: Mapping[str, Model] | None = None,
              stores: Sequence[MemoryStore] | None = None, use_registry: bool = True,
              setup_logs: bool = True) -> "Orchestrator":
        """models: modelos por componente ("actor", "primary_interpreter"); o que faltar vem
        das configs (provider local/nuvem). stores: None = memórias reais da persona."""
        config = config or Config()
        paths = config.paths
        persona_dir = paths.persona_dir(persona_id)
        if not persona_dir.is_dir():
            raise ConfigError(f"persona não encontrada: {persona_id} ({persona_dir})")
        if setup_logs:
            setup_logging(paths.logs)
        models = dict(models or {})

        def model_for(name: str) -> Model:
            return models.get(name) or ChatModel.from_name(name, config)

        databases: list[Database] = []

        def open_db(name: str) -> Database:
            db = Database(paths.memory_db(persona_id, name))
            databases.append(db)
            return db

        flash = FlashMemory(open_db("messages"), k=int(config.engine("flash_memory").get("k", 20)))

        memory_cfgs = {n: config.engine(f"{n}_memory") for n in ("events", "knowledge", "notes")}
        if stores is None:
            def embedder(name: str):
                model_id = memory_cfgs[name].get("embedding_model")
                return make_embedder(config, model_id) if model_id else None

            stores = [EventsMemory(open_db("events"), embedder("events")),
                      KnowledgeMemory(open_db("knowledge")),
                      NotesMemory(open_db("notes"), embedder("notes"))]
        memory = MemoryManager(list(stores), per_store={
            n: {k: v for k, v in cfg.items() if k in ("top_k", "timeout_s")}
            for n, cfg in memory_cfgs.items()})

        dictionary = EmotionDictionary(config.engine("emotions_description"))
        emotion_store = EmotionStore(persona_dir / "emotions.json", dictionary,
                                     EmotionParams.from_config(config.engine("emotions")))
        emotions = emotion_store.load()

        actor_cfg, interp_cfg = config.engine("actor"), config.engine("primary_interpreter")
        builder = ActorPromptBuilder(config.prompt("actor"), flash,
                                     int(actor_cfg.get("max_prompt_chars", 24000)))
        store_names = list(memory.stores)
        interpreter = Interpreter(
            InterpreterPromptBuilder(
                config.prompt("primary_interpreter"), flash, store_names,
                {k: v["label"] for k, v in dictionary.primary.items()},
                int(interp_cfg.get("max_prompt_chars", 12000)), int(interp_cfg.get("flash_k", 6))),
            PrimaryInterpreter(model_for("primary_interpreter")), store_names,
            enabled=bool(interp_cfg.get("enabled", True)))

        registry = None
        if use_registry:
            registry_db = Database(paths.global_db)
            databases.append(registry_db)
            registry = PersonaRegistry(registry_db)
        return cls(persona_id, persona_dir, config.persona_parameters(persona_id), flash, builder,
                   Actor(model_for("actor")), interpreter, memory, emotions, emotion_store,
                   registry, databases)

    @property
    def name(self) -> str:
        return str(self.params.get("name") or "Persona")

    async def chat(self, user_messages: Sequence[str], conversation_id: str | None = None) -> str:
        """Um turno: recebe a(s) mensagem(ns) do usuário e devolve a fala da persona."""
        texts = [m.strip() for m in user_messages if m.strip()]
        if not texts:
            raise ValueError("mensagem vazia")
        async with self._lock:
            return await self._turn(texts, conversation_id)

    async def _turn(self, texts: list[str], conversation_id: str | None) -> str:
        snapshot = self.emotions.snapshot()
        try:
            self.emotions.decay()
            persona, state, scene = self._read_persona(), self._read_text("state.md"), self._read_scene()
            interp = await self.interpreter.interpret(InterpreterContext(
                self.name, texts, persona, self.emotions.describe(), state, scene))

            change = self.emotions.apply(interp.emotion_delta) if interp.emotion_delta else {}
            results = await self.memory.search_all(interp.queries)  # todas em paralelo

            messages = self.builder.build(ActorContext(  # antes de gravar o turno na flash
                persona_name=self.name, user_messages=texts, persona=persona,
                emotions=self.emotions.describe(), tone=interp.tone, state=state, scene=scene,
                memory=self.memory.build_prompt(results)))
            reply = await self.actor.respond(messages)
        except BaseException:
            self.emotions.restore(snapshot)  # turno que falha não deixa rastro emocional
            raise

        self.emotion_store.save(self.emotions)
        for i, text in enumerate(texts):  # a variação emocional fica na última mensagem do turno
            self.flash.add("user", text, conversation_id, change if i == len(texts) - 1 else None)
        self.flash.add("persona", reply, conversation_id)
        if self.registry:
            self.registry.touch_conversation(self.persona_id)
        self.log.info("turno concluído (%d msgs, resposta com %d chars, emoções %s)",
                      len(texts), len(reply), change)
        return reply

    def close(self) -> None:
        for db in self._databases:
            db.close()

    # --- leitura dos arquivos da persona (lidos a cada turno: edições valem na hora) ---

    def _read_text(self, relative: str) -> str:
        path = self.persona_dir / relative
        return path.read_text(encoding="utf-8").strip() if path.is_file() else ""

    def _read_persona(self) -> str:
        sections = []
        for title, filename in (("Personalidade", "personality.md"), ("Traços", "traits.md"),
                                ("Aparência", "appearance.md")):
            text = self._read_text(f"descriptions/{filename}")
            if text:
                sections.append(f"## {title}\n{text}")
        return "\n\n".join(sections)

    def _read_scene(self) -> str:
        raw = self._read_text("scene.json")
        if not raw:
            return ""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(data, dict):
            return "\n".join(f"- {k}: {v}" for k, v in data.items() if v not in ("", None))
        return raw
