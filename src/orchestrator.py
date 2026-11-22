# O orquestrador da persona: carrega as memórias, instancia as emoções, monta o contexto,
# envia/recebe os prompts. Não é a main e não tem rotas: só a lógica da persona
# (a API e o CLI chamam `chat`).
#
# Um turno (chat):
#   1. o interpretador lê a mensagem + emoções e gera consultas de memória, tom e variação emocional
#   2. as emoções são atualizadas; todas as buscas de memória rodam em paralelo
#   3. core/actor monta o prompt e o actor responde
#   4. o turno vai para a flash (com a variação emocional) e o registro de personas é atualizado
#   5. em background (core/pipeline): notetaker, summary e consolidator, quando há mensagens novas
#      suficientes; `finalize` processa o que faltar ao fim da conversa
# Turnos da mesma persona são serializados (lock). Se o turno falha, as emoções voltam ao que eram.
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Mapping, Sequence

from src.core.actor import ActorContext, ActorPromptBuilder
from src.core.emotions import EmotionDictionary, EmotionParams, EmotionState, EmotionStore
from src.core.interpreter import Interpreter, InterpreterContext, InterpreterPromptBuilder
from src.core.consolidator import Consolidator
from src.core.memories import MemoryManager
from src.core.persona_files import PersonaFiles
from src.core.pipeline import BackgroundPipeline, Job
from src.data.database import Database
from src.data.personas import PersonaRegistry
from src.engine.base.consolidator import ConsolidatorEngine
from src.engine.base.notetaker import Notetaker
from src.engine.base.summary import Summarizer
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
                 files: PersonaFiles, pipeline: BackgroundPipeline | None = None,
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
        self.files = files
        self.pipeline = pipeline
        self.registry = registry
        self._databases = list(databases)
        self._lock = asyncio.Lock()
        self.log = get_logger("orchestrator", "core")

    @classmethod
    def build(cls, persona_id: str, config: Config | None = None,
              models: Mapping[str, Model] | None = None,
              stores: Sequence[MemoryStore] | None = None, use_registry: bool = True,
              setup_logs: bool = True, background: bool = True) -> "Orchestrator":
        """models: modelos por componente ("actor", "primary_interpreter", "notetaker", "summary",
        "consolidator"); o que faltar vem das configs (provider local/nuvem).
        stores: None = memórias reais da persona. background: liga notetaker/summary/consolidator."""
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

        files = PersonaFiles(persona_dir)
        params = config.persona_parameters(persona_id)  # dict compartilhado: o nome vale em todo lugar
        pipeline = None
        if background:
            pipeline = cls._build_pipeline(config, flash, list(stores), files, params, emotions,
                                           {k: v["label"] for k, v in dictionary.primary.items()},
                                           model_for)

        registry = None
        if use_registry:
            registry_db = Database(paths.global_db)
            databases.append(registry_db)
            registry = PersonaRegistry(registry_db)
        return cls(persona_id, persona_dir, params, flash, builder, Actor(model_for("actor")),
                   interpreter, memory, emotions, emotion_store, files, pipeline, registry, databases)

    @staticmethod
    def _build_pipeline(config: Config, flash: FlashMemory, stores: Sequence[MemoryStore],
                        files: PersonaFiles, params: dict, emotions: EmotionState,
                        labels: dict[str, str], model_for) -> BackgroundPipeline | None:
        def name() -> str:
            return str(params.get("name") or "Persona")

        def job(cfg: dict, key: str, run) -> Job:
            return Job(key, f"{key}_up_to", run, int(cfg.get("every_messages", 6)),
                       int(cfg.get("max_batch", 40)))

        jobs: list[Job] = []
        cfg = config.engine("summary")
        if cfg.get("enabled", True):
            summarizer = Summarizer(model_for("summary"), config.prompt("summary"), files, name,
                                    emotions.describe, int(cfg.get("summary_max_chars", 2000)))
            jobs.append(job(cfg, "summary", summarizer.update))

        notes = next((s for s in stores if isinstance(s, NotesMemory)), None)
        cfg = config.engine("notetaker")
        if notes and cfg.get("enabled", True):
            notetaker = Notetaker(model_for("notetaker"), config.prompt("notetaker"), name,
                                  files.persona, labels, float(cfg.get("min_significance", 0.3)),
                                  float(cfg.get("block_gap_minutes", 30)))

            async def write_notes(messages) -> None:
                for note in await notetaker.take_notes(messages):
                    await asyncio.to_thread(notes.add, note.text, note.subject, note.importance,
                                            note.emotions or None)

            jobs.append(job(cfg, "notes", write_notes))

        events = next((s for s in stores if isinstance(s, EventsMemory)), None)
        knowledge = next((s for s in stores if isinstance(s, KnowledgeMemory)), None)
        cfg = config.engine("consolidator")
        if events and knowledge and cfg.get("enabled", True):
            consolidator = Consolidator(ConsolidatorEngine(model_for("consolidator")),
                                        config.prompt("consolidator"), events, knowledge, name,
                                        int(cfg.get("similar_k", 5)))
            jobs.append(job(cfg, "consolidated", consolidator.consolidate))
        return BackgroundPipeline(flash, jobs) if jobs else None

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
            persona, state, scene = self.files.persona(), self.files.state, self.files.scene()
            summary = self.files.summary
            interp = await self.interpreter.interpret(InterpreterContext(
                self.name, texts, persona, self.emotions.describe(), state, scene, summary))

            change = self.emotions.apply(interp.emotion_delta) if interp.emotion_delta else {}
            results = await self.memory.search_all(interp.queries)  # todas em paralelo

            messages = self.builder.build(ActorContext(  # antes de gravar o turno na flash
                persona_name=self.name, user_messages=texts, persona=persona,
                emotions=self.emotions.describe(), tone=interp.tone, state=state, scene=scene,
                summary=summary, memory=self.memory.build_prompt(results)))
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
        if self.pipeline:
            self.pipeline.kick()  # notetaker/summary/consolidator, sem bloquear a resposta
        self.log.info("turno concluído (%d msgs, resposta com %d chars, emoções %s)",
                      len(texts), len(reply), change)
        return reply

    async def finalize(self) -> None:
        """Fim da conversa: termina o background e processa as mensagens que ainda faltam."""
        if self.pipeline:
            await self.pipeline.flush()
        self.close()

    def close(self) -> None:
        for db in self._databases:
            db.close()
