# O orquestrador da persona: carrega as memórias, monta o contexto, envia/recebe os prompts.
# Não é a main e não tem rotas: só a lógica da persona (a API/CLI chamam `chat`).
#
# Fluxo atual de um turno (o que falta entra nos pontos marcados):
#   1. buscas nas memórias, todas em paralelo       [consultas ainda = texto do usuário;
#                                                    virão do primary_interpreter]
#   2. monta o prompt (core/actor) e chama o actor  [emoções e resumo entram aqui depois]
#   3. grava o turno na flash e atualiza o registro [notetaker/summary/consolidator em background]
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from src.core.actor import ActorContext, ActorPromptBuilder
from src.core.memories import MemoryManager
from src.data.database import Database
from src.data.personas import PersonaRegistry
from src.engine.main.actor import Actor
from src.engine.model import ChatModel, Model
from src.memory.base import MemoryStore
from src.memory.flash import FlashMemory
from src.utils.config import Config, ConfigError
from src.utils.logger import get_logger, setup_logging


class Orchestrator:
    def __init__(self, persona_id: str, persona_dir: Path, params: dict, flash: FlashMemory,
                 builder: ActorPromptBuilder, actor: Actor, memory: MemoryManager,
                 registry: PersonaRegistry | None = None) -> None:
        self.persona_id = persona_id
        self.persona_dir = persona_dir
        self.params = params
        self.flash = flash
        self.builder = builder
        self.actor = actor
        self.memory = memory
        self.registry = registry
        self.log = get_logger("orchestrator", "core")

    @classmethod
    def build(cls, persona_id: str, config: Config | None = None, model: Model | None = None,
              stores: Sequence[MemoryStore] = (), use_registry: bool = True,
              setup_logs: bool = True) -> "Orchestrator":
        config = config or Config()
        paths = config.paths
        persona_dir = paths.persona_dir(persona_id)
        if not persona_dir.is_dir():
            raise ConfigError(f"persona não encontrada: {persona_id} ({persona_dir})")
        if setup_logs:
            setup_logging(paths.logs)

        actor_cfg = config.engine("actor")
        flash = FlashMemory(Database(paths.messages_db(persona_id)),
                            k=int(config.engine("flash_memory").get("k", 20)))
        builder = ActorPromptBuilder(config.prompt("actor"), flash,
                                     int(actor_cfg.get("max_prompt_chars", 24000)))
        actor = Actor(model or ChatModel.from_name("actor", config))
        memory = MemoryManager(list(stores))
        registry = PersonaRegistry(Database(paths.global_db)) if use_registry else None
        return cls(persona_id, persona_dir, config.persona_parameters(persona_id), flash,
                   builder, actor, memory, registry)

    @property
    def name(self) -> str:
        return str(self.params.get("name") or "Persona")

    async def chat(self, user_messages: Sequence[str], conversation_id: str | None = None) -> str:
        """Um turno: recebe a(s) mensagem(ns) do usuário e devolve a fala da persona."""
        texts = [m.strip() for m in user_messages if m.strip()]
        if not texts:
            raise ValueError("mensagem vazia")

        queries = {store: ["\n".join(texts)] for store in self.memory.stores}
        results = await self.memory.search_all(queries)

        ctx = ActorContext(
            persona_name=self.name,
            user_messages=texts,
            persona=self._read_persona(),
            state=self._read_text("state.md"),
            scene=self._read_scene(),
            memory=self.memory.build_prompt(results),
        )
        messages = self.builder.build(ctx)  # antes de gravar o turno na flash
        reply = await self.actor.respond(messages)

        for text in texts:
            self.flash.add("user", text, conversation_id)
        self.flash.add("persona", reply, conversation_id)
        if self.registry:
            self.registry.touch_conversation(self.persona_id)
        self.log.info("turno concluído (%d msgs do usuário, resposta com %d chars)",
                      len(texts), len(reply))
        return reply

    def close(self) -> None:
        self.flash.db.close()
        if self.registry:
            self.registry.db.close()

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
