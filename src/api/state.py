# Estado compartilhado da API: configuração, registro de personas e um Orchestrator por persona
# (aberto sob demanda e mantido aberto, para o background e o lock de turnos funcionarem).
from __future__ import annotations

import json
import threading
import uuid
from typing import Any, Mapping

from fastapi import HTTPException

from src.core.persona_files import PersonaFiles
from src.data.database import Database
from src.data.personas import PersonaRecord, PersonaRegistry
from src.engine.model import Model
from src.orchestrator import Orchestrator
from src.utils.config import Config
from src.utils.logger import get_logger


class AppState:
    def __init__(self, config: Config, models: Mapping[str, Model] | None = None,
                 background: bool = True) -> None:
        self.config = config
        self.models = dict(models or {})
        self.background = background
        self.registry_db = Database(config.paths.global_db)
        self.registry = PersonaRegistry(self.registry_db)
        self._orchestrators: dict[str, Orchestrator] = {}
        self._lock = threading.Lock()
        self.log = get_logger("api", "api")

    def record(self, persona_id: str) -> PersonaRecord:
        """Só ids UUID registrados chegam ao disco: nada de path traversal via URL."""
        try:
            uuid.UUID(persona_id)
        except ValueError:
            raise HTTPException(404, "persona não encontrada") from None
        record = self.registry.get(persona_id)
        if record is None:
            raise HTTPException(404, "persona não encontrada")
        return record

    def orchestrator(self, persona_id: str) -> Orchestrator:
        self.record(persona_id)
        with self._lock:
            orch = self._orchestrators.get(persona_id)
            if orch is None:
                orch = Orchestrator.build(persona_id, self.config, self.models,
                                          background=self.background, setup_logs=False)
                self._orchestrators[persona_id] = orch
            return orch

    def files(self, persona_id: str) -> PersonaFiles:
        self.record(persona_id)
        return PersonaFiles(self.config.paths.persona_dir(persona_id))

    def write_parameters(self, persona_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Grava parameters.json e mantém registro e Orchestrator aberto em sincronia com o nome."""
        record = self.record(persona_id)
        params = dict(params)
        name = params.get("name", record.name)
        if not isinstance(name, str) or not name.strip() or len(name) > 100:
            raise HTTPException(422, "'name' deve ser um texto de 1 a 100 caracteres")
        params["name"] = name = name.strip()
        try:
            encoded = json.dumps(params, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            raise HTTPException(422, "parâmetros devem ser JSON válido") from None
        if len(encoded) > 50_000:
            raise HTTPException(413, "parâmetros grandes demais")
        self.files(persona_id).write_text("parameters.json", encoded)
        if name != record.name:
            self.registry.rename(persona_id, name)
        with self._lock:
            live = self._orchestrators.get(persona_id)
        if live is not None:
            live.params.clear()
            live.params.update(params)
        return params

    async def shutdown(self) -> None:
        for persona_id, orch in list(self._orchestrators.items()):
            try:
                await orch.finalize()  # termina o background e fecha os bancos
            except Exception:
                self.log.exception("falha ao finalizar %s", persona_id)
        self.registry_db.close()
