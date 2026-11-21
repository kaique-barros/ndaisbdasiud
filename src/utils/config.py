# Carrega as configurações. Deve ser instanciado só no orchestrator, que repassa
# para cada classe apenas o dicionário de config de que ela precisa.
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Erro ao carregar ou interpretar um arquivo de configuração."""


def project_root() -> Path:
    """Raiz do projeto; pode ser sobrescrita pela variável PERSONA_ROOT."""
    env = os.environ.get("PERSONA_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def configs(self) -> Path:
        return self.root / ".configs"

    @property
    def data(self) -> Path:
        return self.root / ".data"

    @property
    def logs(self) -> Path:
        return self.root / ".logs"

    @property
    def global_db(self) -> Path:
        return self.data / "global.db"

    def persona_dir(self, persona_id: str) -> Path:
        return self.data / "personas" / persona_id

    def messages_db(self, persona_id: str) -> Path:
        return self.persona_dir(persona_id) / "memories" / "messages.db"


def load_json(path: Path) -> dict[str, Any]:
    """Lê um JSON. Arquivo inexistente é erro; arquivo vazio vira {}."""
    if not path.is_file():
        raise ConfigError(f"arquivo de configuração não encontrado: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"JSON inválido em {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} deve conter um objeto JSON")
    return data


class Config:
    """Acesso às configs em .configs/ (engine, providers, prompts, appearance)."""

    def __init__(self, paths: Paths | None = None) -> None:
        self.paths = paths or Paths(project_root())

    def engine(self, name: str) -> dict[str, Any]:
        return load_json(self.paths.configs / "engine" / f"{name}.json")

    def provider(self, name: str) -> dict[str, Any]:
        return load_json(self.paths.configs / "providers" / f"{name}.json")

    def appearance(self) -> dict[str, Any]:
        return load_json(self.paths.configs / "appearance" / "config.json")

    def prompt(self, name: str) -> str:
        path = self.paths.configs / "prompts" / f"{name}.md"
        if not path.is_file():
            raise ConfigError(f"prompt não encontrado: {path}")
        return path.read_text(encoding="utf-8")

    def persona_parameters(self, persona_id: str) -> dict[str, Any]:
        return load_json(self.paths.persona_dir(persona_id) / "parameters.json")
