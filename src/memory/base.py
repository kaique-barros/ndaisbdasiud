# Contrato das memórias pesquisáveis (events, knowledge, notes).
# Cada uma implementa `search` de forma assíncrona para que o core/memories
# consiga disparar todas as buscas em paralelo.
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryStore(Protocol):
    name: str  # usado como chave das consultas e como título da seção no prompt

    async def search(self, query: str, top_k: int) -> list[str]:
        """Devolve até top_k trechos já formatados como texto para o prompt."""
        ...


@dataclass(frozen=True)
class MemoryResult:
    source: str
    items: list[str]
    error: str | None = None
    elapsed_s: float = 0.0
