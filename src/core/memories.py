# Gerencia as memórias e gera o prompt de memória.
# Todas as buscas (cada memória x cada consulta) rodam em paralelo com asyncio.gather.
# Falha ou timeout em uma busca não derruba as outras: vira um MemoryResult com `error`.
# Logs: eventos em logs/core/memories.log; conteúdo do prompt em memories.prompt.log.
from __future__ import annotations

import asyncio
import time
from typing import Mapping, Sequence

from src.memory.base import MemoryResult, MemoryStore
from src.utils.logger import get_logger, get_prompt_logger


class MemoryManager:
    def __init__(self, stores: Sequence[MemoryStore], top_k: int = 5,
                 timeout_s: float = 10.0,
                 per_store: Mapping[str, Mapping[str, float]] | None = None) -> None:
        names = [s.name for s in stores]
        if len(set(names)) != len(names):
            raise ValueError(f"nomes de memória duplicados: {names}")
        self.stores = {s.name: s for s in stores}
        self.top_k = top_k
        self.timeout_s = timeout_s
        self.per_store = per_store or {}  # {"events": {"top_k": 3, "timeout_s": 5}, ...}
        self.log = get_logger("memories", "core")
        self.prompt_log = get_prompt_logger("memories", "core")

    async def search_all(self, queries: Mapping[str, Sequence[str]]) -> dict[str, MemoryResult]:
        """queries: nome da memória -> consultas geradas pelo interpretador.
        Memórias sem consulta são ignoradas. Retorna um resultado por memória consultada."""
        jobs: list[tuple[str, str]] = []
        for name, qs in queries.items():
            if name not in self.stores:
                self.log.warning("memória desconhecida ignorada: %s", name)
                continue
            jobs.extend((name, q) for q in qs)

        started = time.perf_counter()
        outcomes = await asyncio.gather(*(self._one(n, q) for n, q in jobs))
        merged: dict[str, MemoryResult] = {}
        for name, items, error, elapsed in outcomes:
            prev = merged.get(name)
            all_items = list(prev.items) if prev else []
            all_items.extend(i for i in items if i not in all_items)
            merged[name] = MemoryResult(
                source=name,
                items=all_items,
                error=error or (prev.error if prev else None),
                elapsed_s=max(elapsed, prev.elapsed_s if prev else 0.0),
            )
        self.log.info("%d buscas em %d memórias em %.3fs", len(jobs), len(merged),
                      time.perf_counter() - started)
        return merged

    async def _one(self, name: str, query: str) -> tuple[str, list[str], str | None, float]:
        t0 = time.perf_counter()
        cfg = self.per_store.get(name, {})
        top_k, timeout_s = int(cfg.get("top_k", self.top_k)), float(cfg.get("timeout_s", self.timeout_s))
        try:
            items = await asyncio.wait_for(self.stores[name].search(query, top_k),
                                           timeout=timeout_s)
            return name, list(items), None, time.perf_counter() - t0
        except asyncio.TimeoutError:
            self.log.warning("timeout em %s (query=%r)", name, query)
            return name, [], f"timeout após {timeout_s}s", time.perf_counter() - t0
        except Exception as exc:  # isolamento: uma memória quebrada não derruba as outras
            self.log.exception("erro em %s (query=%r)", name, query)
            return name, [], f"{type(exc).__name__}: {exc}", time.perf_counter() - t0

    def build_prompt(self, results: Mapping[str, MemoryResult]) -> str:
        sections = []
        for name, res in results.items():
            if res.items:
                body = "\n".join(f"- {item}" for item in res.items)
                sections.append(f"## {name}\n{body}")
        prompt = "\n\n".join(sections)
        self.prompt_log.info("%s", prompt)
        return prompt
