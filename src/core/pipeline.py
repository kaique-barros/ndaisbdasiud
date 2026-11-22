# Processos em background depois de cada turno: notetaker, summary e consolidator.
#
# Cada processo lê as mensagens novas da flash a partir do SEU marcador (guardado no messages.db,
# então sobrevive a reinícios) e roda quando acumulou `threshold` mensagens novas, ou no
# `flush` (fim da conversa). O marcador só avança quando o processo termina com sucesso:
#   - falha de rede/modelo -> o lote é refeito no próximo gatilho (nada se perde);
#   - saída inválida do modelo (ValueError) 3 vezes seguidas -> o lote é descartado (poison batch),
#     para um lote ruim não travar o processo para sempre.
# Só uma rodada roda por vez; um `kick` durante a rodada agenda outra ao terminar.
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from src.memory.flash import FlashMemory, Message
from src.utils.logger import get_logger

MAX_BAD_OUTPUTS = 3


@dataclass
class Job:
    name: str
    marker: str                                   # chave em flash.meta (ex.: "notes_up_to")
    run: Callable[[Sequence[Message]], Awaitable[object]]
    threshold: int = 6
    max_batch: int = 40
    failures: int = 0


class BackgroundPipeline:
    def __init__(self, flash: FlashMemory, jobs: Sequence[Job]) -> None:
        self.flash = flash
        self.jobs = list(jobs)
        self._task: asyncio.Task | None = None
        self._again = False
        self.log = get_logger("pipeline", "core")

    def kick(self) -> None:
        """Dispara uma rodada em background (não bloqueia o turno)."""
        if self._task and not self._task.done():
            self._again = True
            return
        self._task = asyncio.create_task(self._loop())

    async def wait(self) -> None:
        """Espera a rodada em andamento (se houver)."""
        while self._task and not self._task.done():
            await asyncio.shield(self._task)

    async def flush(self) -> None:
        """Fim da conversa: termina a rodada atual e processa tudo que ainda falta."""
        await self.wait()
        while await self.run_due(force=True):
            pass

    async def _loop(self) -> None:
        try:
            while True:
                self._again = False
                await self.run_due()
                if not self._again:
                    return
        except Exception:  # nunca deixa o erro virar "Task exception was never retrieved"
            self.log.exception("rodada de background falhou")

    async def run_due(self, force: bool = False) -> bool:
        """Roda os processos devidos em paralelo. Devolve True se algum avançou (para o flush
        repetir quando o lote máximo não esgotou as mensagens)."""
        due = [j for j in self.jobs
               if self.flash.count_since(self.flash.get_meta(j.marker))
               >= (1 if force else j.threshold)]
        results = await asyncio.gather(*(self._run_job(j) for j in due))
        return any(results)

    async def _run_job(self, job: Job) -> bool:
        after = self.flash.get_meta(job.marker)
        batch = self.flash.since(after, job.max_batch)
        if not batch:
            return False
        try:
            await job.run(batch)
        except ValueError as exc:  # saída inválida do modelo
            job.failures += 1
            self.log.warning("%s: saída inválida (%d/%d): %s", job.name, job.failures,
                             MAX_BAD_OUTPUTS, exc)
            if job.failures < MAX_BAD_OUTPUTS:
                return False
            self.log.error("%s: lote descartado após %d saídas inválidas", job.name, job.failures)
        except Exception as exc:  # rede, modelo fora do ar...: tenta de novo no próximo gatilho
            self.log.warning("%s falhou (%s: %s); será repetido", job.name, type(exc).__name__, exc)
            return False
        job.failures = 0
        self.flash.set_meta(job.marker, batch[-1].id)
        return True
