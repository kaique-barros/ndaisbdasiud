# Interface de linha de comando para conversar com uma persona (a API HTTP fica em src/api).
#   python -m src.cli --new "Nome"      cria uma persona a partir de persona_example
#   python -m src.cli --list            lista as personas (mais recentes primeiro)
#   python -m src.cli <uuid>            conversa com a persona (/sair para encerrar)
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from typing import Callable

from src.data.database import Database
from src.data.personas import PersonaRegistry, create_persona
from src.engine.main.actor import ActorError
from src.orchestrator import Orchestrator
from src.utils.bridge import BridgeError
from src.utils.config import Config, ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="persona", description="Motor de persona conversacional")
    parser.add_argument("persona_id", nargs="?", help="uuid da persona para conversar")
    parser.add_argument("--new", metavar="NOME", help="cria uma persona nova")
    parser.add_argument("--list", action="store_true", help="lista as personas")
    args = parser.parse_args(argv)

    config = Config()
    if args.new or args.list:
        with Database(config.paths.global_db) as db:
            registry = PersonaRegistry(db)
            if args.new:
                record = create_persona(registry, config.paths, args.new)
                print(f"persona criada: {record.name}\nuuid: {record.id}")
                print(f"edite as descrições em {config.paths.persona_dir(record.id)}/descriptions")
            else:
                for p in registry.list():
                    print(f"{p.id}  {p.name}  [{p.status}]  última conversa: "
                          f"{p.last_conversation_at or '-'}")
        return 0
    if not args.persona_id:
        parser.print_help()
        return 2
    return _chat_loop(config, args.persona_id)


def _stdin_line() -> str:
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line


def _chat_loop(config: Config, persona_id: str, models=None, read: Callable[[], str] = _stdin_line) -> int:
    try:
        return asyncio.run(_chat(config, persona_id, models, read))
    except KeyboardInterrupt:
        print("\ninterrompido", file=sys.stderr)
        return 130


async def _chat(config: Config, persona_id: str, models, read: Callable[[], str]) -> int:
    """Um único event loop para a sessão inteira: o lock de turnos e o background dependem disso.
    O teclado é lido em uma thread daemon (não trava a saída) e entregue por uma fila, assim o
    background continua rodando enquanto você digita."""
    try:
        orch = Orchestrator.build(persona_id, config, models)
    except ConfigError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    loop = asyncio.get_running_loop()
    lines: asyncio.Queue[str | None] = asyncio.Queue()

    def pump() -> None:
        while True:
            try:
                line = read()
            except EOFError:
                loop.call_soon_threadsafe(lines.put_nowait, None)
                return
            loop.call_soon_threadsafe(lines.put_nowait, line)

    threading.Thread(target=pump, daemon=True).start()
    print(f"conversando com {orch.name} (/status mostra as emoções, /sair encerra)\n")
    try:
        while True:
            print("você> ", end="", flush=True)
            line = await lines.get()
            if line is None:  # fim da entrada (Ctrl-D ou pipe)
                print()
                break
            text = line.strip()
            if text in ("/sair", "/exit"):
                break
            if text == "/status":
                print(_status(orch))
                continue
            if not text:
                continue
            try:
                print(f"{orch.name}> {await orch.chat([text])}\n")
            except (BridgeError, ActorError) as exc:
                print(f"[falha ao falar com o modelo: {exc}]\n"
                      "Verifique PERSONA_PROVIDER, o `ollama serve` (local) ou a GROQ_API_KEY (nuvem).\n",
                      file=sys.stderr)
    finally:
        print("salvando memórias...", file=sys.stderr)
        await orch.finalize()  # processa o que o background ainda não processou
    return 0


def _status(orch: Orchestrator) -> str:
    parts = ["emoções:\n" + orch.emotion_store.load().describe()]
    if orch.files.summary:
        parts.append("resumo:\n" + orch.files.summary)
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
