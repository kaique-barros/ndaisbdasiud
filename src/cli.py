# Interface de linha de comando para testar o motor (a API virá em src/api).
#   python -m src.cli --new "Nome"      cria uma persona a partir de persona_example
#   python -m src.cli --list            lista as personas (mais recentes primeiro)
#   python -m src.cli <uuid>            conversa com a persona (/sair para encerrar)
from __future__ import annotations

import argparse
import asyncio
import sys

from src.data.database import Database
from src.data.personas import PersonaRegistry, create_persona
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


def _chat_loop(config: Config, persona_id: str) -> int:
    try:
        orch = Orchestrator.build(persona_id, config)
    except ConfigError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    print(f"conversando com {orch.name} (/sair para encerrar)\n")
    try:
        while True:
            try:
                text = input("você> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text in ("/sair", "/exit"):
                break
            if not text:
                continue
            try:
                print(f"{orch.name}> {asyncio.run(orch.chat([text]))}\n")
            except BridgeError as exc:
                print(f"[falha ao falar com o modelo: {exc}]\n"
                      "Verifique PERSONA_PROVIDER, o `ollama serve` (local) ou a GROQ_API_KEY (nuvem).\n",
                      file=sys.stderr)
    finally:
        orch.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
