# python -m src.api [--host 127.0.0.1] [--port 8000]
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="persona-api")
    parser.add_argument("--host", default="127.0.0.1", help="padrão: só local (não há autenticação)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("src.api.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
