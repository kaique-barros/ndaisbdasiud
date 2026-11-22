# Aplicação FastAPI. Sem lógica de persona aqui: só rotas finas sobre o Orchestrator.
# Sem autenticação: pensada para uso local (127.0.0.1). Antes de expor na rede, ponha um proxy
# com autenticação na frente.
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.state import AppState
from src.api.v1 import personas
from src.api.v1.persona.config import assets, descriptions, parameters
from src.api.v1.persona.config.memory import events, knowledge, messages, notes
from src.engine.main.actor import ActorError
from src.engine.model import Model
from src.utils.bridge import BridgeError
from src.utils.config import Config, ConfigError
from src.utils.logger import setup_logging, shutdown_logging


def create_app(config: Config | None = None, models: Mapping[str, Model] | None = None,
               background: bool = True) -> FastAPI:
    """models: modelos por componente (testes); em produção vêm das configs/provider."""
    config = config or Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(config.paths.logs)
        app.state.ctx = AppState(config, models, background)
        try:
            yield
        finally:
            await app.state.ctx.shutdown()
            shutdown_logging()

    app = FastAPI(title="Persona Engine", version="0.1.0", lifespan=lifespan)
    for module in (personas, parameters, descriptions, assets, events, knowledge, notes, messages):
        app.include_router(module.router)

    @app.exception_handler(BridgeError)
    @app.exception_handler(ActorError)
    async def model_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": f"falha ao falar com o modelo: {exc}"}, status_code=502)

    @app.exception_handler(ConfigError)
    async def config_error(_: Request, exc: ConfigError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=500)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    return app
