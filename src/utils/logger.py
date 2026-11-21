# Log unificado.
#   - .logs/general.log                    -> tudo que acontece no sistema
#   - .logs/<categoria>/<classe>.log       -> eventos de cada classe (memory, engine, core...)
#   - .logs/<categoria>/<classe>.prompt.log -> conteúdo dos prompts (NÃO vai para o geral)
# Assim os logs de eventos ficam curtos e o conteúdo enviado ao modelo fica separado.
from __future__ import annotations

import logging
from pathlib import Path

ROOT_NAME = "persona"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_log_dir: Path | None = None


def setup_logging(log_dir: Path, level: int = logging.INFO, console: bool = False) -> None:
    """Configura o log geral. Chamar uma vez (no orchestrator)."""
    global _log_dir
    _log_dir = Path(log_dir)
    _log_dir.mkdir(parents=True, exist_ok=True)

    shutdown_logging()  # fecha handlers antigos (inclusive os das classes já criadas)

    root = logging.getLogger(ROOT_NAME)
    root.setLevel(level)
    root.propagate = False
    root.addHandler(_file_handler(_log_dir / "general.log"))
    if console:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(stream)


def shutdown_logging() -> None:
    """Fecha e remove os handlers de todos os loggers do sistema."""
    for name, obj in list(logging.root.manager.loggerDict.items()):
        if name == ROOT_NAME or name.startswith(ROOT_NAME + "."):
            if isinstance(obj, logging.Logger):
                for handler in list(obj.handlers):
                    obj.removeHandler(handler)
                    handler.close()


def get_logger(name: str, category: str = "core") -> logging.Logger:
    """Logger de eventos de uma classe: vai para o arquivo dela e para o geral."""
    logger = logging.getLogger(f"{ROOT_NAME}.{category}.{name}")
    if not _has_file_handler(logger, f"{name}.log"):
        logger.addHandler(_file_handler(_require_dir() / category / f"{name}.log"))
    return logger


def get_prompt_logger(name: str, category: str = "core") -> logging.Logger:
    """Logger do conteúdo dos prompts: só no arquivo da classe, sem propagar."""
    logger = logging.getLogger(f"{ROOT_NAME}.{category}.{name}.prompt")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if not _has_file_handler(logger, f"{name}.prompt.log"):
        logger.addHandler(_file_handler(_require_dir() / category / f"{name}.prompt.log"))
    return logger


def _require_dir() -> Path:
    if _log_dir is None:
        raise RuntimeError("chame setup_logging() antes de get_logger()")
    return _log_dir


def _file_handler(path: Path) -> logging.FileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    return handler


def _has_file_handler(logger: logging.Logger, filename: str) -> bool:
    return any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename).name == filename
        for h in logger.handlers
    )
