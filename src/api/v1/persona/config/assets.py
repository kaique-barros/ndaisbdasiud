# Assets da persona (imagens, como avatar/aparência) em .data/personas/<uuid>/assets/.
# Segurança: o nome do arquivo é gerado aqui (nunca vem do cliente), o tipo é detectado pelo
# CONTEÚDO (png/jpeg/gif/webp; SVG fica de fora por poder conter script) e há limite de tamanho.
from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.api.deps import get_state
from src.api.state import AppState

router = APIRouter(prefix="/v1/personas/{persona_id}/config/assets", tags=["config"])

MAX_BYTES = 5 * 1024 * 1024
MAX_ASSETS = 100
MEDIA = {"png": "image/png", "jpg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
_NAME = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.(png|jpg|gif|webp)$")


def sniff(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _dir(state: AppState, persona_id: str) -> Path:
    state.record(persona_id)
    return state.config.paths.persona_dir(persona_id) / "assets"


def _path(state: AppState, persona_id: str, name: str) -> Path:
    if not _NAME.match(name):  # impede qualquer caminho vindo da URL
        raise HTTPException(404, "asset não encontrado")
    path = _dir(state, persona_id) / name
    if not path.is_file():
        raise HTTPException(404, "asset não encontrado")
    return path


@router.get("")
def list_assets(persona_id: str, state: AppState = Depends(get_state)) -> list[dict]:
    directory = _dir(state, persona_id)
    if not directory.is_dir():
        return []
    return [{"name": p.name, "size": p.stat().st_size}
            for p in sorted(directory.iterdir()) if _NAME.match(p.name)]


@router.post("", status_code=201)
async def upload_asset(persona_id: str, file: UploadFile, state: AppState = Depends(get_state)) -> dict:
    directory = _dir(state, persona_id)
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"o arquivo passa de {MAX_BYTES // (1024 * 1024)} MB")
    kind = sniff(data)
    if kind is None:
        raise HTTPException(415, "formato não suportado (use png, jpeg, gif ou webp)")
    directory.mkdir(parents=True, exist_ok=True)
    if sum(1 for p in directory.iterdir()) >= MAX_ASSETS:
        raise HTTPException(409, f"limite de {MAX_ASSETS} assets por persona")
    name = f"{uuid.uuid4()}.{kind}"
    (directory / name).write_bytes(data)
    return {"name": name, "size": len(data)}


@router.get("/{name}")
def get_asset(persona_id: str, name: str, state: AppState = Depends(get_state)):
    path = _path(state, persona_id, name)
    return FileResponse(path, media_type=MEDIA[path.suffix[1:]],
                        headers={"X-Content-Type-Options": "nosniff"})


@router.delete("/{name}", status_code=204)
def delete_asset(persona_id: str, name: str, state: AppState = Depends(get_state)) -> None:
    _path(state, persona_id, name).unlink()
