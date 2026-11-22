from __future__ import annotations

from fastapi import Request

from src.api.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.ctx
