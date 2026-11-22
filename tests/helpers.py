# Peças compartilhadas pelos testes: modelos falsos e uma raiz de projeto temporária.
import json
import shutil
from pathlib import Path

from src.engine.model import Model
from src.utils.bridge import ChatResponse
from src.utils.config import Paths

REPO = Path(__file__).resolve().parents[1]
INTERP_OK = json.dumps({"queries": {}, "tone": "caloroso", "emotion_delta": {"joy": 0.3}})


class FakeModel(Model):
    """Responde com as `replies` em ordem e registra as mensagens recebidas."""

    def __init__(self, replies=("Oi!",)):
        super().__init__("fake")
        self.replies, self.calls = list(replies), []

    async def _complete(self, messages, params):
        self.calls.append([dict(m) for m in messages])
        return ChatResponse(text=self.replies.pop(0))


class RepeatModel(FakeModel):
    """Sempre a mesma resposta, quantas vezes forem pedidas."""

    async def _complete(self, messages, params):
        self.calls.append([dict(m) for m in messages])
        return ChatResponse(text=self.replies[0])


class DownModel(FakeModel):
    """Modelo fora do ar: toda chamada lança (e é registrada)."""

    def __init__(self, error=None):
        super().__init__([])
        self.error = error or ConnectionError("ollama fora do ar")

    async def _complete(self, messages, params):
        self.calls.append(messages)
        raise self.error


def make_root(tmp: str) -> Paths:
    root = Path(tmp)
    shutil.copytree(REPO / ".configs", root / ".configs")
    shutil.copytree(REPO / ".data" / "personas" / "persona_example",
                    root / ".data" / "personas" / "persona_example")
    return Paths(root)
