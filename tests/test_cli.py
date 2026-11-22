import contextlib
import io
import json
import unittest
from tempfile import TemporaryDirectory

from helpers import INTERP_OK, DownModel, RepeatModel, make_root
from src import cli
from src.data.database import Database
from src.data.personas import PersonaRegistry, create_persona
from src.utils.config import Config
from src.utils.logger import setup_logging, shutdown_logging

SUMMARY = json.dumps({"summary": "Resumo da sessão", "state": "Calma", "scene": {"local": "sala"}})


def scripted(*lines):
    """Simula o teclado: devolve cada linha e, no fim, EOFError (como o Ctrl-D)."""
    it = iter(lines)

    def read():
        try:
            return next(it) + "\n"
        except StopIteration:
            raise EOFError from None
    return read


class CliChatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.paths = make_root(self.tmp.name)
        self.config = Config(self.paths, load_env=False)
        setup_logging(self.paths.logs)  # os modelos falsos criam loggers no construtor
        with Database(self.paths.global_db) as db:
            self.pid = create_persona(PersonaRegistry(db), self.paths, "Ana").id
        self.models = {"actor": RepeatModel(["Olá, tudo bem!"]), "primary_interpreter": RepeatModel([INTERP_OK]),
                       "summary": RepeatModel([SUMMARY]), "notetaker": RepeatModel(['{"notes": []}']),
                       "consolidator": RepeatModel(['{"events": [], "knowledge": []}'])}

    def tearDown(self):
        shutdown_logging()
        self.tmp.cleanup()

    def run_chat(self, *lines):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli._chat_loop(self.config, self.pid, self.models, scripted(*lines))
        return code, out.getvalue(), err.getvalue()

    def summary_file(self):
        return self.paths.persona_dir(self.pid) / "summary.md"

    def test_reply_is_printed_and_exit_command_finalizes_background(self):
        code, out, err = self.run_chat("oi", "/sair", "nunca lida")
        self.assertEqual(code, 0)
        self.assertIn("conversando com Ana", out)
        self.assertIn("Ana> Olá, tudo bem!", out)
        self.assertIn("salvando memórias", err)
        self.assertEqual(self.summary_file().read_text().strip(), "Resumo da sessão")  # flush no /sair
        self.assertEqual(len(self.models["actor"].calls), 1)  # a linha depois do /sair não foi usada

    def test_end_of_input_also_finalizes(self):
        code, _, _ = self.run_chat("oi")  # sem /sair: a entrada acaba (Ctrl-D / pipe)
        self.assertEqual(code, 0)
        self.assertTrue(self.summary_file().is_file())

    def test_status_command_shows_emotions_and_blank_lines_are_ignored(self):
        code, out, _ = self.run_chat("oi", "", "   ", "/status", "/sair")
        self.assertEqual(code, 0)
        self.assertIn("emoções:", out)
        self.assertIn("alegria", out)  # o interpretador de teste sobe a alegria
        self.assertEqual(len(self.models["actor"].calls), 1)

    def test_model_failure_is_reported_and_session_continues(self):
        self.models["actor"] = DownModel(cli.BridgeError("HTTP 503", status=503))
        code, out, err = self.run_chat("primeira", "segunda", "/sair")
        self.assertEqual(code, 0)
        self.assertEqual(err.count("falha ao falar com o modelo"), 2)  # as duas, sem encerrar
        self.assertEqual(len(self.models["actor"].calls), 2)
        self.assertNotIn("Ana>", out)

    def test_unknown_persona_returns_error(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli._chat_loop(self.config, "nao-existe", self.models, scripted("/sair"))
        self.assertEqual(code, 1)
        self.assertIn("persona não encontrada", err.getvalue())


if __name__ == "__main__":
    unittest.main()
