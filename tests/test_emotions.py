import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.core.emotions import EmotionDictionary, EmotionParams, EmotionState, EmotionStore
from src.utils.config import Config, ConfigError, Paths
from src.utils.logger import setup_logging, shutdown_logging

REPO = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


class EmotionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        setup_logging(Path(self.tmp.name))
        cfg = Config(Paths(REPO), load_env=False)
        self.dict = EmotionDictionary(cfg.engine("emotions_description"))
        self.params = EmotionParams.from_config(cfg.engine("emotions"))
        self.clock = Clock()

    def tearDown(self):
        shutdown_logging()
        self.tmp.cleanup()

    def state(self, baseline=None, state=None):
        return EmotionState(self.dict, self.params, baseline or {}, state, T0, self.clock)

    def test_dictionary_is_the_full_wheel(self):
        d = self.dict
        kinds = [v["kind"] for v in d.dyads.values()]
        self.assertEqual([kinds.count(k) for k in ("primary", "secondary", "tertiary")], [8, 8, 8])
        self.assertEqual(d.opposite("joy"), "sadness")
        self.assertEqual(d.opposite("fear"), "anger")
        self.assertEqual(d.opposite("anticipation"), "surprise")
        pairs = {frozenset(v["components"]) for v in d.dyads.values()}
        self.assertEqual(len(pairs), 24)  # nenhuma combinação repetida
        for v in d.dyads.values():  # a distância na roda bate com o tipo da díade
            i, j = (d.wheel.index(c) for c in v["components"])
            dist = min((i - j) % 8, (j - i) % 8)
            self.assertEqual(v["kind"], {1: "primary", 2: "secondary", 3: "tertiary"}[dist])

    def test_dictionary_validation(self):
        with self.assertRaises(ConfigError):
            EmotionDictionary({})
        bad = {"wheel": self.dict.wheel, "primary": self.dict.primary,
               "dyads": {"x": {"label": "x", "components": ["joy", "joy"]}}}
        with self.assertRaises(ConfigError):
            EmotionDictionary(bad)
        short_wheel = {"wheel": self.dict.wheel[:7], "primary": self.dict.primary, "dyads": {}}
        with self.assertRaises(ConfigError):
            EmotionDictionary(short_wheel)

    def test_erosion_follows_half_life(self):
        s = self.state(baseline={"sadness": 0.1}, state={"sadness": 0.9})
        self.clock.advance(hours=12)  # meia-vida da tristeza = 12h
        s.decay()
        self.assertAlmostEqual(s.values["sadness"], 0.1 + 0.8 * 0.5, places=6)
        self.clock.advance(days=30)
        s.decay()
        self.assertAlmostEqual(s.values["sadness"], 0.1, places=3)  # volta ao temperamento
        self.assertAlmostEqual(s.values["joy"], 0.0)

    def test_decay_never_goes_back_in_time(self):
        s = self.state(state={"joy": 0.8})
        s.decay(T0 - timedelta(hours=5))
        self.assertEqual(s.values["joy"], 0.8)

    def test_apply_clamps_inhibits_opposites_and_reports_net(self):
        s = self.state(state={"sadness": 0.5})
        net = s.apply({"joy": 0.4, "unknown": 1, "anger": 9})  # 9 é limitado a max_delta (0.5)
        self.assertAlmostEqual(s.values["joy"], 0.4)
        self.assertAlmostEqual(s.values["sadness"], 0.3)  # 0.5 - 0.4*0.5
        self.assertAlmostEqual(s.values["anger"], 0.5)
        self.assertAlmostEqual(s.values["fear"], 0.0)  # oposto da raiva já estava em 0
        self.assertEqual(net, {"joy": 0.4, "sadness": -0.2, "anger": 0.5})
        s.apply({"joy": 5})
        self.assertLessEqual(s.values["joy"], 1.0)
        s.apply({"joy": -5})
        self.assertGreaterEqual(s.values["joy"], 0.0)

    def test_dyads_use_min_and_threshold(self):
        s = self.state(state={"joy": 0.6, "trust": 0.5, "fear": 0.1})
        self.assertEqual([(d.key, d.label, d.intensity) for d in s.dyads()], [("love", "amor", 0.5)])
        s2 = self.state(state={"joy": 0.9, "trust": 0.2})
        self.assertEqual(s2.dyads(), [])

    def test_level_names_and_describe(self):
        s = self.state(state={"joy": 0.2})
        self.assertEqual(s.level_name("joy"), "serenidade")
        self.assertEqual(self.state(state={"joy": 0.5}).level_name("joy"), "alegria")
        self.assertEqual(self.state(state={"joy": 0.9}).level_name("joy"), "êxtase")
        self.assertIn("Calma", self.state().describe())
        text = self.state(state={"joy": 0.9, "trust": 0.8}).describe()
        self.assertIn("êxtase (alegria, 0.90)", text)
        self.assertIn("amor (0.80)", text)

    def test_store_roundtrip_and_decay_on_load(self):
        path = Path(self.tmp.name) / "emotions.json"
        store = EmotionStore(path, self.dict, self.params, self.clock)
        s = store.load()  # arquivo inexistente: começa neutra
        self.assertEqual(sum(s.values.values()), 0)
        s.apply({"anger": 0.8})  # limitado a max_delta = 0.5
        store.save(s)
        self.assertEqual(json.loads(path.read_text())["state"]["anger"], 0.5)
        self.clock.advance(hours=3)  # meia-vida da raiva = 3h
        self.assertAlmostEqual(store.load().values["anger"], 0.25, places=3)
        path.write_text("   ")  # arquivo vazio também é aceito
        self.assertEqual(sum(store.load().values.values()), 0)
        path.write_text("{quebrado")
        with self.assertRaises(ConfigError):
            store.load()

    def test_example_persona_file_loads(self):
        store = EmotionStore(REPO / ".data/personas/persona_example/emotions.json",
                             self.dict, self.params, self.clock)
        s = store.load()
        self.assertAlmostEqual(s.baseline["joy"], 0.15)


if __name__ == "__main__":
    unittest.main()
