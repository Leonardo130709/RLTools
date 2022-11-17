import os
import sys
import unittest
import dataclasses

from rltools.config import Config


@dataclasses.dataclass
class TestConfig(Config):
    wrong_type: int = "3"
    std_container: tuple[int, ...] = (1, 2, 3)
    proper: float = 1.


class ConfigTest(unittest.TestCase):

    def test_types_autoconversion(self):
        cfg = TestConfig()
        self.assertIsInstance(cfg.wrong_type, int)
        self.assertIsInstance(cfg.std_container, tuple)
        self.assertIsInstance(cfg.proper, float)

    def test_save_load(self):
        cfg = TestConfig()
        cfg.save("tmp.yaml")
        self.assertTrue(os.path.exists("tmp.yaml"))
        lcfg = TestConfig.load("tmp.yaml")
        os.remove("tmp.yaml")
        self.assertEqual(lcfg, cfg)

    def test_argparse(self):
        orig_argv = sys.argv.copy()
        sys.argv.extend(["--wrong_type", "4",
                         "--std_container", "3", "4", "5",
                         "--proper", "2"])
        cfg = TestConfig.from_entrypoint()
        sys.argv = orig_argv
        self.assertEqual(cfg.wrong_type, 4)
        self.assertEqual(cfg.std_container, (3, 4, 5))
        self.assertEqual(cfg.proper, 2.)
