from typing import Tuple, Mapping
import os
import sys
import unittest
import dataclasses

from rltools.config import Config


@dataclasses.dataclass
class TestConfig(Config):
    wrong_type: int = "3"
    std_container: Tuple[int, ...] = (1., 2., 3.)
    generic_alias: tuple[float] = (3,)
    proper: float = 1.
    mutable_container: list[int] = (2, 3.)


class ConfigTest(unittest.TestCase):

    def test_types_autoconversion(self):
        cfg = TestConfig()
        self.assertIsInstance(cfg.wrong_type, int)
        self.assertIsInstance(cfg.std_container, tuple)
        self.assertIsInstance(cfg.std_container[0], int)
        self.assertIsInstance(cfg.generic_alias, tuple)
        self.assertIsInstance(cfg.generic_alias[0], float)
        self.assertIsInstance(cfg.proper, float)
        self.assertIsInstance(cfg.mutable_container, list)
        self.assertIsInstance(cfg.mutable_container[0], int)

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
                         "--generic_alias", "6",
                         "--mutable_container", "7", "8",
                         "--proper", "2"])
        cfg = TestConfig.from_entrypoint()
        sys.argv = orig_argv
        self.assertEqual(cfg.wrong_type, 4)
        self.assertEqual(cfg.std_container, (3, 4, 5))
        self.assertEqual(cfg.generic_alias, (6.,))
        self.assertEqual(cfg.mutable_container, [7, 8])
        self.assertEqual(cfg.proper, 2.)
