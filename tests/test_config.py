from typing import Tuple, Optional
import os
import sys
import unittest
import dataclasses

from rltools.config import Config


@dataclasses.dataclass
class TestConfig(Config):
    wrong_type: int = "3"
    tuple_container: Tuple[int, ...] = (1., 2., 3.)
    generic_alias: tuple[float] = (3,)
    proper: float = 1.
    mutable_container: list[int] = (2, 3.)
    bool_flag: bool = 1
    opt_value: Optional[int] = None


class ConfigTest(unittest.TestCase):

    def test_types_autoconversion(self):
        cfg = TestConfig()
        self.assertIsInstance(cfg.wrong_type, int)
        self.assertIsInstance(cfg.tuple_container, tuple)
        self.assertTupleEqual(cfg.tuple_container, (1, 2, 3))
        self.assertIsInstance(cfg.generic_alias, tuple)
        self.assertIsInstance(cfg.generic_alias[0], float)
        self.assertIsInstance(cfg.proper, float)
        self.assertIsInstance(cfg.mutable_container, list)
        self.assertIsInstance(cfg.mutable_container[0], int)
        self.assertIsInstance(cfg.bool_flag, bool)
        self.assertEqual(cfg.bool_flag, True)
        self.assertIsInstance(cfg.opt_value, type(None))

    def test_save_load(self):
        cfg = TestConfig()
        cfg.save("tmp.yaml")
        self.assertTrue(os.path.exists("tmp.yaml"))
        lcfg = TestConfig.load("tmp.yaml")
        os.remove("tmp.yaml")
        self.assertEqual(lcfg, cfg)

    def test_argparse(self):
        orig_argv = sys.argv.copy()
        sys.argv.extend([
            "--wrong_type", "4",
            "--tuple_container", "3", "4", "5",
            "--generic_alias", "6",
            "--mutable_container", "7", "8",
            "--proper", "2",
            "--no-bool_flag",
            "--unknown_arg", "0",
            "--opt_value", "1",
        ])
        cfg = TestConfig.from_entrypoint()
        sys.argv = orig_argv
        self.assertEqual(cfg.wrong_type, 4)
        self.assertEqual(cfg.tuple_container, (3, 4, 5))
        self.assertEqual(cfg.generic_alias, (6.,))
        self.assertEqual(cfg.mutable_container, [7, 8])
        self.assertEqual(cfg.proper, 2.)
        self.assertEqual(cfg.bool_flag, False)
        self.assertEqual(cfg.opt_value, 1)
        self.assertTrue(not hasattr(cfg, "unknown_arg"))

