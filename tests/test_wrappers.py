import unittest
import re

import tree
import dm_env
import numpy as np
import numpy.testing as npt
from dm_env import specs, TimeStep

from rltools import dmc_wrappers

from . import mock_env

#TODO test inplace observation and specs mutations


class _CheckEqualTs:
    def __init__(self, ts1: TimeStep, ts2: TimeStep):
        nested = tree.map_structure(np.array_equal, ts1.observation,
                                    ts2.observation)
        self.eq_observations = all(tree.flatten(nested))
        self.eq_rewards = ts1.reward == ts2.reward
        self.eq_discounts = ts1.discount == ts2.discount
        self.eq_step_types = ts1.step_type == ts2.step_type

    def __bool__(self):
        return all([self.eq_observations, self.eq_rewards,
                    self.eq_discounts, self.eq_step_types])

    def __repr__(self):
        return ", ".join(f"{k}={v}" for k, v in self.__dict__.items())


class BaseTest(unittest.TestCase):
    """Base wrapper tests."""

    def setUp(self) -> None:
        """w{$name} states for wrapped_{$name}"""
        self._env = mock_env.TestEnv()
        self._wenv = dmc_wrappers.base.Wrapper(self._env)

    def test_environment_specs(self):
        env_specs = self._wenv.environment_specs
        self.assertEqual(self._env.action_spec(), env_specs.action_spec)
        self.assertDictEqual(
            self._env.observation_spec(), env_specs.observation_spec)
        self.assertEqual(self._env.reward_spec(), env_specs.reward_spec)
        self.assertEqual(self._env.discount_spec(), env_specs.discount_spec)

    def test_unwrapped(self):
        self.assertIs(self._wenv.unwrapped, self._env)

    def test_reset(self):
        timestep = self._env.reset()
        wtimestep = self._wenv.reset()
        check = _CheckEqualTs(wtimestep, timestep)
        self.assertTrue(check, check)

    def test_step(self):
        act = self._wenv.action_spec().generate_value()
        self._env.reset()
        timestep = self._env.step(act)
        self._wenv.reset()
        wtimestep = self._wenv.step(act)
        check = _CheckEqualTs(wtimestep, timestep)
        self.assertTrue(check, check)

    def test_getattr(self):
        if self._wenv is None:
            return

        nested = self._wenv
        for _ in range(10):
            nested = dmc_wrappers.base.Wrapper(nested)

        self.assertTrue(hasattr(nested, "_time_limit"))
        self.assertEqual(
            getattr(nested, "_time_limit"), self._env._time_limit)

        self.assertRaises(AttributeError,
                          lambda: getattr(nested, "attribute_error")
                          )


# abc.ABC didn't worked with unittest.
#  Maybe it shouldn't but now test counts will show wrong number (+2 empty).
class WrapperTest(unittest.TestCase):

    def setUp(self) -> None:
        env = mock_env.TestEnv()
        self.env = dmc_wrappers.base.Wrapper(env)

        self._wenv: dm_env.Environment = None
        self._assumed_env_specs: dmc_wrappers.EnvironmentSpecs = None

    def test_correspondence(self):
        if self._wenv is None:
            return

        def validate(sp, ts):
            # specs.Array raises ValueError if the value is invalid.
            try:
                tree.map_structure(
                    specs.Array.validate, sp.observation_spec, ts.observation)
                sp.reward_spec.validate(ts.reward)
                sp.discount_spec.validate(ts.discount)
            except ValueError as exp:
                self.fail(str(exp))

        # On reset.
        self._wenv.reset()
        action = self._wenv.action_spec().generate_value()
        init_wts = self._wenv.step(action)
        validate(self._assumed_env_specs, init_wts)

        # On step.
        act = self._wenv.action_spec().generate_value()
        wts = self._wenv.step(act)
        validate(self._assumed_env_specs, wts)
        try:
            self._assumed_env_specs.action_spec.validate(act)
        except ValueError as exp:
            self.fail(str(exp))

    def test_types(self):
        if self._wenv is None:
            return

        self.assertIsInstance(self._wenv, dm_env.Environment)
        self.assertIsInstance(
            self._wenv.environment_specs, dmc_wrappers.EnvironmentSpecs)


class ActionRepeatTest(WrapperTest):

    def setUp(self) -> None:
        super().setUp()
        self._wenv = dmc_wrappers.ActionRepeat(self.env, 2)
        self._assumed_env_specs = self.env.environment_specs

    def test_repetition(self):
        self._wenv.reset()
        act = self._assumed_env_specs.action_spec.generate_value()
        self._wenv.step(act)
        self.assertEqual(self._wenv.unwrapped._step, 2)

    def test_not_overcounting(self):
        env = mock_env.MockEnv(self.env.observation_spec(),
                               self.env.action_spec(),
                               time_limit=7)
        env = dmc_wrappers.ActionRepeat(env, 2)
        r = 0
        ts = env.reset()
        step = 0
        while not ts.last():
            step += 1
            act = env.action_spec().generate_value()
            ts = env.step(act)
            r += ts.reward

        self.assertEqual(r, 7)
        self.assertEqual(step, np.ceil(7 / 2.))


class ActionRescaleTest(WrapperTest):

    def setUp(self) -> None:
        super().setUp()
        self._wenv = dmc_wrappers.ActionRescale(self.env)
        act_spec = self.env.environment_specs.action_spec
        lim = np.full(act_spec.shape, 1., act_spec.dtype)
        self._assumed_env_specs = self.env.environment_specs._replace(
            action_spec=act_spec.replace(minimum=-lim,
                                         maximum=lim)
        )


class FramesStackTest(WrapperTest):

    def setUp(self) -> None:
        super().setUp()
        self._wenv = dmc_wrappers.FrameStack(self.env, 2)
        new_obs_spec = tree.map_structure(
            lambda sp: sp.replace(shape=(2,) + sp.shape),
            self.env.environment_specs.observation_spec
        )
        self._assumed_env_specs = self.env.environment_specs._replace(
            observation_spec=new_obs_spec
        )


class DiscreteTest(WrapperTest):

    def setUp(self) -> None:
        super().setUp()
        self._bins = 9
        self._wenv = dmc_wrappers.DiscreteActionWrapper(self.env, self._bins)
        act_spec = self.env.action_spec()
        self._assumed_env_specs = self.env.environment_specs._replace(
            action_spec=specs.BoundedArray(act_spec.shape + (self._bins,),
                                           minimum=0,
                                           maximum=1,
                                           dtype=np.int32)
        )

    def test_limits(self):
        _cache = self.env.step
        self.env.step = lambda act: act
        act_space = self._wenv.action_spec()
        action = np.zeros(act_space.shape, act_space.dtype)
        low_action = action.copy()
        low_action[:, 0] = 1
        action[:, -1] = 1
        low = self._wenv.step(low_action)
        high = self._wenv.step(action)
        self.env.step = _cache
        self.assertTrue(np.allclose(low, -1))
        self.assertTrue(np.allclose(high, 1))


class ObsFilterTest(WrapperTest):

    def setUp(self) -> None:
        super().setUp()
        self._make_obs_spec = lambda k: {k: self.env.observation_spec()[k]}

    def test_by_callable(self):
        def predicate(key, spec):
            return hasattr(spec, "maximum")
        self._wenv = dmc_wrappers.ObsFilter(self.env, predicate)
        obs_spec = self._make_obs_spec("bounded_obs")
        self._assumed_env_specs = self.env.environment_specs._replace(
            observation_spec=obs_spec
        )
        super().test_correspondence()
        super().test_types()

    def test_by_search(self):
        self._wenv = dmc_wrappers.ObsFilter(self.env, r"alar",
                                            method=re.search)
        obs_spec = self._make_obs_spec("scalar_obs")
        self._assumed_env_specs = self.env.environment_specs._replace(
            observation_spec=obs_spec
        )
        super().test_correspondence()
        super().test_types()

    def test_by_match(self):
        self._wenv = dmc_wrappers.ObsFilter(self.env, r"^.*d_obs.*",
                                            method=re.match)
        obs_spec = self._make_obs_spec("bounded_obs")
        self._assumed_env_specs = self.env.environment_specs._replace(
            observation_spec=obs_spec
        )
        super().test_correspondence()
        super().test_types()

    def test_by_dtype(self):
        self._wenv = dmc_wrappers.ObsFilter(self.env, np.uint8)
        obs_spec = self._make_obs_spec("bounded_obs")
        self._assumed_env_specs = self.env.environment_specs._replace(
            observation_spec=obs_spec
        )
        super().test_correspondence()
        super().test_types()


class AdaptersTest(unittest.TestCase):

    def setUp(self) -> None:
        self.dmc_env = mock_env.TestEnv()
        self.gym_env = dmc_wrappers.DmcToGym(self.dmc_env)
        self.cycle_dmc_env = dmc_wrappers.GymToDmc(self.gym_env)
        self.cycle_gym_env = dmc_wrappers.DmcToGym(self.cycle_dmc_env)
        self.cycle_gym_env.reset()

    def test_reset(self):
        self.assertTrue(
            _CheckEqualTs(
                self.dmc_env.reset(),
                self.cycle_dmc_env.reset()
            )
        )
        npt.assert_equal(self.gym_env.reset(), self.cycle_gym_env.reset())

    def test_observation_specs(self):
        self.assertDictEqual(self.dmc_env.observation_spec(),
                             self.cycle_dmc_env.observation_spec()
                             )
        self.assertDictEqual(self.gym_env.observation_space,
                             self.cycle_gym_env.observation_space
                             )

    def test_action_specs(self):
        self.assertEqual(self.dmc_env.action_spec(),
                         self.cycle_dmc_env.action_spec()
                         )
        self.assertEqual(self.gym_env.action_space,
                         self.cycle_gym_env.action_space
                         )

    def test_step(self):
        act = self.dmc_env.action_spec().generate_value()
        self.assertTrue(_CheckEqualTs(self.dmc_env.step(act),
                                      self.cycle_dmc_env.step(act)))
        gym_tuple = self.gym_env.step(act)
        cycle_gym_tuple = self.cycle_gym_env.step(act)
        for v1, v2 in zip(gym_tuple[0].values(), cycle_gym_tuple[0].values()):
            npt.assert_equal(v1, v2)
        self.assertTupleEqual(gym_tuple[1:], cycle_gym_tuple[1:])


class TimeLimitTest(WrapperTest):

    def setUp(self) -> None:
        super().setUp()

        self._wenv = dmc_wrappers.TimeLimit(self.env, 10)
        self._assumed_env_specs = self.env.environment_specs

    def test_reset(self):
        ts1 = self.env.reset()
        ts2 = self._wenv.reset()
        check = _CheckEqualTs(ts1, ts2)
        self.assertTrue(check, check)

    def test_truncation(self):
        self._wenv.reset()
        done = False
        steps = 0
        act_space = self._wenv.action_spec()
        action = np.zeros(act_space.shape, act_space.dtype)
        while not done:
            steps += 1
            ts = self._wenv.step(action)
            done = ts.last()

        self.assertEqual(ts.step_type, dm_env.StepType.LAST)
        self.assertEqual(ts.discount, 1.)
        self.assertEqual(steps, self._wenv._time_limit)


class VectorizeTest(unittest.TestCase):

    def setUp(self) -> None:
        self.num_envs = 3
        env_fn = [lambda: mock_env.TestEnv() for _ in range(self.num_envs)]
        self.env = env_fn[0]()
        self.venv = dmc_wrappers.Vectorize(env_fn)

    def test_specs(self):
        self.assertEqual(self.env.action_spec(), self.venv.action_spec())
        self.assertEqual(self.env.observation_spec(),
                         self.venv.observation_spec()
                         )

    def test_timestep(self):
        ts = self.env.reset()
        vts = self.venv.reset()
        check = _CheckEqualTs(ts, vts[0])
        self.assertTrue(check, check)


if __name__ == "__main__":
    unittest.main()




