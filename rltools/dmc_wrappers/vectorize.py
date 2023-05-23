from typing import Optional
from collections.abc import Sequence, Callable
import multiprocessing as mp
import multiprocessing.connection
from enum import IntEnum

import tree
import numpy as np
import dm_env.specs

from .base import ObservationSpec

_EnvFactory = Callable[[], dm_env.Environment]


class _Commands(IntEnum):
    """Exposed methods."""
    OBS_SPEC = 0
    ACT_SPEC = 1
    REW_SPEC = 2
    DISC_SPEC = 3
    RESET = 4
    STEP = 5
    CLOSE = 6


def _worker(ctor: _EnvFactory,
            pipe: mp.connection.Connection,
            parent_pipe: mp.connection.Connection
            ) -> None:
    """Run environment loop."""
    parent_pipe.close()
    env = ctor()
    comm = None
    try:
        while comm != _Commands.CLOSE:
            comm, payload = pipe.recv()
            if comm == _Commands.OBS_SPEC:
                data = env.observation_spec()
            elif comm == _Commands.ACT_SPEC:
                data = env.action_spec()
            elif comm == _Commands.REW_SPEC:
                data = env.reward_spec()
            elif comm == _Commands.DISC_SPEC:
                data = env.discount_spec()
            elif comm == _Commands.RESET:
                data = env.reset()
            elif comm == _Commands.STEP:
                data = env.step(payload)
            else:
                raise RuntimeError(f"Unknown command {comm}")

            pipe.send(data)
    except Exception as exp:
        # TODO: Handle error
        raise exp
    finally:
        env.close()


class CloudpickleWrapper:
    """Taken from Gymnasium https://github.com/Farama-Foundation/Gymnasium."""

    def __init__(self, fn: Callable[[], dm_env.Environment]) -> None:
        self.fn = fn

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        import pickle
        self.fn = pickle.loads(ob)

    def __call__(self):
        return self.fn()


# TODO: add support for arbitrary method or attr call.
class AsyncEnv(dm_env.Environment):
    """Creates multiple environment instances and calls them simultaneously
    when acting. This may improve performance when policy represented by NN."""

    def __init__(self,
                 env_fns: Sequence[_EnvFactory],
                 context: Optional[str] = None,
                 daemon: bool = False
                 ) -> None:
        ctx = mp.get_context(context)
        self.env_fns = list(env_fns)
        self._setup_specs()

        self._parent_pipes, self._processes = [], []
        for env_fn in self.env_fns:
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=_worker,
                args=(CloudpickleWrapper(env_fn), child_pipe, parent_pipe)
            )
            self._parent_pipes.append(parent_pipe)
            self._processes.append(process)
            process.daemon = daemon
            process.start()
            child_pipe.close()

    def reset(self) -> dm_env.TimeStep:
        for pipe in self._parent_pipes:
            pipe.send((_Commands.RESET, None))

        return self._wait_results()

    def step(self, actions: np.ndarray) -> dm_env.TimeStep:
        for action, pipe in zip(actions, self._parent_pipes):
            pipe.send((_Commands.STEP, action))

        return self._wait_results()

    def action_spec(self) -> dm_env.specs.BoundedArray:
        return self._action_spec

    def observation_spec(self) -> ObservationSpec:
        return self._observation_spec

    def reward_spec(self) -> dm_env.specs.Array:
        return self._reward_spec

    def discount_spec(self) -> dm_env.specs.BoundedArray:
        return self._discount_spec

    def close(self):
        for pipe in self._parent_pipes:
            if not pipe.closed:
                pipe.close()
        for process in self._processes:
            process.join()

    def _wait_results(self) -> dm_env.TimeStep:
        return _stack_timesteps([pipe.recv() for pipe in self._parent_pipes])

    def _setup_specs(self):
        env = self.env_fns[0]()
        self._action_spec = env.action_spec()
        self._observation_spec = env.observation_spec()
        self._reward_spec = env.reward_spec()
        self._discount_spec = env.discount_spec()
        env.close()


class SequentialEnv(dm_env.Environment):
    """Sequential wrapper for computationally simple envs
    to mitigate multiprocessing overheads."""

    def __init__(self,
                 env_fns: Sequence[_EnvFactory],
                 ) -> None:
        self.env_fns = env_fns
        self.envs = [fn() for fn in env_fns]

    def reset(self) -> dm_env.TimeStep:
        return _stack_timesteps([env.reset() for env in self.envs])

    def step(self, actions: np.ndarray) -> dm_env.TimeStep:
        return _stack_timesteps([
            env.step(action) for env, action in zip(self.envs, actions)
                ])

    def action_spec(self) -> dm_env.specs.Array:
        return self.envs[0].action_spec()

    def observation_spec(self) -> ObservationSpec:
        return self.envs[0].observation_spec()

    def reward_spec(self) -> dm_env.specs.Array:
        return self.envs[0].reward_spec()

    def discount_spec(self) -> dm_env.specs.BoundedArray:
        return self.envs[0].discount_spec()


def _stack_timesteps(timesteps: list[dm_env.TimeStep]) -> dm_env.TimeStep:
    """Makes TimeStep methods visible.
    Note that StepType will be replaced by its numeric value."""
    return tree.map_structure(lambda *t: np.stack(t), *timesteps)
