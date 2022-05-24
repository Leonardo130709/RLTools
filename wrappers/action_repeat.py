from .base import Wrapper
from dm_env import TimeStep


class ActionRepeat(Wrapper):
    def __init__(self, env, frames_number: int):
        assert frames_number > 0
        super().__init__(env)
        self.fn = frames_number

    # TODO: apply discount from timestep discount
    def step(self, action):
        """ Sum of rewards ignores task discount factor."""
        rew_sum = 0
        discount = 1.
        for i in range(self.fn):
            timestep = self.env.step(action)
            rew_sum += timestep.reward
            discount *= timestep.discount
            if timestep.last():
                break

        return TimeStep(
            step_type=timestep.step_type,
            reward=rew_sum,
            discount=discount,
            observation=timestep.observation
        )
