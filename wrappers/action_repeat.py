from .base import Wrapper
from dm_env import TimeStep


class ActionRepeat(Wrapper):
    def __init__(self, env, frames_number: int):
        assert frames_number > 0
        super().__init__(env)
        self.fn = frames_number

    def step(self, action):
        rew_sum = 0.
        discount = 1.
        for _ in range(self.fn):
            timestep = self.env.step(action)
            rew_sum += discount*timestep.reward
            discount *= timestep.discount
            if timestep.last():
                break
        return timestep._replace(reward=rew_sum, discount=discount)
