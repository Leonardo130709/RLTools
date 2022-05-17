from .base import Wrapper
import numpy as np


class ActionRepeat(Wrapper):
    def __init__(self, env, frames_number: int):
        assert frames_number > 0
        super().__init__(env)
        self.fn = frames_number

    def step(self, action):
        """ Sum of rewards ignores task discount factor."""
        rew_sum = 0
        for i in range(self.fn):
            next_obs, reward, done = self.env.step(action)
            rew_sum += reward
            if done:
                break
        return np.float32(next_obs), np.float32(rew_sum), done
