""" Unpacks dm_env TimeStep to gym-like tuple """
from dm_env import Environment


class GymWrapper:
    def __init__(self, env: Environment):
        self.env = env

    def step(self, action):
        timestep = self.env.step(action)
        obs = timestep.observation
        done = timestep.last()
        reward = timestep.reward
        return obs, reward, done

    def reset(self):
        return self.env.reset().observation

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        raise NotImplementedError
