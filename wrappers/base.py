import abc
import dm_env


class Wrapper(dm_env.Environment):
    def __init__(self, env: dm_env.Environment):
        self.env = env

    def observation(self, timestep: dm_env.TimeStep):
        return timestep.observation

    def reward(self, timestep: dm_env.TimeStep):
        return timestep.reward

    def done(self, timestep: dm_env.TimeStep):
        return timestep.last()

    def step_type(self, timestep: dm_env.TimeStep):
        return timestep.step_type

    def discount(self, timestep: dm_env.TimeStep):
        return timestep.discount

    def step(self, action) -> dm_env.TimeStep:
        timestep = self.env.step(action)
        return self._wrap_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        return self._wrap_timestep(self.env.reset())

    def _wrap_timestep(self, timestep) -> dm_env.TimeStep:
        return timestep._replace(
            step_type=self.step_type(timestep),
            reward=self.reward(timestep),
            discount=self.discount(timestep),
            observation=self.observation(timestep)
        )

    def action_spec(self):
        return self.env.action_spec()

    def observation_spec(self):
        return self.env.observation_spec()

    # TODO: explicit declaration
    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def unwrapped(self) -> dm_env.Environment:
        if hasattr(self.env, 'unwrapped'):
            return self.env.unwrapped
        else:
            return self.env
