import dm_env

from rltools.dmc_wrappers import base


class ActionRepeat(base.Wrapper):
    """Repeat the same action for multiple times."""

    def __init__(self,
                 env: dm_env.Environment,
                 frames_number: int
                 ) -> None:
        assert frames_number > 0
        super().__init__(env)
        self.frames_number = frames_number

    def reset(self) -> dm_env.TimeStep:
        return self.env.reset()

    def step(self, action: base.Action) -> dm_env.TimeStep:
        reward = 0.
        discount = 1.
        for _ in range(self.frames_number):
            timestep = self.env.step(action)
            reward += timestep.reward
            discount *= timestep.discount
            if timestep.last():
                break
        # pylint: disable-next=protected-access
        return timestep._replace(reward=reward, discount=discount)
