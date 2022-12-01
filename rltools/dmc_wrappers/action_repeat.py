import dm_env

from rltools.dmc_wrappers import base


class ActionRepeat(base.Wrapper):
    """Repeat the same action for multiple times."""

    def __init__(self,
                 env: dm_env.Environment,
                 frames_number: int
                 ) -> None:
        super().__init__(env)
        assert frames_number > 0
        self.frames_number = frames_number

    def step(self, action: base.Action) -> dm_env.TimeStep:
        rew_sum = 0.
        discount = 1.
        for _ in range(self.frames_number):
            timestep = self._env.step(action)
            rew_sum += discount * timestep.reward
            discount *= timestep.discount
            if timestep.last():
                break
        # pylint: disable-next=protected-access
        return timestep._replace(reward=rew_sum, discount=discount)
