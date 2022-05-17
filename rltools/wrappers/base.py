class Wrapper:
    def __init__(self, env):
        self.env = env

    def observation(self, timestamp):
        return timestamp.observation

    def reward(self, timestamp):
        return timestamp.reward

    def done(self, timestamp):
        return timestamp.last()

    def step(self, action):
        timestamp = self.env.step(action)
        obs = self.observation(timestamp)
        r = self.reward(timestamp)
        d = self.done(timestamp)
        return obs, r, d

    def reset(self):
        return self.observation(self.env.reset())

    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def unwrapped(self):
        if hasattr(self.env, 'unwrapped'):
            return self.env.unwrapped
        else:
            return self.env
