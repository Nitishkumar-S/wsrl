

def ScaledRewardWrapper(env, scale: float, bias: float):
    """Dynamically wraps the environment with the correct Gym or Gymnasium wrapper."""
    
    # Try modern Gymnasium first
    try:
        import gymnasium
        if isinstance(env, gymnasium.Env):
            class GymnasiumScaledRewardWrapper(gymnasium.RewardWrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.scale = scale
                    self.bias = bias

                def reward(self, reward):
                    return reward * self.scale + self.bias
                    
            return GymnasiumScaledRewardWrapper(env)
    except ImportError:
        pass

    # Fall back to classic Gym
    try:
        import gym
        if isinstance(env, gym.Env):
            class GymScaledRewardWrapper(gym.RewardWrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.scale = scale
                    self.bias = bias

                def reward(self, reward):
                    return reward * self.scale + self.bias
                    
            return GymScaledRewardWrapper(env)
    except ImportError:
        pass

    raise TypeError(f"Environment {type(env)} is neither a valid Gym nor Gymnasium environment.")