"""Train a reinforcement learning model on the gym environment."""

from stable_baselines3 import PPO

from .env import SymbioteEnv


def train():
    """Train a PPO model."""
    env = SymbioteEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("symbiote_ppo")


if __name__ == "__main__":
    train()
