import ale_py
import gymnasium as gym
import numpy as np
from pong_features import PongFeaturesWrapper

gym.register_envs(ale_py)


def make_env(render: bool):
    mode = "human" if render else None
    return PongFeaturesWrapper(gym.make("ALE/Pong-v5", render_mode=mode))


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        return self.action_space.sample()

    def learn(self, trajectories: list[list[tuple]]) -> None:
        pass


def get_rollout(agent: RandomAgent, env, greedy: bool = False) -> tuple[float, list]:
    """
    Play one full game.
    Returns (total_reward, trajectory) where trajectory is a list of (s, a, r, s') tuples.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    trajectory = []

    while True:
        action = agent.act(obs, greedy=greedy)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        trajectory.append((obs.copy(), action, reward, next_obs.copy()))
        total_reward += reward
        obs = next_obs
        if terminated or truncated:
            break

    env.close()
    return total_reward, trajectory


if __name__ == "__main__":
    agent = RandomAgent(make_env(render=False).action_space)

    # --- 100 games, no rendering ---
    print("Running 100 games (no render)...")
    rewards = []
    for i in range(100):
        r, _ = get_rollout(agent, make_env(render=False))
        rewards.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/100  avg so far: {np.mean(rewards):.2f}")
    print(f"\nAverage reward over 100 games: {np.mean(rewards):.2f}  "
          f"(std {np.std(rewards):.2f}, min {np.min(rewards):.0f}, max {np.max(rewards):.0f})\n")

    # --- 1 game with rendering ---
    print("Running 1 rendered game...")
    reward, trajectory = get_rollout(agent, make_env(render=True))
    print(f"Episode reward: {reward:.0f}  |  trajectory length: {len(trajectory)} steps")
