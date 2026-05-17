import os
import ale_py
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Import the custom feature wrapper provided in the assignment
from pong_features import PongFeaturesWrapper


class RawScoreCallback(BaseCallback):
    """
    Custom callback for logging the raw game score (unshaped return).
    It extracts the data captured by gym.wrappers.RecordEpisodeStatistics.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_timesteps = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                # Extract unshaped accumulated reward
                r = info["episode"]["r"]
                # Safely handle potential array structures depending on gym version
                if isinstance(r, (list, np.ndarray)):
                    r = r[0]
                self.episode_rewards.append(float(r))
                self.episode_timesteps.append(self.num_timesteps)
        return True


# =========================================================
# Reward Shaping Strategies
# =========================================================

class MovingTowardsBallRewardWrapper(gym.Wrapper):
    """
    Strategy 1: Reward based on the player's paddle moving towards the ball's y-coordinate.
    Uses a potential-based shaping approach that rewards the agent when the distance 
    between the paddle and the ball decreases, and penalizes it when the distance increases.
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_distance = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # obs: [bx, by, bvx, bvy, py, pvy, oy, ovy]
        self.prev_distance = abs(obs[1] - obs[4])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        by, py = obs[1], obs[4]
        
        curr_distance = abs(by - py)
        # Positive reward if distance decreases, negative if it increases
        shaping_reward = (self.prev_distance - curr_distance) * 2.0 
        self.prev_distance = curr_distance
        
        return obs, reward + shaping_reward, terminated, truncated, info


class HitRewardWrapper(gym.Wrapper):
    """
    Strategy 2: Reward based on hitting the ball.
    Provides a strong sparse reward (+1.0) when the ball's x-velocity reverses
    from moving towards the player to moving away, signifying a successful hit.
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_bvx = 0.0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_bvx = obs[2]
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bx, bvx = obs[0], obs[2]
        
        shaping_reward = 0.0
        # Player is on the right side.
        # If the ball was approaching (bvx > 0), is now bouncing away (bvx < 0), 
        # and the collision occurred on the player's side of the court (bx > 0.8)
        if self.prev_bvx > 0 and bvx < 0 and bx > 0.8:
            shaping_reward = 1.0
            
        self.prev_bvx = bvx
        return obs, reward + shaping_reward, terminated, truncated, info


class DenseAndPenaltyRewardWrapper(gym.Wrapper):
    """
    Strategy 3: Combined Dense distance + Penalty.
    Gives a continuous penalty based on the y-distance to the ball to encourage 
    tight tracking, and a sharp penalty if the ball slips behind the player's paddle.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bx, by, py = obs[0], obs[1], obs[4]
        
        # 1. Dense distance penalty
        distance = abs(by - py)
        shaping_reward = -distance * 0.1
        
        # 2. Strong penalty if the ball gets behind the player
        # The player's paddle is located roughly at normalized x = 0.95-0.97
        if bx > 0.95:
            shaping_reward -= 1.0
            
        return obs, reward + shaping_reward, terminated, truncated, info


# =========================================================
# Training and Evaluation
# =========================================================

def make_env(wrapper_class=None):
    """
    Constructs the environment stack ensuring that statistics capture the raw
    episode return BEFORE applying the custom reward shaping wrapper.
    """
    def _init():
        env = gym.make("ALE/Pong-v5")
        # 1. Map raw image pixels to the 8-D normalized feature vector
        env = PongFeaturesWrapper(env)
        # 2. Track raw episode scores and lengths (stored in the 'info' dict)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # 3. Apply the specific reward shaping wrapper (if not the Baseline run)
        if wrapper_class is not None:
            env = wrapper_class(env)
        return env
    return _init


def smooth(scalars, weight=0.85):
    """
    Exponential moving average smoothing for cleaner plotting of noisy RL data.
    """
    if not scalars:
        return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def main():
    timesteps = 10_000_000
    
    strategies = {
        # "Baseline (No Shaping)": None,
        # "Strategy 1 (Tracking Ball Y)": MovingTowardsBallRewardWrapper,
        # "Strategy 2 (Hit Reward)": HitRewardWrapper,
        "Strategy 3 (Dense Dist + Miss Penalty)": DenseAndPenaltyRewardWrapper
    }
    
    results = {}
    
    for name, wrapper in strategies.items():
        print(f"Training {name} for {timesteps} steps...")
        
        # Create vectorized environment for SB3
        env = DummyVecEnv([make_env(wrapper)])
        
        # Use MlpPolicy because our observations are flat 8-D vectors
        model = A2C("MlpPolicy", env, verbose=0, seed=42)
        callback = RawScoreCallback()
        
        model.learn(total_timesteps=timesteps, callback=callback)
        
        results[name] = {
            "timesteps": callback.episode_timesteps,
            "rewards": callback.episode_rewards
        }
        
        env.close()
        print(f"Finished {name}. Logged {len(callback.episode_rewards)} episodes.\n")


    # =========================================================
    # Plotting Learning Curves
    # =========================================================
    plt.figure(figsize=(12, 7))
    colors = ['red']
    
    for (name, data), color in zip(results.items(), colors):
        t = data["timesteps"]
        r = data["rewards"]
        if not t:
            continue
            
        # Plot raw noisy data lightly in the background
        plt.plot(t, r, color=color, alpha=0.15)
        
        # Plot exponential moving average as the main line
        smoothed_r = smooth(r, weight=0.85)
        plt.plot(t, smoothed_r, label=name, color=color, linewidth=2.5)
        
    plt.title("A2C on ALE/Pong-v5 (8D Features) - Reward Shaping Comparison", fontsize=14)
    plt.xlabel("Environment Timesteps", fontsize=12)
    plt.ylabel("Raw Game Score (Return [-21 to 21])", fontsize=12)
    plt.ylim(-22, 22)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    # Save to disk and display
    plt.savefig("pong_a2c_reward_shaping.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()