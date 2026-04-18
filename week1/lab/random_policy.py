"""
random_policy.py  –  Random policy and rollout utility for the Crawler
=======================================================================

Classes
-------
  RandomPolicy   – uniform random policy; .act() and .learn() interface

Functions
---------
  get_rollout(policy, n_steps, render)
      Run one episode and return a list of (obs, action, reward, info) tuples.
"""

import random
from crawler_env import CrawlerEnv, plot_trajectory


class RandomPolicy:
    """Uniform random policy.  Acts randomly; learn() is a no-op."""

    def act(self, obs):
        """Return a random legal action for the given obs."""
        return random.choice(CrawlerEnv.get_legal_actions(obs))

    def learn(self, trajectories):
        """No-op – random policy does not learn."""
        pass


def get_rollout(policy, n_steps=400, render=False):
    """
    Run one episode with the given policy and return the trajectory.

    Parameters
    ----------
    policy   : object with an .act(obs) method
    n_steps  : number of steps to run
    render   : if True, open a pygame window at 10 fps

    Returns
    -------
    trajectory : list of (obs, action, reward, info) tuples, one per step
    """
    env = CrawlerEnv(render=render)
    obs, _ = env.reset()
    trajectory = []
    for _ in range(n_steps):
        prev_obs = obs
        action   = policy.act(obs)
        obs, reward, _, _, info = env.step(action)
        trajectory.append((prev_obs, action, reward, info))
    env.close()
    return trajectory


if __name__ == "__main__":
    policy = RandomPolicy()
    traj   = get_rollout(policy, n_steps=400, render=True)

    print(f"{'step':>5}  {'theta1':>8}  {'theta2':>8}  {'action':>6}  {'reward':>8}  {'total':>8}")
    print("-" * 58)
    for i, (obs, action, reward, info) in enumerate(traj):
        print(f"{i+1:>5}  {obs[0]:>8}  {obs[1]:>8}  {action:>6}  "
              f"{reward:>8.3f}  {info['dist']:>8.3f}")

    traj_norender = get_rollout(policy, n_steps=400, render=False)
    print (f"Total reward (no render): {sum(r for _, _, r, _ in traj_norender):.3f}")

    # print all states
    all_states = CrawlerEnv().all_states()
    print (len(all_states), "states total")
    print (all_states[:5], "...")