import numpy as np
import gymnasium as gym


# ALE Pong pixel colors
_PLAYER_COLOR   = np.array([92,  186, 92],  dtype=np.uint8)  # green  - right paddle
_OPPONENT_COLOR = np.array([213, 130, 74],  dtype=np.uint8)  # orange - left paddle
_BALL_COLOR     = np.array([236, 236, 236], dtype=np.uint8)  # white  - ball

# Frame regions
_ROW_LO, _ROW_HI = 35, 193   # excludes score area (< 34) and floor line (194)
_COL_LO, _COL_HI = 16, 144   # playfield columns

# Normalisation constants
_H = _ROW_HI - _ROW_LO   # 158
_W = _COL_HI - _COL_LO   # 128


def _centroid(frame, color, row_lo=_ROW_LO, row_hi=_ROW_HI):
    """Return (row, col) centroid of pixels matching color, or None."""
    mask = np.all(frame == color, axis=-1)
    mask[:row_lo] = False
    mask[row_hi:] = False
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return float(rows.mean()), float(cols.mean())


class PongFeaturesWrapper(gym.Wrapper):
    """
    Replaces the raw pixel observation with a 8-D feature vector:
        [ball_x, ball_y, ball_vx, ball_vy,
         player_y, player_vy,
         opponent_y, opponent_vy]

    All positions are normalised to [0, 1] over the playfield.
    Velocities are pixel-delta / _H per step (so also roughly [−1, 1]).
    When an object is not found (e.g. ball between serves) its values are 0.
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_ball     = None
        self._prev_player   = None
        self._prev_opponent = None

        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(8,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def _extract(self, frame):
        ball     = _centroid(frame, _BALL_COLOR)
        player   = _centroid(frame, _PLAYER_COLOR)
        opponent = _centroid(frame, _OPPONENT_COLOR)

        def vel(cur, prev):
            if cur is None or prev is None:
                return 0.0, 0.0
            return (cur[0] - prev[0]) / _H, (cur[1] - prev[1]) / _W

        bvy, bvx = vel(ball, self._prev_ball)
        pvy, _   = vel(player, self._prev_player)
        ovy, _   = vel(opponent, self._prev_opponent)

        def norm_pos(pos):
            if pos is None:
                return 0.0, 0.0
            r, c = pos
            return (r - _ROW_LO) / _H, (c - _COL_LO) / _W

        by, bx = norm_pos(ball)
        py, _  = norm_pos(player)
        oy, _  = norm_pos(opponent)

        self._prev_ball     = ball
        self._prev_player   = player
        self._prev_opponent = opponent

        return np.array([bx, by, bvx, bvy, py, pvy, oy, ovy], dtype=np.float32)

    def reset(self, **kwargs):
        self._prev_ball     = None
        self._prev_player   = None
        self._prev_opponent = None
        frame, info = self.env.reset(**kwargs)
        return self._extract(frame), info

    def step(self, action):
        frame, reward, terminated, truncated, info = self.env.step(action)
        return self._extract(frame), reward, terminated, truncated, info
