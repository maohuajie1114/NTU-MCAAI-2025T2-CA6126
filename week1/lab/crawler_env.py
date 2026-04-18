#!/usr/bin/env python3
"""
crawler_env.py  –  2D Crawler RL Environment  (Gymnasium-compatible)
=====================================================================

Robot description
-----------------
  Body   : rectangle, slides left/right on the floor (no tipping).
  Arm    : L-shaped, two rigid segments attached at the front-top of the body.
             segment-1  longer, ~horizontal   angle θ1 from horizontal  [−30°, +30°]
             segment-2  shorter, ~vertical    angle θ2 from horizontal  [−120°, −60°]
             (θ2 = −90° → straight down;  +x = right, +y = up)

Observation : np.array([θ1, θ2], dtype=float32)  – angles in degrees
Actions     :  0  θ1 + step      1  θ1 − step
               2  θ2 + step      3  θ2 − step
Reward      : Δbody_x per step  (positive = rightward progress)

Physics
-------
  When the foot tip is at or below the floor (world y ≤ 0) it is *planted*.
  Any subsequent angle change rigidly slides the body so that the foot's
  world-x stays fixed.  When angles lift the foot above y = 0 it detaches.
  The body can only translate horizontally (no vertical dynamics, no torque).

Rendering
---------
  render=False   headless – no window (default, use for training)
  render=True    opens a pygame window; each env.step() draws one frame and
                 caps the loop at AGENT_FPS (10 fps) so the agent's behaviour
                 is watchable in real time.  Esc or closing the window stops
                 rendering cleanly.

  Example – watch a policy run:
      env = CrawlerEnv(render=True)
      obs, _ = env.reset()
      for _ in range(500):
          obs, reward, terminated, truncated, info = env.step(policy.act(obs))
      env.close()

Human-interactive mode (run this file directly)
------------------------------------------------
  python crawler_env.py

  W / ↑   θ1 increase    S / ↓  θ1 decrease
  A / ←   θ2 decrease    D / →  θ2 increase
  R       reset          Esc    quit

Crawling hint  (a working 4-step cycle)
----------------------------------------
  1. W  – raise θ1 to lift the foot off the ground
  2. D  – tilt θ2 toward −60° to push the foot forward (in air)
  3. S  – lower θ1 to plant the foot back on the ground
  4. A  – tilt θ2 toward −120° while planted → body is dragged forward
  Repeat.  Reward = Δbody_x per step (positive = rightward progress).
"""

import sys
import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_BASE = gym.Env
except ImportError:
    _GYM_BASE = object
    spaces = None

# ── world geometry ─────────────────────────────────────────────────────────────
BODY_W         = 80      # body width  (world units ≈ pixels at 1:1)
BODY_H         = 35      # body height  (bottom rests at y=0)
SEG1           = 65      # length of horizontal arm segment
SEG2           = 55      # length of vertical   arm segment

T1_MIN, T1_MAX =  15.0,  21.0    # θ1 limits in degrees  →  {15, 18, 21}
T2_MIN, T2_MAX = -93.0, -75.0   # θ2 limits in degrees  →  {-93, -90, ..., -75}
STEP    = 3.0                     # degrees per discrete action

# ── display ────────────────────────────────────────────────────────────────────
SW, SH      = 960, 430
FLOOR_Y     = 320    # screen-y of the floor line  (screen y increases downward)
BODY_SCR    = SW // 3
HUMAN_FPS   = 60     # interactive mode
AGENT_FPS   = 10     # agent-watched mode

C = dict(
    bg       = (232, 238, 244),
    ground   = ( 95,  95,  95),
    floor    = ( 50,  50,  50),
    tick     = (165, 165, 165),
    tick_lbl = (145, 145, 145),
    body     = ( 65, 125, 200),
    body_ol  = ( 35,  75, 140),
    seg1     = (210, 110,  45),
    seg2     = (175,  70,  25),
    joint    = (255, 210,   0),
    foot     = (215,  45,  45),
    planted  = ( 40, 205,  60),
    text     = ( 20,  20,  20),
    dim      = (130, 130, 130),
    pos      = ( 35, 145,  35),
)


def w2s(wx, wy, cam_x):
    """World coords (x→, y↑, floor at y=0) → screen coords (x→, y↓)."""
    return int(wx - cam_x + BODY_SCR), int(FLOOR_Y - wy)


# ── Environment ────────────────────────────────────────────────────────────────

class CrawlerEnv(_GYM_BASE):
    """
    Gymnasium-compatible 2D crawler.  See module docstring for full description.
    """

    metadata = {"render_fps": AGENT_FPS}

    def __init__(self, render=False):
        self.render = render

        if spaces is not None:
            self.observation_space = spaces.Box(
                low   = np.array([T1_MIN, T2_MIN], dtype=np.int32),
                high  = np.array([T1_MAX, T2_MAX], dtype=np.int32),
                dtype = np.int32,
            )
            self.action_space = spaces.Discrete(4)

        # pygame handles – created lazily on first render() call
        self._screen = None
        self._clock  = None
        self._font   = None
        self._sfont  = None
        self._cam_x  = 0.0

        self._init_state()

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if _GYM_BASE is not object:
            super().reset(seed=seed)
        self._init_state()
        if self.render:
            self._ensure_pygame()
            self._cam_x = float(self.body_x + BODY_W / 2)
            self._draw()
        return self._obs(), {}

    def step(self, action):
        """
        Apply one of four discrete actions.
        Returns (obs, reward, terminated, truncated, info).
        terminated is always False – no goal/failure state is defined.
        info contains: planted (bool), dist (float), body_x (float).
        """
        action = int(action)
        assert action in (0, 1, 2, 3), f"action must be 0-3, got {action}"

        was_planted = self._foot_on_ground()
        if was_planted:
            anchor_x = self._kinematics()[2][0]

        old_x = self.body_x

        if   action == 0: self.theta1 = min(T1_MAX, self.theta1 + STEP)
        elif action == 1: self.theta1 = max(T1_MIN, self.theta1 - STEP)
        elif action == 2: self.theta2 = min(T2_MAX, self.theta2 + STEP)
        elif action == 3: self.theta2 = max(T2_MIN, self.theta2 - STEP)

        if was_planted:
            self.body_x = anchor_x - self._foot_rel_x()

        self.last_action = action
        self.planted     = self._foot_on_ground()
        reward           = self.body_x - old_x
        self.dist    += reward
        self.n_steps += 1

        if self.render:
            self._draw()
            self._clock.tick(AGENT_FPS)
            # pump events so the window stays responsive
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.close()

        return self._obs(), reward, False, False, {
            "planted": self.planted,
            "dist":    self.dist,
            "body_x":  self.body_x,
        }

    def render(self):
        """Explicit render call – draws the current frame (agent mode)."""
        if not self.render:
            return
        self._ensure_pygame()
        self._draw()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    # ── kinematics ──────────────────────────────────────────────────────────

    def _kinematics(self):
        """Return (shoulder, elbow, foot) in world coords."""
        sx = self.body_x + BODY_W
        sy = float(BODY_H)
        t1 = math.radians(self.theta1)
        ex = sx + SEG1 * math.cos(t1)
        ey = sy + SEG1 * math.sin(t1)
        t2 = math.radians(self.theta2)
        fx = ex + SEG2 * math.cos(t2)
        fy = ey + SEG2 * math.sin(t2)
        return (sx, sy), (ex, ey), (fx, fy)

    # ── private ─────────────────────────────────────────────────────────────

    def _init_state(self):
        self.body_x  = 0.0
        self.theta1  = 15.0   # foot just barely planted within MAX_FOOT_DEPTH
        self.theta2  = -90.0
        self.dist        = 0.0
        self.n_steps     = 0
        self.last_action = None
        self.planted     = self._foot_on_ground()

    def _obs(self):
        t1 = int(round(self.theta1 / STEP) * STEP)
        t2 = int(round(self.theta2 / STEP) * STEP)
        return np.array([t1, t2], dtype=np.int32)

    def _foot_on_ground(self):
        return self._kinematics()[2][1] <= 0.5

    @staticmethod
    def get_legal_actions(obs):
        """Return the list of actions that won't hit a joint limit at obs."""
        t1, t2 = int(obs[0]), int(obs[1])
        legal = []
        if t1 < T1_MAX: legal.append(0)   # theta1+ not yet at max
        if t1 > T1_MIN: legal.append(1)   # theta1- not yet at min
        if t2 < T2_MAX: legal.append(2)   # theta2+ not yet at max
        if t2 > T2_MIN: legal.append(3)   # theta2- not yet at min
        return legal

    @staticmethod
    def all_states():
        """Return a list of every valid (theta1, theta2) obs as int tuples."""
        n1 = round((T1_MAX - T1_MIN) / STEP)
        n2 = round((T2_MAX - T2_MIN) / STEP)
        return [
            (int(T1_MIN + i * STEP), int(T2_MIN + j * STEP))
            for i in range(n1 + 1)
            for j in range(n2 + 1)
        ]

    def _foot_rel_x(self):
        t1 = math.radians(self.theta1)
        t2 = math.radians(self.theta2)
        return BODY_W + SEG1 * math.cos(t1) + SEG2 * math.cos(t2)

    def _ensure_pygame(self):
        import pygame
        if self._screen is None:
            pygame.init()
            pygame.display.set_caption("Crawler RL Environment")
            self._screen = pygame.display.set_mode((SW, SH))
            self._clock  = pygame.time.Clock()
            self._font   = pygame.font.SysFont("monospace", 16)
            self._sfont  = pygame.font.SysFont("monospace", 11)
            self._cam_x  = float(self.body_x + BODY_W / 2)

    def _draw(self):
        """Draw one frame into self._screen (camera smoothed)."""
        self._cam_x += (self.body_x + BODY_W / 2 - self._cam_x) * 0.2
        hints = [
            f"step {self.n_steps}   Esc to quit",
        ]
        _draw_frame(self._screen, self, self._font, self._sfont, self._cam_x,
                    bottom_hints=hints)


# ── shared drawing routine ─────────────────────────────────────────────────────

def _draw_frame(screen, env, font, sfont, cam_x, bottom_hints=None):
    import pygame

    screen.fill(C['bg'])

    pygame.draw.rect(screen, C['ground'], (0, FLOOR_Y, SW, SH - FLOOR_Y))
    pygame.draw.line(screen, C['floor'],  (0, FLOOR_Y), (SW, FLOOR_Y), 2)

    first_tick = int(cam_x - BODY_SCR) // 50 * 50
    for tick in range(first_tick, first_tick + SW + 100, 50):
        sx = int(tick - cam_x + BODY_SCR)
        if -20 <= sx <= SW + 20:
            pygame.draw.line(screen, C['tick'], (sx, FLOOR_Y), (sx, FLOOR_Y + 6), 1)
            lbl = sfont.render(str(tick), True, C['tick_lbl'])
            screen.blit(lbl, (sx - lbl.get_width() // 2, FLOOR_Y + 9))

    shoulder, elbow, foot = env._kinematics()
    bx, by = w2s(env.body_x, BODY_H, cam_x)
    body_rect = pygame.Rect(bx, by, BODY_W, BODY_H)
    pygame.draw.rect(screen, C['body'],    body_rect)
    pygame.draw.rect(screen, C['body_ol'], body_rect, 2)

    for wx_off in (BODY_W // 4, 3 * BODY_W // 4):
        pygame.draw.circle(screen, C['body_ol'], (bx + wx_off, FLOOR_Y), 6, 2)

    sp = w2s(*shoulder, cam_x)
    ep = w2s(*elbow,    cam_x)
    fp = w2s(*foot,     cam_x)

    pygame.draw.line(screen, C['seg1'], sp, ep, 6)
    pygame.draw.line(screen, C['seg2'], ep, fp, 5)
    pygame.draw.circle(screen, C['joint'], sp, 7)
    pygame.draw.circle(screen, C['joint'], ep, 6)

    fc = C['planted'] if env.planted else C['foot']
    pygame.draw.circle(screen, fc, fp, 7)
    if env.planted:
        pygame.draw.line(screen, C['planted'], fp, (fp[0], FLOOR_Y), 1)

    _ACTION_LABELS = {0: "0  theta1+  (W)", 1: "1  theta1-  (S)",
                      2: "2  theta2+  (D)", 3: "3  theta2-  (A)"}
    action_str = _ACTION_LABELS.get(env.last_action, "—")
    hud = [
        (f"last action    : {action_str}",                                               C['text']),
        (f"theta1 (horiz) : {env.theta1:+6.1f} deg   [{T1_MIN:+.0f}, {T1_MAX:+.0f}]", C['text']),
        (f"theta2 (vert)  : {env.theta2:+6.1f} deg   [{T2_MIN:+.0f}, {T2_MAX:+.0f}]", C['text']),
        (f"position       : {env.body_x:+.0f}",                              C['pos']),
        (f"total distance : {env.dist:+.1f}",                                            C['pos']),
        (f"foot           : {'PLANTED  [green dot]' if env.planted else 'lifted   [red dot]'}",
         C['planted'] if env.planted else C['foot']),
    ]
    if bottom_hints:
        hud.append(("", C['text']))
        for h in bottom_hints:
            hud.append((h, C['dim']))

    for i, (txt, col) in enumerate(hud):
        screen.blit(font.render(txt, True, col), (12, 10 + i * 23))

    pygame.display.flip()


# ── state-space trajectory plot ───────────────────────────────────────────────

def plot_trajectory(trajectory, title="State trajectory  (θ1 × θ2)"):
    """
    Plot the path of states visited in the θ1 × θ2 space.

    Parameters
    ----------
    trajectory : list of (obs, action, reward, info) tuples
        As returned by get_rollout() in random_policy.py.
    title : str
        Window / figure title.

    The path is drawn as a colour-gradient line (blue → red over time) with
    small arrows showing direction of travel.  Start is marked with a green
    circle, end with a red square.  All valid grid points are shown as faint
    grey dots in the background.
    """
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    import numpy as np

    states = CrawlerEnv.all_states()
    grid_t1 = [s[0] for s in states]
    grid_t2 = [s[1] for s in states]

    t1s = [int(obs[0]) for obs, *_ in trajectory]
    t2s = [int(obs[1]) for obs, *_ in trajectory]
    n   = len(t1s)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(title)

    # faint grid dots for all valid states
    ax.scatter(grid_t1, grid_t2, s=12, color="lightgrey", zorder=1, label="valid states")

    # colour-gradient path: segments coloured blue→red by time
    cmap   = plt.get_cmap("coolwarm")
    colors = [cmap(i / max(n - 1, 1)) for i in range(n - 1)]
    segments = [[(t1s[i], t2s[i]), (t1s[i+1], t2s[i+1])] for i in range(n - 1)]
    lc = mc.LineCollection(segments, colors=colors, linewidths=1.5, zorder=2)
    ax.add_collection(lc)

    # arrows every ~20 steps to show direction
    step = max(1, n // 20)
    for i in range(0, n - 1, step):
        dt1 = t1s[i+1] - t1s[i]
        dt2 = t2s[i+1] - t2s[i]
        if dt1 != 0 or dt2 != 0:
            ax.annotate("", xy=(t1s[i+1], t2s[i+1]), xytext=(t1s[i], t2s[i]),
                        arrowprops=dict(arrowstyle="->", color=cmap(i / max(n-1, 1)),
                                        lw=1.2), zorder=3)

    # start / end markers
    ax.scatter([t1s[0]],  [t2s[0]],  s=80, color="green",  marker="o",
               zorder=4, label="start")
    ax.scatter([t1s[-1]], [t2s[-1]], s=80, color="red",    marker="s",
               zorder=4, label="end")

    ax.set_xlabel("θ1 (horizontal arm, deg)")
    ax.set_ylabel("θ2 (vertical arm, deg)")
    ax.set_xlim(T1_MIN - 5, T1_MAX + 5)
    ax.set_ylim(T2_MIN - 5, T2_MAX + 5)
    ax.set_xticks(range(int(T1_MIN), int(T1_MAX) + 1, int(STEP) * 3))
    ax.set_yticks(range(int(T2_MIN), int(T2_MAX) + 1, int(STEP) * 3))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    # colour bar showing time progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="step")

    plt.tight_layout()
    plt.show()


# ── human-interactive main ─────────────────────────────────────────────────────

def main():
    """Run the crawler in human-interactive mode (WASD / arrow keys)."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("Crawler RL Environment  –  Human Play")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 16)
    sfont  = pygame.font.SysFont("monospace", 11)

    env   = CrawlerEnv(render=False)   # headless – this loop owns the rendering
    obs, _ = env.reset()
    cam_x = float(env.body_x + BODY_W / 2)

    KEY_MAP = {
        pygame.K_w: 0, pygame.K_UP:    0,
        pygame.K_s: 1, pygame.K_DOWN:  1,
        pygame.K_a: 3, pygame.K_LEFT:  3,
        pygame.K_d: 2, pygame.K_RIGHT: 2,
    }

    controls = [
        "W/↑  theta1+    S/↓  theta1-",
        "A/←  theta2-    D/→  theta2+",
        "R  reset    Esc  quit",
    ]

    print(__doc__)
    print("─" * 56)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    cam_x = float(env.body_x + BODY_W / 2)
                    print("  [reset]")
                elif event.key in KEY_MAP:
                    a = KEY_MAP[event.key]
                    obs, rew, _, _, info = env.step(a)
                    arrow = "→" if rew > 0.05 else ("←" if rew < -0.05 else "·")
                    print(f"  step {env.n_steps:4d}  "
                          f"θ1={obs[0]:+6.1f}°  θ2={obs[1]:+7.1f}°  "
                          f"reward={rew:+5.1f} {arrow}  "
                          f"planted={'Y' if info['planted'] else 'n'}  "
                          f"total={info['dist']:+.1f}")

        cam_x += (env.body_x + BODY_W / 2 - cam_x) * 0.12
        _draw_frame(screen, env, font, sfont, cam_x, bottom_hints=controls)
        clock.tick(HUMAN_FPS)


if __name__ == "__main__":
    main()
