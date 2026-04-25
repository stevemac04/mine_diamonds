from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class GatherRewardConfig:
    """Tunable reward shaping for experiments."""

    step_penalty: float = -0.01
    wrong_collect_penalty: float = -0.05
    success_bonus: float = 10.0
    # Potential-based shaping: prev_dist - dist (scaled); 0 disables
    distance_scale: float = 0.02


class SimpleGatherEnv(gym.Env):
    """
    Minimal grid world: move to a visible resource cell and "collect".

    Observation (normalized): agent x/y, resource x/y, collected flag.
    Actions: noop, north, south, west, east, collect.

    This stands in for "see resource → approach → interact" before MineRL wiring.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 11,
        max_steps: int = 200,
        reward_config: GatherRewardConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        if grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.reward_cfg = reward_config or GatherRewardConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(6)
        low = np.zeros(5, dtype=np.float32)
        high = np.ones(5, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self._rng: np.random.Generator | None = None
        self._steps = 0
        self._agent = np.zeros(2, dtype=np.int64)
        self._resource = np.zeros(2, dtype=np.int64)
        self._collected = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = self.np_random
        self._steps = 0
        self._collected = False

        self._agent = self._rng.integers(0, self.grid_size, size=2)
        self._resource = self._rng.integers(0, self.grid_size, size=2)
        while np.array_equal(self._agent, self._resource):
            self._resource = self._rng.integers(0, self.grid_size, size=2)

        return self._get_obs(), {}

    def _dist(self) -> float:
        return float(np.linalg.norm(self._agent.astype(np.float64) - self._resource.astype(np.float64)))

    def _get_obs(self) -> np.ndarray:
        g = float(self.grid_size - 1)
        ax, ay = float(self._agent[0]), float(self._agent[1])
        rx, ry = float(self._resource[0]), float(self._resource[1])
        return np.array(
            [ax / g, ay / g, rx / g, ry / g, 1.0 if self._collected else 0.0],
            dtype=np.float32,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._steps += 1
        cfg = self.reward_cfg
        reward = float(cfg.step_penalty)
        terminated = False
        info: dict = {"distance": self._dist()}

        if self._collected:
            truncated = self._steps >= self.max_steps
            return self._get_obs(), reward, terminated, truncated, info

        prev_dist = self._dist()

        if action == 0:
            pass
        elif action in (1, 2, 3, 4):
            ax, ay = int(self._agent[0]), int(self._agent[1])
            if action == 1:
                ay = max(0, ay - 1)
            elif action == 2:
                ay = min(self.grid_size - 1, ay + 1)
            elif action == 3:
                ax = max(0, ax - 1)
            elif action == 4:
                ax = min(self.grid_size - 1, ax + 1)
            self._agent = np.array([ax, ay], dtype=np.int64)
        elif action == 5:
            if np.array_equal(self._agent, self._resource):
                self._collected = True
                reward = float(cfg.success_bonus)
                terminated = True
            else:
                reward += float(cfg.wrong_collect_penalty)
        else:
            raise ValueError(f"Invalid action {action}")

        if cfg.distance_scale != 0.0 and not self._collected:
            new_dist = self._dist()
            reward += float(cfg.distance_scale) * (prev_dist - new_dist)

        truncated = self._steps >= self.max_steps
        info["distance"] = self._dist()
        info["collected"] = self._collected
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        lines: list[str] = []
        for y in range(self.grid_size):
            row: list[str] = []
            for x in range(self.grid_size):
                pos = (x, y)
                if np.array_equal(self._agent, pos):
                    ch = "A"
                elif np.array_equal(self._resource, pos):
                    ch = "R"
                else:
                    ch = "."
                row.append(ch)
            lines.append(" ".join(row))
        return "\n".join(lines)
