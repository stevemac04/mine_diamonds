"""Gymnasium adapter around the `crafter` package.

Crafter (https://github.com/danijar/crafter) is a 2D Minecraft-inspired RL
environment with a full crafting tech tree:

    collect_wood -> place_table -> make_wood_pickaxe -> collect_stone ->
    place_stone -> make_stone_pickaxe -> collect_coal -> place_furnace ->
    collect_iron -> make_iron_pickaxe -> collect_diamond

The upstream package uses the legacy gym API (4-tuple step, old BoxSpace),
so we wrap it to satisfy stable-baselines3 2.x + gymnasium 1.x.

Reward model (from crafter): +1 per newly unlocked achievement,
-0.1 per point of HP lost, +0.1 per point of HP gained, -0.1 on death.
This means training reward is directly a count of "things the agent
learned to do", which is exactly what we want to show for "it learned
to craft".
"""

from __future__ import annotations

from typing import Any

import crafter
import gymnasium as gym
import numpy as np
from gymnasium import spaces


ACHIEVEMENTS: tuple[str, ...] = (
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
)

CRAFTING_ACHIEVEMENTS: frozenset[str] = frozenset(
    a for a in ACHIEVEMENTS if a.startswith(("make_", "place_"))
)


class CrafterGymnasiumEnv(gym.Env):
    """Thin Gymnasium wrapper over `crafter.Env`.

    Differences vs. upstream crafter:
      * Obs/action spaces are `gymnasium.spaces.Box` / `Discrete` so that
        SB3's `CnnPolicy` + `VecTransposeImage` detect the image space.
      * `reset()` returns `(obs, info)`; `step()` returns
        `(obs, reward, terminated, truncated, info)`.
      * `info["achievements_unlocked"]` is the *set* of achievement names
        unlocked this episode so far (cumulative), and
        `info["new_achievements"]` is the subset newly unlocked this step.
        This makes per-achievement TensorBoard logging trivial.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        area: tuple[int, int] = (64, 64),
        view: tuple[int, int] = (9, 9),
        length: int = 10_000,
        seed: int | None = None,
        reward: bool = True,
    ) -> None:
        super().__init__()
        self._area = tuple(area)
        self._view = tuple(view)
        self._length = int(length)
        self._reward = bool(reward)
        self._seed = seed

        self._inner: crafter.Env | None = None
        self._unlocked: set[str] = set()

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(17)

    def _make_inner(self, seed: int | None) -> crafter.Env:
        return crafter.Env(
            area=self._area,
            view=self._view,
            length=self._length,
            seed=seed,
            reward=self._reward,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed
        self._inner = self._make_inner(effective_seed)
        obs = self._inner.reset()
        self._unlocked = set()
        info = {
            "achievements_unlocked": frozenset(self._unlocked),
            "new_achievements": frozenset(),
        }
        return np.asarray(obs, dtype=np.uint8), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._inner is None:
            raise RuntimeError("step() called before reset()")
        obs, reward, done, info = self._inner.step(int(action))

        # crafter stores per-achievement unlock counts; > 0 means unlocked.
        achievements: dict[str, int] = info.get("achievements", {}) or {}
        newly: set[str] = set()
        for name, count in achievements.items():
            if count > 0 and name not in self._unlocked:
                self._unlocked.add(name)
                newly.add(name)

        # crafter's `done` fires on death OR length timeout; we report death
        # as terminated and length-timeout as truncated. Inner length is
        # exposed via info["discount"] == 0 on death.
        discount = float(info.get("discount", 1.0))
        terminated = bool(done and discount == 0.0)
        truncated = bool(done and not terminated)

        info = dict(info)
        info["achievements_unlocked"] = frozenset(self._unlocked)
        info["new_achievements"] = frozenset(newly)
        # Strip the raw achievements dict and semantic map from info; SB3
        # loggers choke on exotic types and they're huge in VecEnv buffers.
        info.pop("semantic", None)

        return (
            np.asarray(obs, dtype=np.uint8),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self) -> np.ndarray | None:
        if self._inner is None:
            return None
        # crafter.Env.render() returns the same RGB frame as the observation.
        frame = self._inner.render()
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        self._inner = None
