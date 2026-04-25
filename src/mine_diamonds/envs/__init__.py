from gymnasium.envs.registration import register

from mine_diamonds.envs.simple_gather import SimpleGatherEnv

register(
    id="MineDiamonds/SimpleGather-v0",
    entry_point="mine_diamonds.envs.simple_gather:SimpleGatherEnv",
    max_episode_steps=200,
)


# Crafter is an optional dependency (not required for SimpleGatherEnv smoke
# tests). Register lazily so import-time errors don't break the grid env.
try:
    from mine_diamonds.envs.crafter_env import (  # noqa: F401
        ACHIEVEMENTS,
        CRAFTING_ACHIEVEMENTS,
        CrafterGymnasiumEnv,
    )

    register(
        id="MineDiamonds/Crafter-v0",
        entry_point="mine_diamonds.envs.crafter_env:CrafterGymnasiumEnv",
    )
    _CRAFTER_OK = True
except Exception:  # pragma: no cover — crafter not installed yet
    _CRAFTER_OK = False


__all__ = ["SimpleGatherEnv"]
if _CRAFTER_OK:
    __all__ += ["CrafterGymnasiumEnv", "ACHIEVEMENTS", "CRAFTING_ACHIEVEMENTS"]
