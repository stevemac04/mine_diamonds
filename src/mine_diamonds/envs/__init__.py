from gymnasium.envs.registration import register

from mine_diamonds.envs.simple_gather import SimpleGatherEnv

register(
    id="MineDiamonds/SimpleGather-v0",
    entry_point="mine_diamonds.envs.simple_gather:SimpleGatherEnv",
    max_episode_steps=200,
)

__all__ = ["SimpleGatherEnv"]
