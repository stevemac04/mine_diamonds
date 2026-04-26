"""Gymnasium env registration for the project.

The single supported env is :class:`MinecraftRealEnv`, registered as
``MineDiamonds/MinecraftReal-v0``. It wraps a running Minecraft Java
client via screen capture + Windows SendInput and is the focus of the
RL training pipeline.

Legacy experiments live under ``archive/`` and are not imported by the
package.
"""

from gymnasium.envs.registration import register

from mine_diamonds.envs.minecraft_real import (
    MinecraftRealConfig,
    MinecraftRealEnv,
)

register(
    id="MineDiamonds/MinecraftReal-v0",
    entry_point="mine_diamonds.envs.minecraft_real:MinecraftRealEnv",
)

__all__ = ["MinecraftRealEnv", "MinecraftRealConfig"]
