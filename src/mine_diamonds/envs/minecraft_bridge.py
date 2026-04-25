"""
Placeholder for Java Minecraft integration (e.g. MineRL / Malmo).

End state: Gymnasium Env whose step() sends actions to the game client and
reads observations (pixels + inventory + pose). For now, train on
`SimpleGatherEnv` and swap the env factory when MineRL is installed and working.
"""


class MinecraftBridgeNotConfiguredError(RuntimeError):
    pass


def make_minecraft_env():
    raise MinecraftBridgeNotConfiguredError(
        "Real Minecraft env not wired yet. Use mine_diamonds.envs.SimpleGatherEnv "
        "or install/configure MineRL (often needs JDK 8 and a supported Python)."
    )
