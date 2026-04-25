"""PPO baseline on SimpleGatherEnv — proves the training loop before MineRL."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mine_diamonds.envs.simple_gather import GatherRewardConfig, SimpleGatherEnv


def main() -> None:
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = ROOT / "runs"

    cfg = GatherRewardConfig(
        step_penalty=-0.01,
        wrong_collect_penalty=-0.05,
        success_bonus=10.0,
        distance_scale=0.02,
    )

    def make_env():
        return SimpleGatherEnv(grid_size=11, max_steps=200, reward_config=cfg)

    venv = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        tensorboard_log=str(runs_dir / "ppo_gather"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
    )
    model.learn(total_timesteps=80_000)
    model.save(str(ckpt_dir / "ppo_gather_baseline"))
    print("saved", ckpt_dir / "ppo_gather_baseline.zip")


if __name__ == "__main__":
    main()
