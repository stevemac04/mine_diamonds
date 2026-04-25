# mine-diamonds

Reinforcement-learning scaffolding toward Minecraft-style resource gathering
and crafting. The repo has two tracks:

1. **RL track (this is the real thing):** PPO on
   [Crafter](https://github.com/danijar/crafter), a 2D Minecraft-inspired
   environment with a full crafting tech tree (wood → table → pickaxe → stone
   → coal → furnace → iron → diamond). This is where actual learning
   happens.
2. **Scripted Minecraft Java track:** non-RL demos (`scripts/demo_treechop_macro.py`,
   `scripts/demo_treechop_vision.py`) that drive the actual Minecraft client
   with keyboard/mouse and screen capture. Useful for demos and future
   screen-capture RL work; not part of the learning loop.

## Installing

```bash
# Python 3.10+ venv (3.13 confirmed working on Windows)
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[crafter]"

# For an NVIDIA Blackwell GPU (RTX 50-series, sm_120) install the matching
# CUDA Torch wheel AFTER the base install:
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
```

If `pip install crafter` fails on Windows with a `UnicodeDecodeError`, run
with `PYTHONUTF8=1` set (it's a bug in crafter's `setup.py` README read).
The training/eval scripts set this automatically.

## Quickstart

### 1. Smoke test the env

```bash
python scripts/smoke_crafter.py
```

Random policy for 1500 steps; verifies spaces and prints the achievements a
random agent happens to unlock (typically 4–5 easy ones).

### 2. Short sanity training (~2 minutes on GPU)

```bash
python scripts/train_ppo_crafter.py \
    --total-steps 16384 --n-envs 4 --run-name crafter_smoke \
    --checkpoint-every 0 --no-subproc
```

You should see `rollout/ep_rew_mean` and `crafter/ep_ach_count_mean` climb,
and `crafter/unlock_rate/place_table`, `crafter/unlock_rate/make_wood_sword`
start lifting off 0.

### 3. Real training run (hours, GPU-backed)

```bash
python scripts/train_ppo_crafter.py \
    --total-steps 3000000 --n-envs 16 \
    --run-name crafter_ppo_long
```

Monitor with:

```bash
tensorboard --logdir runs
```

### 4. Evaluate a checkpoint

```bash
python scripts/eval_crafter.py \
    --checkpoint checkpoints/crafter_ppo_long/ppo_crafter_final.zip \
    --episodes 10 --run-name crafter_ppo_long --deterministic
```

Produces:

* `eval/<run>/rollout.mp4`   — one episode of upscaled gameplay.
* `eval/<run>/summary.json`  — per-achievement unlock rate over all episodes.
* `eval/<run>/episodes.jsonl`— per-episode return, length, and unlocked set.

## What "learns to craft" means here

Crafter's reward is `+1` for every *newly unlocked* achievement in an
episode, so `ep_ach_count_mean` is literally "average count of distinct
things the agent learned to do per episode." The 22 achievements include:

| Tier              | Achievements                                                                 |
|-------------------|------------------------------------------------------------------------------|
| Free / easy       | `wake_up`, `collect_sapling`, `place_plant`, `collect_drink`, `collect_wood` |
| Wood crafting     | `place_table`, `make_wood_pickaxe`, `make_wood_sword`                        |
| Stone crafting    | `collect_stone`, `place_stone`, `make_stone_pickaxe`, `make_stone_sword`     |
| Mid-game          | `collect_coal`, `place_furnace`, `eat_cow`, `eat_plant`                      |
| Combat            | `defeat_zombie`, `defeat_skeleton`                                           |
| Late tech tree    | `collect_iron`, `make_iron_pickaxe`, `make_iron_sword`, `collect_diamond`    |

PPO reliably learns the wood-tier crafts within a few hundred thousand
environment steps on this machine. Stone tier typically needs 1–3M steps,
and the iron/diamond tier is a stretch goal that published Crafter
baselines need millions to tens of millions of steps (plus reward shaping
or curiosity bonuses) to hit.

## Non-RL demos

The original scripted demos still work; see their top-of-file docstrings:

* `scripts/demo_treechop_macro.py`  — deterministic WASD + LMB tree chop.
* `scripts/demo_treechop_vision.py` — OpenCV color-mask vision loop.
* `scripts/train_ppo_baseline.py`   — PPO on the toy `SimpleGatherEnv`
  grid world (retained as a sanity check that the training loop itself works
  without image obs).

## Not yet wired

* **Real Minecraft Java via MineRL / Malmo** — `minecraft_bridge.py`
  remains a placeholder. MineRL requires JDK 8 and Python 3.8–3.10 and is
  currently not installable in this venv.
* **Screen-capture Gymnasium env wrapping the actual MC client** —
  possible using the existing `mine_diamonds.input` and
  `mine_diamonds.vision` modules, but non-trivial to define a reliable
  crafting reward signal. Deferred.

## Layout

```
src/mine_diamonds/
  envs/
    simple_gather.py          toy grid env
    crafter_env.py            Gymnasium adapter over crafter.Env
    minecraft_bridge.py       MineRL placeholder (unimplemented)
  input/                      SendInput / pyautogui for scripted demos
  vision/                     BGR color masks for the custom MC texture pack
scripts/
  smoke_env.py                SimpleGather sanity
  smoke_crafter.py            Crafter sanity
  train_ppo_baseline.py       PPO on SimpleGather
  train_ppo_crafter.py        PPO on Crafter (the RL run)
  eval_crafter.py             checkpoint -> MP4 + per-achievement summary
  demo_treechop_macro.py      scripted MC macro
  demo_treechop_vision.py     scripted MC vision loop
runs/        tensorboard logs (gitignored)
checkpoints/ PPO checkpoints   (gitignored)
eval/        eval artifacts    (gitignored)
```
