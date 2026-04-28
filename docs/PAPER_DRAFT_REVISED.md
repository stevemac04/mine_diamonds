> **Note:** Use **`docs/PAPER_WORKING_DRAFT.md`** for the class submission. This file is an older intro/methods variant kept for reference.

# Reinforcement Learning Minecraft Agent

Daniel Cusack & Stephan MacDougall  
Professor Kevin Gold  
DS340 Machine Learning and AI  
April 27, 2026

## Introduction

Reinforcement learning has achieved strong performance in structured environments such as board games, robotics simulations, and arcade games, where agents operate within clearly defined objectives and constrained state/action spaces. Open-world games are a harder setting because agents must act under sparse rewards, incomplete information, and long chains of dependent actions. Minecraft is one of the clearest examples of this challenge, since progression requires exploration, resource gathering, crafting, and planning over many intermediate steps.

In Minecraft, players spawn in a procedurally generated world and must gather resources to progress. Early gameplay usually starts with finding trees and collecting wood by punching logs. Wood can be converted into planks and used to craft basic tools such as pickaxes, axes, and shovels. These tools then enable faster progression toward stronger materials and deeper game stages. Although this sequence is intuitive for human players, it is difficult for reinforcement learning agents because each stage depends on prerequisite actions in the correct order.

Our original goal was to train an agent that could progress from spawn toward eventually obtaining diamonds. During development, we found that direct training in standard Minecraft worlds introduced major scope constraints: large terrain variation, long-horizon dependencies, unstable resets, and sparse success events. We therefore narrowed the final project objective to a realistic early-game milestone: reliably identifying trees, mining wood, and verifying inventory acquisition from the live game window.

## Methodology

The agent was trained and evaluated directly in Minecraft Java Edition running on Windows. Instead of a simulation-only environment, the system interacted with the live client through window capture and synthetic keyboard/mouse control. Observations were pixel frames from the game window, and actions were discrete control choices mapped to movement, camera control, and attack.

A core early obstacle was visual complexity: Minecraft contains many block types and visual variants, including multiple wood textures. To reduce unnecessary visual entropy for this project, we used a simplified texture pack in which target wood textures are visually standardized. This made reward-aligned log detection feasible with a lightweight color-mask pipeline.

Reward shaping was required because sparse success alone ("collected a log") produced too little learning signal in early training. We decomposed the task into intermediate behaviors and assigned dense rewards for: (1) seeing log evidence in the frame, (2) increasing visible log area (approach), (3) centering logs in the view, and (4) attacking while aimed. A sparse acquisition reward remained the terminal objective.

To implement this, we used two key visual regions:

1. **Fovea (center region):** used to estimate whether the agent is aimed at likely log pixels.  
2. **Hotbar slot 1:** used as acquisition confirmation. At reset, an empty-slot baseline image is recorded; runtime slot changes are measured against this baseline to detect actual pickup events.

False positives in acquisition were handled with stricter confirmation gates: threshold margins, consecutive-frame confirmation, and minimum mining duration before counting success. We also added behavior-level constraints in scripted validation runs to reduce mining interruption and oscillation around targets.

Training used PPO with a CNN policy. We logged both policy diagnostics and task-specific metrics (for example, episode log rate, aim rate, and per-episode JSONL traces) and generated paper-ready figures with repository scripts.

## Results

Use `docs/PAPER_RESULTS_ASSETS.md` as the source of final figures and tables.  
The recommended core set is:

- Figure: `eval/smoke_minecraft/capture_after_input.png` (agent view)
- Figure: `docs/figs/mc_real_v14_no_kill_reset/cumulative_logs.png`
- Figure: `docs/figs/mc_real_v14_no_kill_reset/episodes_aim_rate.png`
- Table: multi-run comparison (`docs/metrics/summary.md`)
- Table: headline metrics and PPO diagnostics (`docs/figs/mc_real_v14_no_kill_reset/results_summary.md`)

## Conclusion

This project shows that applying RL in a live open-world game environment is as much a systems problem as an algorithm problem. The largest blockers were not only PPO hyperparameters, but also robust interaction with the game client: stable capture, reliable action injection, reset safety, and trustworthy success detection.

We found that narrowing scope to a concrete early-game objective was essential for progress. Breaking the full objective into measurable sub-behaviors, then shaping rewards around those sub-behaviors, produced a workable training loop in a setting where naive sparse-reward learning repeatedly failed. Beyond model performance, the project emphasized iterative debugging, instrumentation, and environment control as necessary components of practical reinforcement learning in non-simulated game environments.
