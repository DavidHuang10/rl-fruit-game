# CS 372 Rubric Checklist

**Goal:** 75 pts (Category 1 — ML, 2-point buffer)  
**Due:** April 26, 2026 at 11:59 pm

---

## 15 Selected Items

| Done | # | Item | Pts | Evidence |
|------|---|------|-----|----------|
| ✅ | 1  | Modular code design | 3 | `suika_env/` package, `agents/` package |
| ✅ | 2  | Baseline model for comparison (random + center agents) | 3 | `agents/random_agent.py`, `agents/center_agent.py`; 50-episode evals in `results/eval/` |
| ✅ | 3  | Properly normalized input features | 3 | `suika_env/env.py:_get_obs()` — x/W, y/H, vx/800, vy/800 |
| ✅ | 4  | GPU/MPS training | 3 | `agents/dqn.py:_auto_device()` — cuda → mps → cpu |
| ✅ | 5  | Custom neural network architecture (Q-network) | 5 | `agents/network.py` — DeepSets per-fruit MLP + mean/max pool |
| ✅ | 6  | Used Gymnasium API | 3 | `suika_env/env.py` — full gymnasium.Env subclass |
| ✅ | 7  | Convergence/learning curves + reward plots | 3 | `results/dqn/learning_curve.png`, `metrics.csv` |
| ✅ | 8  | Developed DQN (replay buffer + target network) | 10 | `agents/dqn.py` — Double DQN, `agents/replay_buffer.py` |
| ✅ | 9  | Custom Gymnasium environment + reward justification | 7 | `suika_env/` — Suika env with merge-score reward |
| ✅ | 10 | Used PPO through library (SB3), meaningfully configured | 5 | `scripts/train_ppo_sb3.py` — SB3 PPO `MultiInputPolicy`, n_steps, batch_size, lr, GAE, entropy coef, 4 envs, custom pi/vf net_arch; artifacts in `results/ppo/` |
| ✅ | 11 | 3+ distinct evaluation metrics | 3 | `scripts/eval_agent.py` — episode return, game score, episode length, final fruit count, physics frames |
| ⬜ | 12 | Compared multiple approaches quantitatively | 7 | `results/eval/*_summary.json` has all 4 agents, but DQN/PPO are preliminary 1-episode evals; still need final 50-episode table/plot |
| ⬜ | 13 | Qualitative + quantitative evaluation | 5 | README Evaluation section — behavioral analysis + failure modes |
| ✅ | 14 | Regularization / training stabilization | 5 | `agents/dqn.py` — Adam L2 weight decay (`weight_decay=1e-5`), gradient clipping (`clip_grad_norm_`), Huber loss; PPO entropy coefficient in `scripts/train_ppo_sb3.py` |
| ✅ | 15 | Completed project individually | 10 | — |

**Total: 75 pts**

---

## Risks to Keep in Mind

- **Items 5 + 8** — defend as distinct in self-assessment: #5 = `agents/network.py` (NN design), #8 = `agents/dqn.py` (RL algorithm)
- **Item 10** — write up PPO as SB3 `MultiInputPolicy` over dict observations. Do not claim a flatten wrapper unless one is added.
- **Items 12 + 13** — keep clearly distinct: #12 = numbers/table across agents, #13 = behavioral write-up of DQN specifically
- **Regularization item** — strongest evidence is DQN L2 weight decay. Gradient clipping, Huber loss, and PPO entropy coefficient are useful supporting evidence, but lead with weight decay.
- **Final eval risk** — random/center have 50-episode summaries, but current DQN/PPO eval summaries are only 1 episode each. Re-run DQN/PPO evals after cluster training finishes, then generate a comparison table/plot before treating item 12 as complete.

---

## Still To Build

1. Re-run DQN/PPO evaluation with the final cluster checkpoints, preferably matching the 50-episode baseline evals — item 12
2. `scripts/compare_agents.py` or equivalent generated comparison table/plot from the four eval summaries — item 12
3. README Evaluation section write-up with quantitative table plus qualitative behavior/failure-mode analysis — item 13
4. Demo video, technical walkthrough, and Gradescope self-assessment
