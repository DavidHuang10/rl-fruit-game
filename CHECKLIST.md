# CS 372 Rubric Checklist

**Goal:** 73 pts (Category 1 — ML)  
**Due:** April 26, 2026 at 11:59 pm

---

## 15 Selected Items

| Done | # | Item | Pts | Evidence |
|------|---|------|-----|----------|
| ✅ | 1  | Modular code design | 3 | `suika_env/` package, `agents/` package |
| ⬜ | 2  | Baseline model for comparison (random agent) | 3 | `agents/random_agent.py`, `agents/center_agent.py` |
| ✅ | 3  | Properly normalized input features | 3 | `suika_env/env.py:_get_obs()` — x/W, y/H, vx/800, vy/800 |
| ✅ | 4  | GPU/MPS training | 3 | `agents/dqn.py:_auto_device()` — cuda → mps → cpu |
| ✅ | 5  | Custom neural network architecture (Q-network) | 5 | `agents/network.py` — DeepSets per-fruit MLP + mean/max pool |
| ✅ | 6  | Used Gymnasium API | 3 | `suika_env/env.py` — full gymnasium.Env subclass |
| ✅ | 7  | Convergence/learning curves + reward plots | 3 | `results/dqn/learning_curve.png`, `metrics.csv` |
| ✅ | 8  | Developed DQN (replay buffer + target network) | 10 | `agents/dqn.py` — Double DQN, `agents/replay_buffer.py` |
| ✅ | 9  | Custom Gymnasium environment + reward justification | 7 | `suika_env/` — Suika env with merge-score reward |
| ⬜ | 10 | Used PPO through library (SB3), meaningfully configured | 5 | `scripts/train_ppo_sb3.py` — tune n_steps, batch_size, lr, net_arch |
| ⬜ | 11 | 3+ distinct evaluation metrics | 3 | `scripts/eval_agent.py` — episode return, game score, ep length, merges |
| ⬜ | 12 | Compared multiple approaches quantitatively | 7 | `scripts/compare_agents.py` — DQN vs PPO vs random vs center table |
| ⬜ | 13 | Qualitative + quantitative evaluation | 5 | README Evaluation section — behavioral analysis + failure modes |
| ⬜ | 14 | ATTRIBUTION.md substantive account | 3 | `ATTRIBUTION.md` — what was AI-generated vs. directed/debugged |
| ✅ | 15 | Completed project individually | 10 | — |

**Total: 73 pts**

---

## Risks to Keep in Mind

- **Items 5 + 8** — defend as distinct in self-assessment: #5 = `agents/network.py` (NN design), #8 = `agents/dqn.py` (RL algorithm)
- **Items 12 + 13** — keep clearly distinct: #12 = numbers/table across agents, #13 = behavioral write-up of DQN specifically
- **Zero buffer** — every item must land. Insurance: swap #14 (3pts) for regularization/dropout (5pts) → 75pts → 2pt buffer

---

## Still To Build

1. `agents/random_agent.py`, `agents/center_agent.py` — item 2
2. `scripts/train_ppo_sb3.py` — item 10
3. `scripts/eval_agent.py` — item 11
4. `scripts/compare_agents.py` — items 12, 13
5. README Evaluation section write-up — item 13
6. `ATTRIBUTION.md` update — item 14
