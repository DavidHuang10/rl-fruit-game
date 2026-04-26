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
| ✅ | 6  | Used learning rate scheduling (cosine annealing) | 3 | `agents/dqn.py` — Adam lr=3e-4 cosine→1e-5 over 500k steps |
| ✅ | 7  | Convergence/learning curves + reward plots | 3 | `results/dqn/learning_curve.png`, `metrics.csv` |
| ✅ | 8  | Developed DQN (replay buffer + target network) | 10 | `agents/dqn.py` — Double DQN, `agents/replay_buffer.py` |
| ✅ | 9  | Custom Gymnasium environment + reward justification | 7 | `suika_env/` — Suika env with merge-score reward |
| ✅ | 10 | Used PPO through library (SB3), meaningfully configured | 5 | `scripts/train_ppo_sb3.py` — SB3 PPO `MultiInputPolicy`, n_steps, batch_size, lr, GAE, entropy coef, 4 envs, custom pi/vf net_arch; artifacts in `results/ppo/` |
| ✅ | 11 | 3+ distinct evaluation metrics | 3 | `scripts/eval_agent.py` — episode return, game score, episode length, final fruit count, physics frames |
| ✅ | 12 | Compared multiple approaches quantitatively | 7 | README Evaluation table (all 4 agents, 50 episodes each); `results/eval/comparison.png` bar chart with error bars; `scripts/compare_agents.py` |
| ✅ | 13 | Qualitative + quantitative evaluation | 5 | README Evaluation section — quantitative table + behavioral analysis + failure modes for each agent |
| ✅ | 14 | Regularization / training stabilization | 5 | `agents/dqn.py` — Adam L2 weight decay (`weight_decay=1e-5`), gradient clipping (`clip_grad_norm_`), Huber loss; PPO entropy coefficient in `scripts/train_ppo_sb3.py` |
| ✅ | 15 | Completed project individually | 10 | — |

**Total: 75 pts**

---

## Risks to Keep in Mind

- **Items 5 + 8** — defend as distinct in self-assessment: #5 = `agents/network.py` (NN architecture design — DeepSets, permutation invariance), #8 = `agents/dqn.py` (RL algorithm — experience replay, target network, Bellman update)
- **Item 6** — LR scheduling is standalone; cosine annealing schedule is separate from the optimizer choice and separate from gradient clipping claimed under #14
- **Item 10** — write up PPO as SB3 `MultiInputPolicy` over dict observations. Do not claim a flatten wrapper unless one is added.
- **Items 12 + 13** — keep clearly distinct: #12 = breadth (numbers/table across all 4 agents), #13 = depth (behavioral write-up + failure modes for DQN specifically)
- **Item 14** — lead with L2 weight decay as the primary regularization evidence. Gradient clipping and Huber loss are supporting evidence, but do not list gradient clipping as the primary argument (it overlaps with training stability, not overfitting prevention)

---

## Still To Do

1. Record demo video (non-technical, no code shown — check course site for required length)
2. Record technical walkthrough (code structure + DQN mechanism + key decisions — check course site for required length)
3. Fill in `README.md` Video Links section with actual URLs
4. Submit Gradescope self-assessment using `self_assessment_template.docx`, claiming these 15 items
