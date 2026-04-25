# CS 372 Final Project — ML Rubric Target

**Goal:** Max out Category 1 (ML) at 73 pts using these 15 items.
**Due:** April 26, 2026 at 11:59 pm

---

## Selected Items (73 pts exactly — zero buffer)

| # | Item | Pts | Status | Risk |
|---|---|---|---|---|
| 1 | Modular code design | 3 | ✅ done | none |
| 2 | Baseline model for comparison (random agent) | 3 | needs code | low |
| 3 | Properly normalized input features | 3 | ✅ done (env.py `_get_obs`) | none |
| 4 | GPU/MPS training | 3 | needs 1 line | low |
| 5 | Custom neural network architecture (Q-network) | 5 | needs DQN | **⚠ may double-count with #8** |
| 6 | Used Gymnasium API | 3 | ✅ done | none |
| 7 | Convergence/learning curves + reward plots | 3 | needs training script | low |
| 8 | **Developed DQN** (replay + target network) | 10 | needs code | none |
| 9 | Custom Gymnasium environment + reward justification | 7 | ✅ done | none |
| 10 | Used PPO through library (SB3) | 5 | needs code | medium — must configure meaningfully |
| 11 | 3+ distinct eval metrics | 3 | needs eval script | low |
| 12 | Compared multiple approaches quantitatively | 7 | needs DQN + PPO + random trained | medium — ⚠ may overlap with #13 |
| 13 | Qualitative + quantitative evaluation | 5 | needs analysis write-up | medium — keep clearly distinct from #12 |
| 14 | ATTRIBUTION.md substantive account | 3 | needs update | low |
| 15 | Completed project individually | 10 | ✅ free | none |
| **Total** | | **73** | | |

---

## Key Risks

1. **Items 5 + 8 double-count:** Custom NN and DQN both describe the Q-network. Defend them as distinct:
   - Item 8 = the RL algorithm (replay buffer, target network, epsilon-greedy, Bellman update)
   - Item 5 = the neural network design (obs set-encoding, layer architecture in PyTorch)
   In self-assessment, point to these separately: `agents/network.py` for #5, `agents/dqn.py` for #8.

2. **Items 12 + 13 overlap:** Keep them clearly distinct in write-up:
   - Item 12 = comparison table across DQN vs PPO vs random (numbers, controlled setup)
   - Item 13 = behavioral/qualitative analysis of DQN specifically (what it learned, failure modes, gameplay description)

3. **Zero buffer:** Every item must land. If one gets rejected, cap is missed.
   - Cheapest insurance: swap item 14 (ATTRIBUTION, 3pts) for regularization dropout+weight decay (5pts) → 75pts → 2pt buffer

---

## Build Priority (ordered)

1. **DQN** (`agents/dqn.py`, `agents/network.py`, `scripts/train_dqn.py`) — covers items 5, 7, 8, and enables 12
2. **Random baseline** (`scripts/eval_agent.py`) — item 2, 5 min
3. **SB3 PPO** (`scripts/train_ppo_sb3.py`) — item 10, ~45 min, must configure meaningfully
4. **Comparison + eval** (`scripts/compare_agents.py`) — items 11, 12, 13
5. **ATTRIBUTION.md update** — item 14

---

## Self-Assessment Evidence Map (fill in after building)

| Item | File/Location |
|---|---|
| Modular code | `suika_env/` package, `agents/` package |
| Baseline comparison | `scripts/eval_agent.py`, results table in README |
| Normalized features | `suika_env/env.py:_get_obs()` lines ~220-240 |
| GPU training | `agents/dqn.py` device auto-detect |
| Custom NN | `agents/network.py` SuikaQNetwork + obs_to_tensor |
| Gymnasium | `suika_env/__init__.py` register + env.py |
| Convergence plots | `results/dqn/reward_curve.png`, `loss_curve.png` |
| Developed DQN | `agents/dqn.py` full implementation |
| Custom env | `suika_env/` — point to design doc or README justification |
| PPO SB3 | `scripts/train_ppo_sb3.py` + results |
| 3+ metrics | episode reward, game score, episode length (+ merges/ep) |
| Compared approaches | `results/comparison_table.png` or markdown table in README |
| Qual + quant eval | README Evaluation section — qualitative behavioral analysis |
| ATTRIBUTION | `ATTRIBUTION.md` |
| Solo | — |
