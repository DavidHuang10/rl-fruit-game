# TODO — CS 372 Final Project
**Due: April 26, 2026 at 11:59pm**

---

## ML (Category 1)

### DQN — core build
- [x] `agents/network.py` — SuikaQNetwork (Q-network in PyTorch) + obs_to_tensor encoder
  - Set-aggregation encoding: mean + max pool over active fruits → flat feature vector
  - 2-layer MLP head → Q-values per action
- [x] `agents/replay_buffer.py` — circular replay buffer, compressed storage
- [x] `agents/dqn.py` — DQNAgent with:
  - Main (online) network + target network (hard copy every N steps)
  - Epsilon-greedy with linear decay
  - Adam optimizer with gradient clipping
  - LR scheduling
  - save/load checkpoints
- [x] `scripts/train_dqn.py` — training loop, outputs reward + loss curves to `results/dqn/`

### Agents for comparison
- [x] `agents/random_agent.py` — uniform random action selection
- [x] `agents/center_agent.py` — always drops at center bin
- [x] `scripts/train_ppo_sb3.py` — SB3 PPO with meaningful configuration (not defaults):
  - MultiInputPolicy over Dict observations
  - Tune: n_steps, batch_size, learning_rate, net_arch
  - Save model + training curves to `results/ppo/`

### Evaluation
- [x] `scripts/eval_agent.py` — run any agent for N episodes, collect:
  - Episode reward (RL reward signal)
  - Game score (actual Suika score)
  - Episode length (steps survived)
  - Final fruit count and final physics-frame count
- [ ] `scripts/compare_agents.py` — run all 4 agents (DQN, PPO, random, center), generate:
  - Comparison table (markdown + PNG)
  - Side-by-side reward distribution plot
- [ ] Re-run DQN/PPO evals for 50 episodes after final cluster checkpoints are ready

### Analysis (do after training)
- [ ] Watch each agent play, write qualitative analysis for README:
  - What behaviors did DQN learn?
  - How does PPO differ?
  - What do the heuristics reveal about the problem?
  - Where does the DQN fail (failure mode description)?
- [x] Update `ATTRIBUTION.md` — substantive account: what was AI-generated, what was directed/debugged/reworked by you

---

## Following Directions (Category 2)

### Documentation
- [x] `SETUP.md` — step-by-step install instructions (uv/venv, dependencies, how to train, how to eval)
- [ ] `README.md` — add/complete these sections:
  - **What it Does** — 1 paragraph, non-technical
  - **Quick Start** — how to run training and evaluation
  - **Evaluation** — quantitative results table, convergence plots
  - **Video Links** — fill in after recording

### Videos
- [ ] **Demo video** — correct length, no code shown, non-specialist audience
- [ ] **Technical walkthrough** — walk through code structure, explain DQN (main + target network, replay buffer), key design decisions

### Submission
- [ ] Self-assessment on Gradescope — use `self_assessment_template.docx`, claim exactly the 15 items in `CHECKLIST.md`
