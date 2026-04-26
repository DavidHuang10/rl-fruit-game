# TODO — CS 372 Final Project
**Due: April 26, 2026 at 11:59pm**

---

## Remaining (do these to finish the project)

### Videos
- [ ] **Demo video** — 3-5 minutes, non-technical audience, no code shown. Show DQN playing Suika and explain why the problem is interesting.
- [ ] **Technical walkthrough** — 5-10 minutes. Walk through code structure (`src/suika_env/`, `src/agents/`, `src/scripts/`), explain DQN internals (experience replay, target network, Bellman update), explain DeepSets Q-network design choice, show training curves + eval results.

### Submission
- [ ] **README video links** — save/upload videos and make sure the `## Video Links` section points to the final demo and walkthrough files
- [ ] **Gradescope self-assessment** — use `self_assessment_template.docx`, claim exactly the 15 items in `CHECKLIST.md`. For each item, cite the specific file and line numbers as evidence.

---

## Completed

### ML (Category 1)

#### DQN — core build
- [x] `src/agents/network.py` — SuikaQNetwork (DeepSets per-fruit MLP 13→128→128, mean+max pool, head 266→256→256→32)
- [x] `src/agents/replay_buffer.py` — circular replay buffer, compressed storage
- [x] `src/agents/dqn.py` — Double DQN: main + target network, epsilon-greedy, Adam + cosine LR, gradient clipping, L2 weight decay, Huber loss, save/load checkpoints
- [x] `src/scripts/train_dqn.py` — training loop, reward + loss curves to `results/dqn/`
- [x] `scripts/train_dqn_slurm.sh` — SLURM cluster job (500k steps, CUDA)

#### PPO
- [x] `src/scripts/train_ppo_sb3.py` — SB3 PPO `MultiInputPolicy` over Dict obs, tuned n_steps/batch_size/lr/net_arch/GAE/entropy coef, 4 parallel envs; artifacts in `results/ppo/`
- [x] `scripts/train_ppo_slurm.sh` — SLURM cluster job

#### Baselines
- [x] `src/agents/random_agent.py` — uniform random action selection
- [x] `src/agents/center_agent.py` — always drops at center bin

#### Evaluation
- [x] `src/scripts/eval_agent.py` — 50-episode eval for any agent; outputs episode return, game score, episode length, final fruit count, physics frames
- [x] `src/scripts/compare_agents.py` — comparison table + bar chart with error bars (`results/eval/comparison.png`)
- [x] All 4 agents evaluated for 50 episodes; summaries in `results/eval/`

### Environment (Category 1)
- [x] `src/suika_env/` — full Gymnasium environment (37/37 tests passing, `check_env` clean)
- [x] Merge-score reward with design justification in README
- [x] Dict observation space: padded fruit positions/types, mask, current/next fruit

### Documentation (Category 2)
- [x] `SETUP.md` — step-by-step install + train + eval instructions
- [x] `ATTRIBUTION.md` — substantive account of AI-generated vs. human-directed work
- [x] `requirements.txt` — accurate dependency list
- [x] `README.md` — What it Does, Quick Start, Video Links, Evaluation (table + qualitative analysis + plots), Individual Contributions, Environment, Project Structure

### Category 3 — Cohesion
- [x] README articulates single unified goal (train RL agents on custom Suika env, compare DQN vs PPO vs heuristics)
- [x] Design choices justified in README (DeepSets, merge-score reward, dict obs, SB3 PPO)
- [x] Evaluation metrics match project objective (return = game score)
- [x] Progression: problem → env → agents → training → eval → analysis
