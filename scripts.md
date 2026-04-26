# Video Scripts

Record two videos and save them in `videos/`:

- Demo video: `videos/demo.mp4`
- Technical walkthrough: `videos/technical_walkthrough.mp4`

After recording, confirm the README links point to those files.

## Demo Video

Target length: 3-5 minutes.

Audience: non-technical. Do not show code.

Recording command:

```bash
uv run python scripts/watch_agent.py --agent dqn --episodes 1 --speed 2
```

Optional baseline clips:

```bash
uv run python scripts/watch_agent.py --agent random --episodes 1 --speed 2
uv run python scripts/watch_agent.py --agent center --episodes 1 --speed 2
```

### Script

0:00-0:30

Hi, this project is RL Fruit Game, a reinforcement-learning version of a Suika-style fruit merging game. The agent chooses where to drop each fruit. Matching fruits merge into larger fruits, the score increases, and the game ends when the stack gets too high.

0:30-1:15

The challenge is that this is a physics-based game with delayed consequences. A bad drop may not end the game immediately, but it can block future merges or trap small fruits at the bottom of the board. The agent has to learn actions that improve both immediate score and long-term board stability.

1:15-2:30

This is the trained DQN agent playing. Each turn, it observes the board state, the current fruit, and the next fruit, then chooses one of 32 drop columns. You can see that it does not simply drop randomly or always use the center. It uses different parts of the board and tries to keep space available for future merges.

2:30-3:30

I evaluated four approaches over 50 episodes each: random, center-drop, PPO, and DQN. DQN achieved the highest average score and survived the longest. The center baseline was surprisingly strong, which makes the comparison meaningful, but DQN still performed best overall.

3:30-4:30

The training curves show that DQN improved over time, while PPO was much flatter under this setup. The final project includes a custom Gymnasium environment, trained DQN and PPO agents, simple baselines, and quantitative evaluation.

Closing line:

The main result is that a custom DQN learned useful game behavior from reward signal alone and outperformed the comparison agents in this custom physics-based environment.

## Technical Walkthrough

Target length: 5-10 minutes.

Audience: grader or ML engineer. Show code and results.

### Script

0:00-0:45

This project is organized around a custom Gymnasium environment and several agents. Source code is in `src/`: `src/suika_env/` contains the environment and physics, `src/agents/` contains DQN, PPO loading, replay buffer, network, and baselines, and `src/scripts/` contains training, evaluation, comparison, and play scripts. Trained checkpoints are in `models/`, while metrics, plots, and evaluation summaries are in `results/`.

0:45-2:00

Open `src/suika_env/env.py`.

The environment exposes a discrete action space with 32 drop columns. The observation is a dictionary containing padded fruit positions, fruit types, a fruit mask, the current fruit, and the next fruit. The reward is the merge score gained after a drop, which directly matches the game score objective. Episodes terminate when the fruit stack crosses the danger line.

2:00-4:00

Open `src/agents/dqn.py` and `src/agents/replay_buffer.py`.

The main custom ML component is Double DQN. It uses an online Q-network, a target network, epsilon-greedy exploration, and replay buffer sampling. The update computes Bellman targets from the target network. Training uses Huber loss, gradient clipping, Adam with L2 weight decay, and a cosine learning-rate schedule.

4:00-5:15

Open `src/agents/network.py`.

The Q-network uses a DeepSets-style architecture. Each fruit is encoded independently, then active fruit embeddings are pooled with mean and max pooling. This is useful because the board is naturally a set of fruits, not a fixed ordered sequence. The pooled board representation is combined with current and next fruit information to produce Q-values for all 32 actions.

5:15-6:15

Open `src/scripts/train_ppo_sb3.py`.

For comparison, I trained PPO using Stable-Baselines3 with `MultiInputPolicy`, so it can consume the same dictionary observation space. PPO is meaningfully configured with rollout length, batch size, learning rate, GAE, entropy coefficient, parallel environments, and custom policy/value network sizes.

6:15-7:00

Open `src/agents/random_agent.py` and `src/agents/center_agent.py`.

I also implemented random and center-drop baselines. These make the evaluation more informative because DQN is compared against both a naive policy and a simple hand-coded heuristic.

7:00-8:30

Open `src/scripts/eval_agent.py`, `src/scripts/compare_agents.py`, and the README Evaluation section.

Evaluation uses 50 complete episodes per agent. I report episode return, final score, episode length, final fruit count, and physics frames. DQN had the best mean return and longest survival. PPO improved over random but did not beat the center heuristic.

8:30-9:30

The main design choices were structured dict observations instead of pixels for faster learning, merge score as the reward because it directly matches the objective, a DeepSets network because fruits are unordered, and a comparison between custom DQN, library PPO, and simple baselines under the same environment.

Closing line:

The significant technical contributions are the custom Gymnasium environment, the custom Double DQN implementation, the DeepSets Q-network, and the complete evaluation pipeline comparing learned and heuristic policies.
