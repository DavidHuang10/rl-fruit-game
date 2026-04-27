# Attribution

## AI Tools

I used a blend of Codex and Claude Code throughout this project. AI tools contributed heavily to the implementation, tests, scripts, documentation polish, and repository cleanup. I used them as coding assistants: I described the behavior I wanted, reviewed the generated code, tested it, changed direction when the results did not match the project goals, and made the final decisions about what to keep.

I am responsible for the final submitted code, experiments, documentation, and results.

## What AI Helped Generate

- `src/suika_env/`: AI helped implement the custom Gymnasium environment, Pymunk physics wrapper, rendering code, observation construction, reward plumbing, and configuration structure.
- `src/agents/`: AI helped implement the DQN training logic, DeepSets-style Q-network, replay buffer, PPO loading wrapper, and random/center baseline agents.
- `src/scripts/` and top-level `scripts/`: AI helped draft training scripts, evaluation scripts, comparison plotting, watch/play modes, compatibility wrappers, and Slurm scripts.
- `tests/`: AI helped draft unit tests and Gymnasium API tests. I used these tests to check environment behavior, replay buffer sampling, network output shapes, and agent loading behavior.
- Documentation files, including `README.md`, `SETUP.md`, `CHECKLIST.md`, `TODO.md`, and `scripts.md`, were written and polished with AI assistance.

## What I Directed or Modified

I chose the overall project direction: a Suika-style game as a custom reinforcement-learning environment, with DQN as the main custom learned agent and PPO as a library comparison method.

I directed the main modeling choices. For DQN, I chose a structured state representation instead of pixels and decided to use a DeepSets-style architecture because the active fruits are naturally an unordered set. This meant a MLP per fruit with mean and max pooling. I also directed important replay buffer choices, including storing fruit type information compactly and expanding it during sampling. For PPO, I researched and implemented the Stable-Baselines3 setup with `MultiInputPolicy` so the library baseline could consume the same dictionary observation space as DQN.

I designed the action and reward setup: 32 discrete drop columns, current/next fruit observations, and merge-score reward because it directly matches the game score objective. I also play-tested the environment and adjusted the physics feel, including damping, friction, elasticity, and game-over behavior.

I chose the evaluation structure: random baseline, center-drop heuristic, PPO, and DQN, each evaluated over 50 complete episodes using return, final score, episode length, final fruit count, and physics-frame counts.

I debugged the physics environment when it seemed too bouncy, adjusted friction and damping parameters, and added code to clear stuck fruits, after playing the game on my own to replicate how it is on other platforms.

I also chose the hyperparameters for the DQN and PPO agents, including learning rate, batch size, and discount factor, and observed preliminary training sessions to inform my choices for the number of training iterations, learning rate, and architecture tweaks. For example, I decided to include max pooling in addition to mean pooling after observing a lack of convergence in early training.

## Debugging, Rework, and Review

Several parts of the project were modified after initial AI drafts. I reorganized the code so agent logic lived under `src/agents/` instead of being scattered through scripts. I reviewed and adjusted setup documentation so it matched the actual workflow using `uv sync`, `uv run`, and `.venv`. I also reviewed generated training and evaluation scripts against the artifacts they produced, including checkpoints, metrics CSVs, learning curves, and summary JSON files.

During testing and play-testing, I used agent behavior to diagnose issues and interpret results. PPO improved over random but did not beat the center heuristic, so I described that honestly in the evaluation instead of presenting PPO as a success. DQN's main failure mode was also identified by watching gameplay: aggressive merging can trap small fruits near the bottom and reduce usable board space later in the episode.

## My Contributions

- Project concept and game choice
- Physics feel, including iterative play-testing to tune damping, friction, elasticity, and game-over behavior
- Observation and action space design
- Reward function design
- Choice to use a custom DQN as the main learned agent and Stable-Baselines3 PPO as a comparison method
- DQN architecture direction, including the DeepSets-style board representation for unordered fruit states
- Replay buffer design direction, including compact storage of fruit type information and sampling behavior
- PPO research and configuration choices, including using `MultiInputPolicy` with the dictionary observation space
- Experiment design, training runs, baseline comparisons, and evaluation metrics
- Diagnosis of behavioral bugs and policy failure modes from play-testing and watching agents
- Evaluation and interpretation of all results
- Final review of README, setup instructions, attribution, and rubric evidence

## External Libraries and Resources

This project uses external open-source libraries for implementation:

- Gymnasium for the reinforcement-learning environment API
- PyTorch for the DQN neural network and optimization
- Stable-Baselines3 for PPO
- Pymunk for 2D physics
- Pygame for rendering and human/watch modes
- NumPy, Pandas, and Matplotlib for numerical work, metrics, and plots

No external dataset was used. Training and evaluation episodes were generated by the custom Suika environment. The trained model files and result artifacts in `models/` and `results/` were produced by local and cluster training/evaluation runs for this project.
