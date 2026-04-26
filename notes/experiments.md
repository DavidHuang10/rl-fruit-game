
## DQN v1 — 2026-04-25 (smoke tests, discarded)

Two short local smoke tests (2k steps each) on MPS to verify the training loop ran end-to-end. Results not meaningful — both runs were in warmup the entire time.

## DQN v1 — 2026-04-25 18:30 (production run, cluster CUDA)

**Config**
- total_steps: 500,000
- warmup: 5,000
- buffer_cap: 100,000
- batch_size: 128
- grad_freq: every 4 env steps
- n_envs: 4
- device: cuda
- network: SuikaQNetwork (per-fruit MLP 13→128→128, mean+max pool, head 266→256→256→32)
- algorithm: Double DQN, γ=0.99, Adam lr=3e-4 cosine→1e-5, grad_clip=10, target_sync=1000 grad steps
- reward: raw (no scaling), Huber loss
- epsilon: 1.0→0.05 over 100k env steps

**Results (50-episode eval, results/dqn/)**
- Mean episode return: 4085.6 ± 786.9
- Mean episode length: 304.3 ± 47.4
- Mean final fruits: 42.1 ± 4.2
- Learning curve: returns rose steadily from ~2200 at 5k steps to ~3900 by 400k steps, then plateaued. Final 50k steps oscillated between 3846–3965 as cosine LR decayed to 1e-5.

**What changed vs. prior runs**
First full production run. Baseline for all future DQN iterations.

**Observations driving next iteration**
The learning curve shows a plateau in the final 100k steps as the cosine LR schedule bottomed out at 1e-5. Returns were still slowly improving at step 400k (3681→3882), suggesting the initial lr=3e-4 was too conservative to fully exploit the available training budget. Additionally, the high eval variance (±787) indicates occasional episode collapses from the trapped-fruit failure mode. Next run will test lr=3e-3 (10x) to see if faster early optimization reaches a stronger policy within a shorter 300k-step budget.
