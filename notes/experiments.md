
## DQN run — 2026-04-25 15:40

**Config**
- total_steps: 2,000
- warmup: 500
- buffer_cap: 100,000
- batch_size: 128
- grad_freq: every 4 env steps
- device: mps
- network: SuikaQNetwork (per-fruit MLP 13→128→128, mean+max pool, head 266→256→256→32)
- algorithm: Double DQN, γ=0.99, Adam lr=3e-4 cosine→1e-5, grad_clip=10, target_sync=1000 grad steps
- reward: raw (no scaling), Huber loss
- epsilon: 1.0→0.05 over 100k env steps

**Results**
*(fill in after run: final mean return, game score, steps/ep, convergence step)*

**What changed vs. prior runs**
*(fill in)*

## DQN run — 2026-04-25 15:42

**Config**
- total_steps: 2,000
- warmup: 500
- buffer_cap: 100,000
- batch_size: 128
- grad_freq: every 4 env steps
- device: mps
- network: SuikaQNetwork (per-fruit MLP 13→128→128, mean+max pool, head 266→256→256→32)
- algorithm: Double DQN, γ=0.99, Adam lr=3e-4 cosine→1e-5, grad_clip=10, target_sync=1000 grad steps
- reward: raw (no scaling), Huber loss
- epsilon: 1.0→0.05 over 100k env steps

**Results**
*(fill in after run: final mean return, game score, steps/ep, convergence step)*

**What changed vs. prior runs**
*(fill in)*

## DQN run — 2026-04-25 18:30

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

**Results**
*(fill in after run: final mean return, game score, steps/ep, convergence step)*

**What changed vs. prior runs**
*(fill in)*
