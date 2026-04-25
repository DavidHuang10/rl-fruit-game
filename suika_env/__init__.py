from gymnasium.envs.registration import register

from .env import SuikaEnv
from .config import EnvConfig

register(
    id="suika_env/SuikaEnv-v0",
    entry_point="suika_env.env:SuikaEnv",
)
