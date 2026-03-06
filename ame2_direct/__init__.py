"""AME-2 Direct Workflow Environment (Isaac Lab DirectRLEnv).

Bypasses Isaac Lab's manager infrastructure for 3-5x speedup vs manager-based env.
Equivalent to legged_gym coding style, but runs on modern hardware (RTX 5090 / Blackwell).

Reference: AME-2 paper used Isaac Gym [61] (arXiv:2601.08485, Sec.IV-E.1).
We port to Isaac Lab Direct Workflow — same physics, same speed class, no old GPU constraints.
"""

from .config import AME2DirectEnvCfg
from .env import AME2DirectEnv, AME2DirectWrapper

__all__ = ["AME2DirectEnv", "AME2DirectEnvCfg", "AME2DirectWrapper"]
