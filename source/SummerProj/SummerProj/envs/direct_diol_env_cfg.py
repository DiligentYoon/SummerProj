
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from isaaclab.envs.common import SpaceType

@configclass
class DirectDIOLCfg(DirectRLEnvCfg):
    """Configuration for a DirectDIOL environment.

    This class extends the DirectRLEnvCfg to include additional configurations specific to DirectDIOL.
    """

    high_level_observation_space: SpaceType = MISSING
    high_level_action_space: SpaceType = MISSING

    low_level_observation_space: SpaceType = MISSING
    low_level_action_space: SpaceType = MISSING
    