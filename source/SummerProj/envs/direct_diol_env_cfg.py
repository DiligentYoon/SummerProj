from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from isaaclab.utils.configclass import configclass

@configclass
class DirectDIOLCfg(DirectRLEnvCfg):
    """Configuration for a DirectDIOL environment.

    This class extends the DirectRLEnvCfg to include additional configurations specific to DirectDIOL.
    """

    # Low-Level Goal Dimension (Not Task-Specific but, High-Level is not)
    # Gripper Pose (Location & Rotation) : (7)
    # Gripper Joint State : (2)
    low_level_goal_dim = 9
    achieved_goal_dim = 9
    
    # Reward shaping settings
    low_level_loc_threshold: float = 0.02
    low_level_rot_threshold: float = 0.1

    high_level_loc_threshold: float = 0.02
    high_level_loc_rot_threshold: float = 0.1
    