from enum import IntEnum

class Demonstrator(object):
    def __init__(self):
        pass




class FrankaPapAction(IntEnum):
    """
        High-level actions for the Franka Pick-and-Place task.
        Each member represents a distinct phase of the task.
    """
    # === Pick Phase ===
    APPROACH_OBJECT = 0  # 1. Move TCP to a pre-grasp position above the object
    DESCEND_TO_OBJECT = 1  # 2. Lower the TCP to the object's height
    GRASP_OBJECT = 2       # 3. Close the gripper
    LIFT_OBJECT = 3        # 4. Lift the object vertically

    # === Place Phase ===
    MOVE_TO_GOAL = 4       # 5. Move the object towards the final goal position
    DESCEND_TO_GOAL = 5    # 6. Lower the object to the goal height
    RELEASE_OBJECT = 6     # 7. Open the gripper
    RETREAT_FROM_GOAL = 7  # 8. Move the TCP up to a safe retreat position

    