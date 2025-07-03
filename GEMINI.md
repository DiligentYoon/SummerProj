# Project Context: HRL for Franka Pick-and-Place

## 1. Project Overview

This project aims to train a Franka Emika Panda robot for a **Pick-and-Place task** within the **Isaac Sim** simulator. The core of the project is a **Hierarchical Reinforcement Learning (HRL)** framework built upon the **skrl** library, designed to decompose the complex task into manageable sub-tasks.

- **Primary Goal**: Successfully pick up an object and move it to a target location.
- **Core Frameworks**: Isaac Sim, PyTorch, skrl
- **Key Architectural Pattern**: Hierarchical Reinforcement Learning (HRL) using a custom implementation of DIOL (Diversity is All You Need) for the high-level policy and DDPG with HER for the low-level policy.

---

## 2. Core HRL Architecture

The HRL framework is composed of three main components that work in concert:

### 2.1. High-Level Policy (The "What")

- **Agent**: `DIOLAgent` (`/source/SummerProj/SummerProj/tasks/direct/franka_pap/agents/diol_agent.py`)
- **Model**: `FrankaQNetwork` (`/source/.../models/custom_diol_net.py`)
- **Algorithm**: DQN-based. It learns a Q-function to estimate the value of taking a specific abstract action in a given state.
- **Role**: Acts as the **master controller**. It observes the environment's state and decides **"what"** to do next by selecting a single, discrete, abstract action from a predefined set.
- **Action Space**: A discrete set of integers, where each integer corresponds to an abstract task phase (e.g., `0` for `APPROACH_OBJECT`). The agent outputs the **single integer index** of the action with the highest Q-value (`argmax`).

### 2.2. The Bridge: Environment-based Goal Mapping

- **Location**: `_map_high_level_action_to_low_level_goal()` method within `franka_pap_env.py`.
- **Dictionary**: `FrankaPapAction` Enum in `/source/.../task_tables/pap_task_table.py`.
- **Role**: This is the crucial link between the high-level and low-level policies. It translates the **abstract "what"** from the high-level policy into a **concrete "how"** for the low-level policy.
- **Process**: It takes the integer action from the `DIOLAgent` and, based on the current state of the environment (e.g., object position, final goal), generates a specific, continuous **sub-goal** (e.g., a target 6D pose for the robot's end-effector and a gripper state).

### 2.3. Low-Level Policy (The "Doer")

- **Agent**: `DDPG` (a standard agent from the `skrl` library).
- **Models**:
    - `FrankaDeterministicPolicy` (Actor)
    - `FrankaValue` (Critic)
    - Both are located in `/source/.../models/custom_diol_net.py`.
- **Role**: Acts as the **skill executor**. Its sole job is to achieve the concrete sub-goal provided by the environment's mapping function.
- **Action Space**: A continuous space representing the robot's delta end-effector pose and impedance controller gains.
- **Key Technique**: Uses **Hindsight Experience Replay (HER)**, managed by the `HRLTrainer`, to learn efficiently from both successful and failed attempts at reaching a sub-goal. This is critical for sample efficiency in sparse reward settings.

---

## 3. Execution and Data Flow

The training process follows a clear, orchestrated flow:

1.  **Initiation**: The user runs `python scripts/skrl/train_custom.py`.
2.  **Assembly**: The `AISLDIOLRunner` (`runner_diol.py`) is instantiated. It reads the central configuration file (`skrl_custom_diol_cfg.yaml`).
3.  **Instantiation**: Based on the config, the runner instantiates all necessary components:
    -   The `FrankaPapEnv` environment.
    -   The `HRLTrainer`.
    -   The `DIOLAgent` (high-level) and `DDPG` agent (low-level).
    -   All associated neural network models (`FrankaQNetwork`, `FrankaDeterministicPolicy`, etc.).
4.  **Training Loop Start**: `runner.run()` calls the `HRLTrainer.train()` method.
5.  **Inner HRL Loop (per step)**:
    a. The `HRLTrainer` gets the current state from the environment.
    b. It calls the **high-level `DIOLAgent`**'s `act()` method, which returns a single integer action (e.g., `2` for `GRASP_OBJECT`).
    c. This integer is passed to the **environment's** `_map_high_level_action_to_low_level_goal()` method.
    d. The environment returns a concrete, continuous sub-goal (e.g., target TCP pose `[x,y,z,qx,qy,qz,qw]` and gripper state).
    e. This sub-goal is fed into the **low-level `DDPG` agent**.
    f. The `DDPG` agent outputs a continuous action (delta pose, gains) to be executed by the robot controllers.
    g. The environment is stepped, and the results (next_state, rewards) are collected.
6.  **Experience Replay & Learning**:
    -   Transitions are stored in an episode buffer.
    -   When an episode ends, the `HRLTrainer`'s `_apply_her_and_record_transition` method processes the buffer, generates augmented experiences via HER, and populates the low-level agent's replay memory.
    -   The high-level agent's transitions are stored directly.
    -   Both agents are periodically trained on their respective replay buffers.

---

## 4. Key Files & Directories Reference

-   `scripts/skrl/train_custom.py`: **Main entry point** for starting HRL training.
-   `scripts/skrl/runner_diol.py`: **Orchestrator** that builds all HRL components.
-   `source/SummerProj/tasks/direct/franka_pap/agents/skrl_custom_diol_cfg.yaml`: **Central configuration file**. Defines all hyperparameters, models, and agent classes.
-   `source/SummerProj/tasks/direct/franka_pap/task_tables/pap_task_table.py`: Defines the **abstract action sequence** (`FrankaPapAction` Enum).
-   `source/SummerProj/tasks/direct/franka_pap/franka_pap_env.py`: The main Gym environment, containing the crucial **goal mapping logic**.
-   `source/SummerProj/tasks/direct/franka_pap/trainers/trainer.py`: The `HRLTrainer` which contains the main **HRL training loop** and **HER implementation**.
-   `source/SummerProj/tasks/direct/franka_pap/agents/diol_agent.py`: The custom **high-level agent**.
-   `source/SummerProj/tasks/direct/franka_pap/models/custom_diol_net.py`: The custom **neural network architectures** for both high-level and low-level agents.

---

## 5. Session Summary & Modifications

This section tracks changes made during our interactive sessions.

-   **Created `task_tables/pap_task_table.py`**: Established a clear, Enum-based definition for high-level actions to improve code clarity and modularity.
-   **Updated `franka_pap_env_cfg.py`**:
    -   Dynamically set `high_level_goal_dim` based on the `FrankaPapAction` Enum size.
    -   Added configuration constants for goal mapping (e.g., `approach_offset_z`).
-   **Implemented Goal Mapping Logic**: Filled in the `_map_high_level_action_to_low_level_goal` function in `franka_pap_env.py` with the complete logic for all 8 phases of the Pick-and-Place task.
-   **Bug Fixes**:
    -   Corrected the return signature of `DIOLAgent.act()` to conform to the `skrl` standard (returning a tuple `(actions, None, {})` instead of just a tensor).
    -   Adjusted the call site in `HRLTrainer.train()` to correctly unpack the tuple returned by `DIOLAgent.act()`.
