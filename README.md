# Hierarchical Reinforcement Learning for Robotic Manipulation

This repository contains a project for training a Franka Emika Panda robot to perform a **Pick-and-Place** task within the **NVIDIA Isaac Sim** environment. The project leverages a sophisticated **Hierarchical Reinforcement Learning (HRL)** framework, decomposing the complex manipulation task into a series of simpler, manageable sub-goals.

## Core Architecture: A Two-Level HRL Approach

The learning process is driven by a two-tiered hierarchical structure, separating high-level strategic decisions from low-level motor control.

```
+-----------------------------+
|      HRL Trainer            |
| (Orchestrator: trainer.py)  |
+-----------------------------+
            |
            v
+-----------------------------+      (1. "What to do?")      +--------------------------------+
| High-Level Policy (DIOL)    |---------------------------->| Environment (Task-Specific Env) |
| (The "Mastermind")          |      (Integer Action)       | (The "Translator")              |
+-----------------------------+                             +--------------------------------+
            ^                                                              |
            | (4. State Update)                                            | (2. "How to do it?")
            |                                                              | (Sub-Goal Pose)
+-----------------------------+                             +--------------------------------+
| Low-Level Policy (DDPG+HER) |<----------------------------|                                |
| (The "Skill Executor")      |      (3. Continuous Action) |                                |
+-----------------------------+---------------------------->+--------------------------------+
```

1.  **High-Level Policy (The "Mastermind")**:
    *   **Agent**: `DIOLAgent`, a custom DQN-style agent.
    *   **Responsibility**: Observes the state of the world and makes strategic, high-level decisions (e.g., "approach the object," "grasp it"). It outputs a single integer representing the chosen abstract action.

2.  **Environment as a Translator**:
    *   **Component**: `_map_high_level_action_to_low_level_goal` method in `franka_pap_env.py`.
    *   **Responsibility**: Translates the abstract integer action from the high-level policy into a concrete, physical sub-goal (e.g., a specific 6D pose for the robot's end-effector).

3.  **Low-Level Policy (The "Skill Executor")**:
    *   **Agent**: Standard `DDPG` agent from the `skrl` library.
    *   **Responsibility**: Receives the concrete sub-goal and generates the continuous, low-level motor commands (delta poses and impedance control gains) required to achieve it.
    *   **Key Feature**: Employs **Hindsight Experience Replay (HER)** to learn efficiently from both successful and failed attempts, dramatically improving sample efficiency.

## Key Features

-   **Hierarchical Control**: Decomposes a long-horizon task into simpler skills, making learning more tractable.
-   **Custom HRL Framework**: Built on top of `skrl`, featuring a custom `HRLTrainer`, `DIOLAgent`, and specialized environment wrappers.
-   **Hindsight Experience Replay (HER)**: Enables the low-level policy to learn from failures by retroactively treating achieved states as intended goals.
-   **Task-Specific Action Abstraction**: High-level actions are clearly defined in an Enum (`FrankaPapAction`), making the code readable and easily extensible.
-   **Impedance Control**: Utilizes joint impedance control for safer and more compliant robot interaction with the environment.

## Getting Started

### Prerequisites

-   **NVIDIA Isaac Sim**: Ensure Isaac Sim is installed correctly.
-   **Isaac Lab**: This project is built upon Isaac Lab. Follow the official [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### Installation

1.  Clone this repository to a location of your choice.
2.  Install the project package in editable mode. This allows you to modify the source code without reinstalling.

    ```bash
    # From the repository's root directory
    # Ensure your python environment has Isaac Lab's dependencies
    pip install -e source/SummerProj
    ```

### Running the HRL Training

To start the training process for the Franka Pick-and-Place task, run the custom training script:

```bash
# Use the full path to isaaclab.sh/bat if it's not in your PATH
python scripts/skrl/train_custom.py --task Franka-PickandPlace-Direct-v0
```

-   `--task Franka-PickandPlace-Direct-v0`: Specifies the custom HRL environment.
-   This script will automatically instantiate the `AISLDIOLRunner`, which assembles the entire HRL framework (agents, models, trainer) based on the configurations in `skrl_custom_diol_cfg.yaml`.

### Evaluating a Trained Agent

To watch a trained agent perform, use the `play_custom.py` script, pointing it to the desired checkpoint file.

```bash
python scripts/skrl/play_custom.py --task Franka-PickandPlace-Direct-v0 --checkpoint /path/to/your/checkpoint.pt
```

## Code Structure

-   `GEMINI.md`: A detailed document outlining the project architecture, data flow, and key files for AI-assisted development.
-   `scripts/skrl/`: Contains the primary scripts for **training (`train_custom.py`)** and **evaluation (`play_custom.py`)**.
    -   `runner_diol.py`: The `AISLDIOLRunner` class, responsible for orchestrating the setup of the HRL framework.
-   `source/SummerProj/`: The core Python package for the project.
    -   `tasks/direct/franka_pap/`: The heart of the project, defining the Pick-and-Place task.
        -   `franka_pap_env.py`: The main Gym environment, including the crucial goal-mapping logic.
        -   `franka_pap_env_cfg.py`: Configuration for the environment.
        -   `task_tables/pap_task_table.py`: Defines the high-level action abstractions (`FrankaPapAction`).
        -   `agents/`: Contains the custom `DIOLAgent` and the central `skrl_custom_diol_cfg.yaml` configuration file.
        -   `models/`: Defines the custom PyTorch neural network architectures for both policies.
        -   `trainers/`: Includes the `HRLTrainer` that manages the entire hierarchical learning loop and HER.

## Code Formatting

This project uses `pre-commit` for automated code formatting.

```bash
# Install pre-commit
pip install pre-commit
# Run on all files
pre-commit run --all-files
```
