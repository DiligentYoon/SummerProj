# Multi-Stage Reinforcement Learning for Robotic Manipulation

This repository contains a project for training a Franka Emika Panda robot to perform a **Pick-and-Place** task within the **NVIDIA Isaac Sim** environment. The project leverages a sophisticated **Phase Based Sequence Learning** framework, decomposing the complex manipulation task into a series of simpler, manageable sub-goals.

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

### Running the Multi-Stage Sequential Training

To start the training process for the Franka Pick-and-Place task, run the custom training script:

```bash
# Use the full path to isaaclab.sh/bat if it's not in your PATH
python scripts/skrl/train.py --task Franka-Graps-Direct-v0
```

-   This script will automatically instantiate the `AISLRunner`, which assembles the entire framework.

### Evaluating a Trained Agent

To watch a trained agent perform, use the `play.py` script, pointing it to the desired checkpoint file.

```bash
python scripts/skrl/play.py --task Franka-Grasp-Direct-v0 --checkpoint /path/to/your/checkpoint.pt
```

## Code Formatting

This project uses `pre-commit` for automated code formatting.

```bash
# Install pre-commit
pip install pre-commit
# Run on all files
pre-commit run --all-files
```

## Demo GIF
<p align="center">
<img src="post/Multi_Stage_RL.gif" width="720"/>
</p>

## Poster
<p align="center">
<img src="post/Poster.png" alt="Poster" width="720">
</p>
