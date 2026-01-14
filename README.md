# Two-Link Robot Navigation with PPO

A reinforcement learning project that trains a two-link robot to navigate through a randomly generated terrain to reach a goal door using Proximal Policy Optimization (PPO).

## Overview

This project implements a physics-based simulation environment where a two-link robot must learn to navigate across uneven terrain to reach a target door. The robot is trained using PPO (Proximal Policy Optimization), a state-of-the-art policy gradient method for reinforcement learning.

## Features

- **Physics Simulation**: Uses PyMunk for realistic 2D physics simulation
- **PPO Training**: Implements PPO with GAE (Generalized Advantage Estimation)
- **Visualization**: Real-time rendering with Pygame and episode replay functionality
- **Episode Recording**: Saves training episodes for later analysis and video rendering
- **TensorBoard Integration**: Comprehensive logging of training metrics
- **Adaptive Learning Rate**: Optional automatic learning rate reduction based on success rate

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 1dof-copy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: For CUDA support with PyTorch, install from [pytorch.org](https://pytorch.org/get-started/locally/) instead of using the requirements.txt version.

## Project Structure

```
.
├── environment.py    # Physics environment with terrain generation
├── robot.py          # Two-link robot implementation
├── brains.py         # PPO algorithm and neural network
├── drawing.py        # Visualization and episode replay
├── train.py          # Training script
└── requirements.txt  # Python dependencies
```

## Usage

### Training

Run the training script:

```bash
python train.py
```

To resume training from a saved policy:

```python
# In train.py, modify:
policy = brains.load_policy("runs/policy.pth")
```

### Training Parameters

Key parameters in `train.py`:

- `episodes`: Total number of training episodes
- `lr`: Learning rate (default: 5e-5)
- `target_steps`: Timesteps to collect before each PPO update (default: 4096)
- `minibatch_size`: Size of minibatches for PPO updates (default: 256)
- `ppo_epochs`: Number of epochs per update (default: 3)
- `episode_save_interval`: Save episodes every N episodes (default: 20)
- `plot_interval`: Generate plots every N episodes (default: 200)
- `adaptive_lr`: Enable automatic learning rate reduction (default: False)

### Environment Configuration

The environment can be customized in `train.py`:

```python
env = environment.WalkerEnv(
    width=800,              # Environment width
    height=600,             # Environment height
    time_limit=20,          # Episode time limit (seconds)
    substeps=5,             # Physics substeps per frame
    max_torque=120000.0,    # Maximum joint torque
    seed=0,                 # Random seed
    number_of_ground_points=20,  # Terrain complexity
    fps=30                  # Simulation framerate
)
```

### Episode Replay

Replay a saved episode:

```python
from drawing import replay_episode

replay_episode("runs/episodes/ep_000020.npz", speed=1.0)
```

Controls:
- **SPACE**: Pause/Unpause
- **ESC**: Exit
- **R**: Replay (after completion)

### Video Rendering

Render episodes to video files:

```python
from drawing import render_episode_to_video

render_episode_to_video(
    "runs/episodes/ep_000020.npz",
    "output.mp4",
    fps=30,
    speed=1.0,
    crop_16_9=True  # Optional: crop to 16:9 aspect ratio
)
```

Batch render all episodes:

```python
# In drawing.py __main__ section
episodes_dir = "runs/episodes"
videos_dir = "runs/videos"
# ... (see drawing.py for full example)
```

## Training Output

Training creates the following directory structure:

```
runs/
├── tb/              # TensorBoard logs
├── episodes/        # Saved episode data (.npz files)
├── plots/           # Training progress plots
└── policy.pth       # Saved policy checkpoint
```

### TensorBoard

View training metrics:

```bash
tensorboard --logdir runs/tb
```

Metrics logged include:
- Policy loss, value loss, entropy loss
- Approximate KL divergence
- Clip fraction
- Mean return and value estimates
- Explained variance
- Reward components (progress, time penalty, angular penalty, goal bonus)
- Success rate (100-episode moving average)
- Learning rate

## Algorithm Details

### PPO (Proximal Policy Optimization)

- **Policy**: Squashed Gaussian policy (tanh activation)
- **Value Function**: Separate critic network
- **Advantage Estimation**: GAE (Generalized Advantage Estimation)
- **Clipping**: ε = 0.2 (default)
- **Discount Factor**: γ = 0.99
- **GAE Lambda**: λ = 0.95

### Observation Space

The observation includes:
- Relative position to door
- Ground point coordinates
- Robot tip vertices
- Linear velocities (normalized)
- Angular velocities (normalized)

### Action Space

- Continuous action in [-1, 1] representing joint torque

### Reward Function

- **Progress Reward**: Positive reward for moving closer to door
- **Time Penalty**: Small penalty per timestep
- **Angular Velocity Penalty**: Penalty for excessive rotation
- **Goal Bonus**: Large bonus (100.0) for reaching the door

## Requirements

- Python 3.7+
- PyMunk (physics simulation)
- Pygame (visualization)
- NumPy
- PyTorch
- Matplotlib
- TensorBoard

See `requirements.txt` for specific versions.

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]
