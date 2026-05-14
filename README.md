# Unitree Go2: Blind Locomotion on Challenging Terrains

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Isaac_Lab-RL-green?style=for-the-badge&logo=nvidia&logoColor=white" alt="Isaac Lab">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Robotics-Quadruped-orange?style=for-the-badge&logo=robot&logoColor=white" alt="Robotics">
</p>

This project demonstrates a robust RL-based locomotion policy for the Unitree Go2 quadruped, trained using Isaac Lab. The policy enables the robot to traverse diverse, non-flat terrains using only proprioceptive sensing.

## 🚀 Performance Overview

| Rough Terrain (Noise 0.25) | Stairs (Up & Down) | Sloped Terrain |
|:---:|:---:|:---:|
| ![Rough Terrain GIF](PLACEHOLDER_ROUGH_TERRAIN_GIF_URL) | ![Stairs GIF](PLACEHOLDER_STAIRS_GIF_URL) | ![Slopes GIF](PLACEHOLDER_SLOPES_GIF_URL) |

*The policy utilizes an asymmetric Actor-Critic architecture to handle partial observability during deployment.*

---

## 🛠️ Training Setup

### 1. Asymmetric Actor-Critic
To bridge the gap between simulation and reality, I used an **Asymmetric Actor-Critic** setup:
- **Critic:** Has access to "privileged information" during training (e.g., ground truth friction, height scans around feet, and contact forces).
- **Actor:** Only observes proprioceptive data (joint positions/velocities, IMU, and previous actions). This ensures the policy can be deployed on real hardware without needing external depth sensors for basic gait.

### 2. Terrain Curriculum
The robot was trained on a mixture of terrains to ensure generalization:
- **Noisy Rough Terrain:** Stochastic height variations up to 0.5m to build balance robustness.
- **Stairs:** Designed to force the robot to utilize its full range of motion (ROM).
- **Slopes:** Challenging the orientation estimation and torque limits.

---

## 🧠 Technical Insights & Troubleshooting

### The "RandomGridMesh" Challenge
During development, I observed that training on `RandomGridMesh` often led to policy crashes and numerical instability (negative standard deviation). 

**Analysis of the issue:**
1. **Overconfidence & Stumbling:** In high-frequency grid noise, the robot often catches a "toe" on a sharp vertex. If the reward function doesn't heavily penalize high-impact foot contacts, the agent learns to move fast but "trips" on geometry it cannot see (since it's blind).
2. **Entropy Collapse:** The "negative standard deviation" usually indicates the distribution is collapsing because the agent is receiving massive negative rewards (crashes) and the optimizer is struggling to find a gradient, or the KL divergence is spiking.

**Potential Cues/Fixes:**
- **Curriculum Scaling:** Start the `RandomGridMesh` amplitude at 0.0 and slowly scale it up only after the robot masters flat ground.
- **Action Smoothing:** Add a penalty for `action_rate` and `torques` to prevent the "jittery" behavior that causes stumbles on fine meshes.
- **Contact Buffers:** Ensure your collision bitmasks for the feet are precise; sometimes "snagging" on mesh edges is a simulation artifact that can be mitigated by using small spheres for feet collisions instead of complex meshes.

---

## 📹 Full Demonstration

![Collage of all terrains](PLACEHOLDER_MAIN_COLLAGE_GIF_URL)

*Full range of motion being exercised on 15cm stairs.*

---

## 💻 How to Run

If you have Isaac Lab installed, you can play the trained policy using:

```bash
python scripts/reinforcement_learning/play.py --task=Template-Isaac-Velocity-Go2-v0 --num_envs 16
```

## 🔗 Connect with me
- **LinkedIn:** [Your Profile Link]
- **Portfolio:** [Your Website Link]
- **GitHub:** [Your GitHub Link]

---
*Keywords: Reinforcement Learning, Isaac Lab, Robotics, Unitree Go2, Quadruped Locomotion, NVIDIA Isaac Sim.*