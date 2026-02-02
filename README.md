# MUST: Smartwatch-based Multimodal Framework for Predicting Driver State and Takeover Performance (CHI' 26)
## Authors: 
Seokyong Sheem, Yujin Cho, In Kyung Lee, Hanjun Cho, Taegeun Kim, Byung Hyung Kim*, Daekyum Kim*

## Abstract: 
Ensuring timely takeover in conditionally autonomous vehicles presents a significant challenge, especially when drivers are distracted by non-driving-related tasks or are in suboptimal emotional states. Existing driver monitoring systems struggle with a trade-off between practicality and reliability. Physiological sensors are intrusive, vision-based methods are sensitive to occlusions and variable lighting, and current multimodal learning approaches often rely on simple fusion strategies that fail to reconcile heterogeneous data. We introduce MUST (Multimodal Unified Smartwatch-based Takeover), a framework that predicts driver state and takeover performance using unobtrusive smartwatch signals. MUST employs an asymmetric causal fusion mechanism to model the interplay between driver behavior and emotion. The performance of the architecture was validated in diverse simulator environments reflecting real-world driving conditions, demonstrating robust driver state estimation and takeover prediction. This work establishes the smartwatch as a practical tool for adaptive takeover support, enabling reliable readiness assessment without intrusive hardware or fragile vision systems.

# Project Structure
- `main.py`: Entry point for training and evaluation.
- `config/`: Configuration files for hyperparameters and environment settings.
- `models/`: Architecture definitions for deep learning models.
- `trainers/`: Modules for training loops, validation, and optimization logic.
- `data/`: Data pipeline (includes a Dummy Dataset for reproducibility).

# Getting Started
1. Environment: Ensure you have Python 3.x and required libraries installed.
2. Data: For security and storage reasons, the full dataset is excluded. However, a dummy `train.pkl` is provided in the `data/` folder to verify the pipeline execution.

# Usage
To run the project with the dummy dataset:
```bash
python main.py --config config/default.yaml
```

---
*Developed by sheemsy @ Mint Lab*
