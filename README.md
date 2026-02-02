# MUST: [Full Project Name or Short Description]

This repository contains the core implementation of the MUST project at Mint Lab.

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
Further if you have any trouble, please feel free to mail sheemsy@korea.ac.kr
