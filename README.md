# MUST

Multimodal driving-state modeling code with a public synthetic sample dataset.

The original training data and model weights are not included because they are
private. The repository ships with `data/sample/sample_train.pkl` and
`data/sample/sample_survey.csv`, which are synthetic files that only match the
loader schema.

## Quick Start

```bash
pip install -r requirements.txt
python main.py --profile sample --smoke
```

When the private dataset exists at `data/data/train/train_ver2.pkl`, `python
main.py` uses the full profile automatically. In a public clone without that
file, the code falls back to the sample profile and runs the smoke check.

Runtime paths are resolved from the repository root, so the command works even
when invoked from another current working directory.

## Private Data Overrides

Use environment variables if local private files live outside the repository:

```bash
MUST_PROFILE=full \
MUST_PKL_ALL=/path/to/train_ver2.pkl \
MUST_SURVEY_CSV=/path/to/survey.csv \
python main.py
```

Optional pretrained checkpoint overrides:

```bash
MUST_EMOTION_CKPT=/path/to/best_emotion.pt \
MUST_MOTION_CKPT=/path/to/best_motion.pt \
python main.py --profile full
```

## Regenerate Public Sample Data

```bash
python scripts/create_sample_data.py
```
