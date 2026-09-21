# MUST

Multimodal driving-state modeling code with a public synthetic sample dataset.

The original training data and model weights are not included because they are
private. The repository ships with `data/sample/sample_train.pkl` and
`data/sample/sample_survey.csv`, which are synthetic files that only match the
loader schema.

## Shared source snapshot (2026-09-20)

The executable source matches the shared `MUST_source_20260920.zip` package.
See [delivery notes (Korean)](DELIVERY_NOTES_KO.md).

Known limitations of this snapshot:

- The 31-feature survey CSV is synthetic. The actual 31-feature survey schema
  and preprocessing remain unverified. Full mode also defaults to the synthetic
  CSV unless `MUST_SURVEY_CSV` is set; missing participants receive zero vectors.
- With the default modalities, the active fusion v28_assym path does not consume
  survey values. The standalone emotion model does use survey as context.
- The TOT/ACT enhancer context issue remains unresolved in this shared version.
  This snapshot supersedes the implementation from commit `829f698` to match
  the delivered package; that commit remains in history. Its enhancer fixes
  have not been ported into this implementation.
- The sample smoke check validates data loading and batch shapes only. It does
  not validate full training, model inference, or reported performance.

The August change record describes the historical local working tree. Its
references to a pending pull, local logs, and archive files are historical;
those files are not included here.

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

Full training requires compatible pretrained emotion and motion checkpoints,
which are not included. The default entry point loads these checkpoints rather
than running their pretraining stages. Override their paths if necessary:

```bash
MUST_EMOTION_CKPT=/path/to/best_emotion.pt \
MUST_MOTION_CKPT=/path/to/best_motion.pt \
python main.py --profile full
```

## Regenerate Public Sample Data

```bash
python scripts/create_sample_data.py
```
