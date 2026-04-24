# AutoNav-v2-34 evaluation report

- Evaluation command: `python3 docs/final_artifacts/model_eval/generate_model_eval.py`
- Model path: `checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth`
- Architecture: `resnet34`
- Input shape: `3 x 120 x 160`
- Output targets: normalized `steer_norm` and `throttle_norm`
- Dataset used: archived raw run CSVs under `jetracer/train/runs_rgb_depth/run_*/dataset.csv` (RGB-only / experiment-3 style evaluation)
- Usable labeled RGB rows found: `9535`
- Evaluation samples used: `1431`
- Train / val / test split used for recovered evaluation: `6674 / 1430 / 1431`

## Metrics
- Steering MAE: `0.1054`
- Throttle MAE: `0.1287`
- Combined MAE: `0.1171`
- Steering pseudo-accuracy: `90.01%`
- Left bin accuracy: `87.33%`
- Center bin accuracy: `89.95%`
- Right bin accuracy: `91.41%`

## Overfitting / underfitting evidence
- Original training and validation curves were not available in this repo snapshot.
- This means overfitting/underfitting cannot be concluded directly from the checkpoint alone.
- What we can say: the recovered evaluation uses a held-out split from the archived raw runs, and the dataset remains center-heavy.

## Limitations
- This evaluation uses the archived raw run CSVs under `jetracer/train/runs_rgb_depth/`, because the original `combined_augmented_dataset.csv` / `combined_cleaned_dataset.csv` artifacts are not checked in.
- The recreated 70/15/15 split matches the project proportions but is not guaranteed to be identical to the original training/test partition used when the checkpoint was first produced.
- Evaluation compares raw checkpoint outputs to normalized `steer_norm` / `throttle_norm` labels. It does not apply runtime-only steering inversion or throttle scaling hooks.
- Original training curves were not available, so overfitting/underfitting cannot be concluded directly from this checkpoint-only evaluation.
- The archived dataset is center-heavy, so pseudo-accuracy may overstate edge-case performance on sharp recoveries.
- 1 rows were excluded because they lacked `rgb_path`, `steer_norm`, `throttle_norm`, or a resolvable image path.
