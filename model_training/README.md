# Model Training

This folder contains training code for the autonomous steering/throttle models and shared dataset loading utilities.

## Training tracks in this repo

There are two generations of training code here:

- `train_model_experiments.py` (newer): experiment-oriented workflow for training with different sensor combinations (for example RGB-only, RGB + depth, RGB + rear camera).
- `train_model_resnet.py` (older): generic RGB-only style pipeline used by the earlier baseline training flow.

Use the experiments script for new model development. Keep the older script for reproducibility of historical RGB-only results.

## Files

- `train_model_experiments.py`: Runs multiple predefined experiment configurations and model backbones.
- `train_model_resnet.py`: Legacy single-track ResNet trainer.
- `dataset_loader.py`: Shared dataset parsing/validation helpers, including metadata checks.
- `requirements.txt`: Python dependencies for model training.

## Typical workflow

1. Prepare/clean CSV datasets in `data_collection/`.
2. Install dependencies from `model_training/requirements.txt`.
3. Run `train_model_experiments.py` for sensor-combination experiments.
4. Evaluate checkpoint metrics and select best model for optimization/deployment.

## Example commands

```bash
python model_training/train_model_experiments.py
python model_training/train_model_resnet.py --dataset-path combined_augmented_dataset.csv --model-architecture resnet34
```

## Notes on experiments

- Experiments are designed to compare how sensor modality combinations affect control quality.
- Results typically include per-experiment artifacts and metrics under experiment output directories.
- Keep dataset metadata consistent (`rgb_source`, `preprocess_profile`) unless explicitly running mixed-source training.
