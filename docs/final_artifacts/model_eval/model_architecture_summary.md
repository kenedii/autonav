# Model architecture summary

- Backbone: `resnet34` from torchvision, truncated before the classification head.
- Feature extraction: convolutional backbone outputs are passed through `AdaptiveAvgPool2d((1, 1))` and flattened.
- Regression head: `Linear(feature_dim -> 256)`, `ReLU`, `Dropout(0.4)`, `Linear(256 -> 128)`, `ReLU`, `Dropout(0.3)`, `Linear(128 -> 2)`, `Tanh`.
- Output dimensions: 2 values representing normalized steering and normalized throttle.

## Why Tanh is used
- `Tanh` bounds the raw control outputs to `[-1, 1]`, which matches the normalized control-label convention used across the repo.
- Bounded outputs reduce the chance of unreasonably large regression values during inference.

## Why normalized steering / throttle are used
- Steering and throttle are learned on a common normalized scale, which stabilizes regression training compared with raw PWM units.
- Normalized outputs are easier to compare across datasets and can later be mapped back into hardware control space inside runtime code.
