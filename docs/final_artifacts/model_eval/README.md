# Model Evaluation

This folder summarizes evaluation artifacts for the final tested AutoNav-v2-34 checkpoint.

## Evaluation snapshot

- Architecture: ResNet34
- Inputs: RGB (120x160)
- Output targets: normalized steering and throttle
- Test split size used for recovered evaluation: 1,431 samples

## Reported metrics

- Steering MAE: 0.1054
- Throttle MAE: 0.1287
- Combined MAE: 0.1171
- Steering pseudo-accuracy: 90.01%

## Plots
<img width="868" height="735" alt="confusion_left_center_right" src="https://github.com/user-attachments/assets/987e61ab-aade-4cb3-8502-8f29fbae86fc" />
<img width="1185" height="735" alt="steering_error_histogram" src="https://github.com/user-attachments/assets/ce9b2cd7-7682-472c-bc10-ceab9fd2347a" />
<img width="885" height="885" alt="steering_pred_vs_true" src="https://github.com/user-attachments/assets/59b603dd-e3b3-4da4-a87f-8f5404861b7d" />
<img width="885" height="885" alt="throttle_pred_vs_true" src="https://github.com/user-attachments/assets/a93e10de-3729-4d54-a257-9f575b036675" />


## Supporting artifacts

- [model_eval_metrics.csv](model_eval_metrics.csv)
- [generate_model_eval.py](generate_model_eval.py)
