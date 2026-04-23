# Slide-ready model bullets

- Final live lane-follow checkpoint: `AutoNav-v2-34` at `checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth`.
- Architecture recovered from the checkpoint: `resnet34` with a 2-output regression head for steering and throttle.
- Recovered evaluation used `9535` usable RGB-labeled rows from archived run CSVs, with a recreated `70 / 15 / 15` split (`1431` held-out test samples).
- Held-out recovered metrics: steering MAE `0.1054`, throttle MAE `0.1287`, steering pseudo-accuracy `90.01%`.
- Bin-level steering accuracy: left `87.33%`, center `89.95%`, right `91.41%`.
- This is a recovered evaluation on archived raw runs, not the original combined training CSV artifact used when the checkpoint was first created.
- The README-reported `94.20%` pseudo-accuracy should be treated as historical unless regenerated from the original training artifact.
