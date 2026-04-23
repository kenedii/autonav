# Speaker notes: model and evaluation

For the final live lane-follow model, we are using the AutoNav-v2-34 checkpoint, which is a ResNet34-based regression model with two outputs: normalized steering and normalized throttle.

Because the original combined training CSVs are not checked into this repo snapshot, this evaluation is a recovered evaluation on the archived raw run data. I used the RGB-only path that matches experiment 3 and recreated the same 70/15/15 split proportions from the training code.

On that recovered held-out split, the checkpoint reached a steering MAE of 0.1054, a throttle MAE of 0.1287, and a steering pseudo-accuracy of 90.01 percent.

One thing to explain honestly is that pseudo-accuracy is based on left, center, and right steering bins. It is useful for presentation, but it can look better than the true edge-case control quality because the dataset is still center-heavy.

Also, this repo snapshot does not include the original training curves, so we should avoid making strong claims about overfitting or underfitting. The safest wording is that we recovered a held-out evaluation from the archived raw data and used that as evidence for the final checkpoint.
