# Speaker notes: testing and reproducibility

For the final review, we focused on rerunning the tests that best match the current repo architecture and the validated live-demo path. In this pass, the AutoNav v2 loading tests, the SLAM core and replay tests, the fleet API tests, the mission tests, and the preprocessing tests all passed.

That gives us 25 passing targeted tests, plus a clean `py_compile` sweep over the main data collection, training, inference, client API, SLAM, and host-server modules. So the project does have real regression coverage around the parts we are actually presenting.

We also need to be honest about what is not green. `tests/test_runtime_split.py` is stale against the current runtime layout and fails because it expects older hardware helper APIs that no longer exist in `hardware.py`. That is a maintenance gap, but it is not the same thing as the validated live lane-following path being broken.

There is also a data-pipeline metadata suite that currently fails during collection in this host environment because of package-path drift and a pandas / pyarrow / NumPy compatibility issue. We should not count that suite as passing in the final presentation.

For reproducibility, the repo is in a decent state for code review: the live Jetson command is documented, the platform baseline is documented, and the module layout is clear. But it still depends on external model weights and the documented Jetson environment, so we should describe reproducibility as “good with known setup dependencies,” not as one-click turnkey.
