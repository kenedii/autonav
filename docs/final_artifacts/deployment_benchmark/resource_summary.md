# Resource summary

- CPU / GPU / RAM / thermal observations: `not measured` in this pass.
- Reason: the current environment is not the Jetson Nano target and `tegrastats` is unavailable here.

## What still needs to be measured on Jetson
- average inference time from `--debug-timings` output
- min/max inference time if visible in logs
- end-to-end loop timing if visible
- camera FPS if printed
- CPU / GPU / RAM / thermal behavior from `tegrastats`

## Practical safety note
- Keep a manual override path active during the live benchmark.
- Have one team member on manual rescue and one on the terminal/dashboard.
- Confirm throttle neutral and steering center before enabling autonomy.
