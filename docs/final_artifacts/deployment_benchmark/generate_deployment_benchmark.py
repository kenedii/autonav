#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "docs" / "final_artifacts" / "deployment_benchmark"
MODEL_PATH = REPO_ROOT / "checkpoints" / "AutoNav-v2" / "AutoNav-v2-34" / "AutoNav-v2-34.pth"
TRT_MODEL_PATH = REPO_ROOT / "inference" / "best_model_trt.pth"
BENCHMARK_COMMAND = """python3 inference/run_autonomous_resnet.py \\
  --arch resnet34 \\
  --exp 3 \\
  --camera cam0 \\
  --cam-backend argus \\
  --cam-sensor-id 0 \\
  --controller-backend pca9685 \\
  --model-path checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth \\
  --trt-model-path inference/best_model_trt.pth \\
  --throttle 0.20 \\
  --no-invert-steering \\
  --debug-timings"""


def run_command(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        text = (result.stdout or "") + (result.stderr or "")
        return result.returncode, text
    except Exception as exc:  # pragma: no cover - defensive
        return 1, f"{type(exc).__name__}: {exc}"


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    uname_code, uname_out = run_command(["uname", "-a"])
    python_version = sys.version.split()[0]
    python_executable = sys.executable

    tegrastats_path = shutil.which("tegrastats")
    nv_tegra_release = Path("/etc/nv_tegra_release")
    is_jetson = nv_tegra_release.exists() or tegrastats_path is not None or platform.system() == "Linux"

    nv_tegra_text = ""
    if nv_tegra_release.exists():
        try:
            nv_tegra_text = nv_tegra_release.read_text(encoding="utf-8").strip()
        except Exception as exc:
            nv_tegra_text = f"Could not read /etc/nv_tegra_release: {exc}"

    model_exists = MODEL_PATH.exists()
    trt_exists = TRT_MODEL_PATH.exists()

    archived_frame_candidates = [
        REPO_ROOT / "cam0_model_view_preview.png",
        REPO_ROOT / "cam0_model_view_160x120.png",
        REPO_ROOT / "cam0_snapshot.png",
    ]
    archived_frame_used = None
    for candidate in archived_frame_candidates:
        if candidate.exists():
            shutil.copyfile(candidate, OUTPUT_DIR / "cam0_runtime_frame.png")
            archived_frame_used = candidate.relative_to(REPO_ROOT)
            break

    timing_lines = [
        "Manual procedure only: no live Jetson benchmark run was executed in this pass.",
        "",
        f"Exact benchmark command to run on Jetson:\n{BENCHMARK_COMMAND}",
        "",
        "Preflight observations from the current environment:",
        f"- uname exit code: {uname_code}",
        f"- uname -a: {uname_out.strip() or 'not available'}",
        f"- python executable: {python_executable}",
        f"- python version: {python_version}",
        f"- checkpoint present: {'yes' if model_exists else 'no'} ({MODEL_PATH.relative_to(REPO_ROOT)})",
        f"- TensorRT engine present: {'yes' if trt_exists else 'no'} ({TRT_MODEL_PATH.relative_to(REPO_ROOT)})",
        f"- tegrastats present: {'yes' if tegrastats_path else 'no'}",
        f"- /etc/nv_tegra_release present: {'yes' if nv_tegra_release.exists() else 'no'}",
    ]
    if nv_tegra_text:
        timing_lines.append(f"- nv_tegra_release: {nv_tegra_text}")
    timing_lines.extend(
        [
            "",
            "Why no benchmark was run:",
            "- The current host is not the Jetson Nano presentation target.",
            "- The validated command depends on CAM0 + Argus + PCA9685 hardware access.",
            "- The expected TensorRT engine file is not present in this local repo snapshot.",
        ]
    )
    write_text(OUTPUT_DIR / "timing_log.txt", "\n".join(timing_lines))

    tegrastats_lines = [
        "Manual procedure only: tegrastats was not captured in this pass.",
        "",
        f"- tegrastats path: {tegrastats_path or 'not found'}",
        f"- /etc/nv_tegra_release present: {'yes' if nv_tegra_release.exists() else 'no'}",
        "- Required Jetson capture command:",
        "  sudo tegrastats --interval 1000",
        "- Recommended capture window:",
        "  start tegrastats, run the validated live command for 30-60 seconds, then stop tegrastats and save the output here.",
    ]
    write_text(OUTPUT_DIR / "tegrastats_log.txt", "\n".join(tegrastats_lines))

    resource_lines = [
        "# Resource summary",
        "",
        "- CPU / GPU / RAM / thermal observations: `not measured` in this pass.",
        "- Reason: the current environment is not the Jetson Nano target and `tegrastats` is unavailable here.",
        "",
        "## What still needs to be measured on Jetson",
        "- average inference time from `--debug-timings` output",
        "- min/max inference time if visible in logs",
        "- end-to-end loop timing if visible",
        "- camera FPS if printed",
        "- CPU / GPU / RAM / thermal behavior from `tegrastats`",
        "",
        "## Practical safety note",
        "- Keep a manual override path active during the live benchmark.",
        "- Have one team member on manual rescue and one on the terminal/dashboard.",
        "- Confirm throttle neutral and steering center before enabling autonomy.",
    ]
    write_text(OUTPUT_DIR / "resource_summary.md", "\n".join(resource_lines))

    report_lines = [
        "# Deployment benchmark report",
        "",
        "## Status",
        "- Manual procedure only: live deployment benchmark not executed in this pass.",
        "",
        "## Environment used for this pass",
        f"- Hardware platform: `{platform.machine()}` host (`{platform.system()}`), not confirmed Jetson Nano",
        f"- OS: `{platform.platform()}`",
        f"- JetPack: `not detected`",
        f"- Python version: `{python_version}`",
        f"- Model path: `{MODEL_PATH.relative_to(REPO_ROOT)}` ({'present' if model_exists else 'missing'})",
        f"- TensorRT model path: `{TRT_MODEL_PATH.relative_to(REPO_ROOT)}` ({'present' if trt_exists else 'missing'})",
        "",
        "## Validated live path to run on Jetson",
        "```bash",
        BENCHMARK_COMMAND,
        "```",
        "",
        "## Measured deployment metrics in this pass",
        "- FPS: `not measured`",
        "- Average inference time: `not measured`",
        "- Min / max inference time: `not measured`",
        "- Total loop time: `not measured`",
        "- Camera FPS: `not measured`",
        "- Controller backend: `pca9685` (expected validated presentation path, not validated in this pass)",
        "",
        "## Hardware validation status",
        "- CAM0 opens: `not validated in this pass`",
        "- Model warms up: `not validated in this pass`",
        f"- TensorRT engine loads: `not validated`; local engine file is {'present' if trt_exists else 'missing'}`",
        "- PCA9685 backend initializes: `not validated in this pass`",
        "- Manual override path still available: `should be confirmed on the vehicle before demo`",
        "",
        "## Why benchmarking was not completed here",
        "- The current environment is a synced development host, not the Jetson Nano presentation target.",
        "- Jetson-specific pieces such as Argus camera access, PCA9685 hardware access, and `tegrastats` are not available here.",
    ]
    if not trt_exists:
        report_lines.append("- The expected `inference/best_model_trt.pth` file is not present in this repo snapshot.")
    if archived_frame_used is not None:
        report_lines.extend(
            [
                "",
                "## Saved frame artifact",
                f"- `cam0_runtime_frame.png` was copied from archived local artifact `{archived_frame_used}`.",
                "- Treat it as a representative model-view/runtime image, not as a fresh live capture from this pass.",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Manual procedure to complete on Jetson",
            "1. Confirm the Jetson boots into the known-good environment.",
            "2. Confirm `checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth` and `inference/best_model_trt.pth` are present.",
            "3. Start `sudo tegrastats --interval 1000` and save the output to `tegrastats_log.txt`.",
            "4. Run the validated live command with `--debug-timings` and tee stdout/stderr into `timing_log.txt`.",
            "5. Let the system run for 30-60 seconds with a team member on manual override.",
            "6. Stop the run, summarize FPS / inference timings / thermal behavior, and update this folder with the measured values.",
            "",
            "## Notes on safety and manual override",
            "- Keep the RC/manual override path active and tested before the benchmark.",
            "- Start with conservative throttle and stop immediately if the vehicle behaves unpredictably.",
            "- Do not present YOLO or SLAM as part of the validated live control loop during this benchmark.",
        ]
    )
    write_text(OUTPUT_DIR / "deployment_benchmark_report.md", "\n".join(report_lines))

    slide_lines = [
        "# Slide-ready deployment bullets",
        "",
        "- Validated live deployment path: CAM0 + AutoNav-v2-34 + TensorRT + PCA9685 on Jetson Nano.",
        "- This repo snapshot contains the main PyTorch checkpoint, but the local TensorRT engine file is not present here.",
        "- Deployment benchmark numbers were not re-measured in this pass because this environment is not the Jetson target.",
        "- The exact Jetson benchmark command is already documented and should be run with `--debug-timings` plus `tegrastats` capture.",
        "- Safety requirement: keep manual override active and assign one team member to rescue control during the live run.",
        "- For the final deck, present measured Jetson numbers only after they are captured on the actual hardware.",
    ]
    write_text(OUTPUT_DIR / "slide_ready_deployment_bullets.md", "\n".join(slide_lines))

    speaker_lines = [
        "# Speaker notes: deployment and inference benchmark",
        "",
        "This is the validated live path we used for the Jetson presentation setup: CAM0 as the driving camera, the AutoNav-v2-34 checkpoint, TensorRT acceleration, and PCA9685 motor control.",
        "",
        "In this documentation pass, I did not rerun the live benchmark because the current environment is not the Jetson Nano itself. That means I can document the exact command and the required benchmark procedure, but I should not claim fresh FPS or inference timing numbers from this host.",
        "",
        "What I can say honestly is that the checkpoint is present, the validated command is documented, and the remaining benchmark evidence should be captured directly on the Jetson using `--debug-timings` and `tegrastats` during a 30 to 60 second run.",
        "",
        "When presenting this slide, keep the emphasis on the exact deployment path and the safety procedure: manual override stays active, one person watches the vehicle, and one person watches the terminal or dashboard.",
    ]
    write_text(OUTPUT_DIR / "speaker_notes_deployment.md", "\n".join(speaker_lines))

    created = sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file())
    print("Exact validated live command:")
    print(BENCHMARK_COMMAND)
    print(f"Hardware available for benchmark: {'yes' if is_jetson and trt_exists else 'no'}")
    print(f"Checkpoint present: {'yes' if model_exists else 'no'}")
    print(f"TensorRT engine present: {'yes' if trt_exists else 'no'}")
    print(f"tegrastats available: {'yes' if tegrastats_path else 'no'}")
    print("Created files:")
    for name in created:
        print(f"- {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
