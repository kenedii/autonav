#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[3]
CLIENT_API_DIR = REPO_ROOT / "fleet" / "fleet_management_app" / "client_api"
if str(CLIENT_API_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_API_DIR))

os.environ.setdefault("YOLO_CONFIG_DIR", str((REPO_ROOT / "docs" / "final_artifacts" / "yolo").resolve()))

from models import ObjectDetector  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run the repo YOLO wrapper on a single image.")
    parser.add_argument("--image", required=True, help="Path to the image to benchmark")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights path passed into ObjectDetector")
    parser.add_argument(
        "--output-image",
        default=str(REPO_ROOT / "docs" / "final_artifacts" / "yolo" / "yolo_annotated_example.png"),
        help="Where to save the annotated image if inference succeeds",
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "docs" / "final_artifacts" / "yolo" / "yolo_smoke_result.md"),
        help="Where to save the markdown smoke-test result",
    )
    return parser.parse_args()


def annotate_image(image_bgr, detections):
    annotated = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        label = f"class {det['class']} conf {det['conf']:.3f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def main():
    args = parse_args()
    image_path = Path(args.image)
    model_path = args.model
    output_image = Path(args.output_image)
    report_path = Path(args.report_path)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# YOLO smoke benchmark result",
        "",
        f"- Command: `python3 docs/final_artifacts/yolo/yolo_smoke_benchmark.py --image {args.image} --model {args.model}`",
        f"- Image used: `{image_path}`",
        f"- Model argument: `{model_path}`",
    ]

    if not image_path.exists():
        report_lines.extend(
            [
                "- Status: could not run",
                "- Reason: input image file is missing.",
                "- Detections: `not measured`",
                "- Latency: `not measured`",
                "- FPS: `not measured`",
            ]
        )
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(json.dumps({"status": "error", "reason": "missing_image", "image": str(image_path)}, indent=2))
        return 0

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        report_lines.extend(
            [
                "- Status: could not run",
                "- Reason: OpenCV could not decode the input image.",
                "- Detections: `not measured`",
                "- Latency: `not measured`",
                "- FPS: `not measured`",
            ]
        )
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(json.dumps({"status": "error", "reason": "image_decode_failed", "image": str(image_path)}, indent=2))
        return 0

    config = {"detection_model": model_path}
    detector = ObjectDetector(config)
    model_loaded = detector.model is not None

    if not model_loaded:
        reasons = []
        if not os.path.exists(model_path):
            reasons.append("weights file does not exist at the provided path")
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            reasons.append("ultralytics is not installed")
        if not reasons:
            reasons.append("ObjectDetector did not load a YOLO model")

        report_lines.extend(
            [
                "- Status: could not run full inference",
                "- Detections: `not measured`",
                "- Latency: `not measured`",
                "- FPS: `not measured`",
                f"- Reason: {'; '.join(reasons)}.",
                "- False positives / failure cases: `not evaluated in this run`.",
            ]
        )
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "model_loaded": False,
                    "image": str(image_path),
                    "model": model_path,
                    "reason": reasons,
                },
                indent=2,
            )
        )
        return 0

    start = time.perf_counter()
    detections = detector.detect(image_bgr)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

    annotated = annotate_image(image_bgr, detections)
    cv2.imwrite(str(output_image), annotated)

    report_lines.extend(
        [
            "- Status: inference succeeded",
            f"- Detections: `{len(detections)}`",
            f"- Latency: `{elapsed_ms:.3f} ms`",
            f"- FPS: `{fps:.3f}`",
            "- False positives / failure cases: review the annotated output manually; this smoke test does not score detections against ground truth.",
            "",
            "## Detection list",
        ]
    )
    if detections:
        for det in detections:
            report_lines.append(
                f"- class `{det['class']}` bbox `{[round(v, 2) for v in det['bbox']]}` conf `{det['conf']:.4f}`"
            )
    else:
        report_lines.append("- No detections returned.")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "model_loaded": True,
                "image": str(image_path),
                "model": model_path,
                "latency_ms": elapsed_ms,
                "fps": fps,
                "detections": detections,
                "output_image": str(output_image),
                "report_path": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
