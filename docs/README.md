# AutoNav Documentation

This repository documentation is organized around the final tested version of AutoNav and supporting technical artifacts.

## Final tested scope

The final tested AutoNav runtime includes:
- Jetson Nano lane following with CAM0 + ResNet34 (AutoNav-v2-34) + TensorRT + PCA9685
- Optional deployment path for Rockchip/Radxa devices using RKNN runtime and Pi Pico serial control
- Fleet management APIs and frontend controls for model deployment, YOLO feature toggles, and SLAM navigation controls

## Documentation map

- [final_artifacts/README.md](final_artifacts/README.md): index for technical evidence artifacts
- [capstone_report_notes.md](capstone_report_notes.md): presentation-oriented, planning, and report material moved out of the core GitHub docs flow

## Notes

The docs were cleaned for maintainability and typical GitHub conventions:
- one canonical markdown file per retained docs folder
- image-heavy evidence embedded directly in folder README files
- duplicate and presentation-script style content moved to capstone report notes
