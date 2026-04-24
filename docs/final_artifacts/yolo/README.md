# YOLO Status

This folder documents the YOLO detection prototype path in the final tested AutoNav repository.

## Runtime status

- YOLO integration exists in fleet client code.
- Intended for advisory detections and iterative development.
- Not claimed as a validated autonomous control safety path by itself.

## Evidence

- [yolo_smoke_result.md](yolo_smoke_result.md)
- [yolo_smoke_benchmark.py](yolo_smoke_benchmark.py)

## Related runtime behavior

- Detections are produced in client state as `state["detections"]`.
- Host dashboard now supports YOLO enable/disable, model replacement, threshold updates, and detection export.
