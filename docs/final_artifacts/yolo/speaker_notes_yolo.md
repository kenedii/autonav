# Speaker notes: YOLO prototype

We added YOLO as a prototype perception feature inside the fleet-management runtime. The key point is that this is a real code path, not just a slide idea: the repo has an `ObjectDetector` wrapper, the car loop can call it, and detections are stored in runtime state.

At the same time, we should present it honestly. In the current repo, YOLO is advisory only. It does not influence steering, throttle, or braking, and the dashboard does not yet have a dedicated detections view. That means it is appropriate for code review and prototype discussion, but not as a production-ready obstacle-avoidance claim.

For the final presentation, the safest wording is that YOLO demonstrates how semantic perception could plug into the platform. The evidence we have here is the wrapper implementation, the normalized output format, and a smoke-test path. If local weights are missing, we should say that clearly rather than pretend we ran a full benchmark.
