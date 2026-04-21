import cv2
import numpy as np
import logging
import math
import time

logger = logging.getLogger("SLAM")

class VisualSlamSystem:
    def __init__(self, width=640, height=480, focal_length=None):
        self.width = width
        self.height = height
        # Estimate focal length if not provided. D435 approx FOV ~87deg
        self.focal_length = focal_length if focal_length else (width / 2) / math.tan(math.radians(87 / 2))
        self.pp = (width / 2, height / 2)

        # CV Config
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.prev_gray = None
        self.prev_pts = None
        self.prev_depth = None
        
        # World State [x, y, theta] (2D plane)
        # Coordinates: X (Right), Y (Forward from start)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0 # Radians
        
        self.initialized = False
        self.trajectory = [] # List of (x,y)
        
        # Scaling factor for Monocular (No Depth) - Heuristic
        self.estimated_speed_scale = 0.1 # meters per frame if unknown
        self.last_imu_time = None
        self.imu_data = {'accel': None, 'gyro': None} # Store last seen
        self.last_tracking_points = 0
        self.last_rgbd_points = 0
        self.last_motion_source = "bootstrap"
        self.last_motion = {"forward_m": 0.0, "lateral_m": 0.0, "yaw_rad": 0.0}

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.trajectory = []
        self.initialized = False
        self.prev_gray = None
        self.prev_pts = None
        self.prev_depth = None
        self.last_imu_time = None
        self.last_tracking_points = 0
        self.last_rgbd_points = 0
        self.last_motion_source = "bootstrap"
        self.last_motion = {"forward_m": 0.0, "lateral_m": 0.0, "yaw_rad": 0.0}

    @staticmethod
    def _normalize_depth_map(depth_map):
        if depth_map is None:
            return None

        depth = np.asarray(depth_map)
        if depth.ndim == 3:
            # Colorized preview depth is not metric, so do not use it for RGB-D odometry.
            return None

        if depth.dtype == np.uint8:
            return None

        depth = depth.astype(np.float32)
        if depth.size == 0:
            return None

        if np.nanmax(depth) > 50.0:
            depth /= 1000.0

        return depth

    @staticmethod
    def _rotation_y(angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ], dtype=np.float32)

    @staticmethod
    def _clip_motion(value, limit):
        return float(np.clip(value, -limit, limit))

    def _sample_depth_m(self, depth_map_m, u, v):
        if depth_map_m is None:
            return None

        x = int(round(float(u)))
        y = int(round(float(v)))
        x0 = max(0, x - 1)
        x1 = min(depth_map_m.shape[1], x + 2)
        y0 = max(0, y - 1)
        y1 = min(depth_map_m.shape[0], y + 2)
        patch = depth_map_m[y0:y1, x0:x1]
        valid = patch[(patch > 0.05) & (patch < 5.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _depth_intrinsics(self, frame_shape, depth_shape):
        frame_h, frame_w = frame_shape[:2]
        depth_h, depth_w = depth_shape[:2]
        scale_x = float(depth_w) / float(frame_w)
        scale_y = float(depth_h) / float(frame_h)
        fx = self.focal_length * scale_x
        fy = self.focal_length * scale_y
        cx = self.pp[0] * scale_x
        cy = self.pp[1] * scale_y
        return scale_x, scale_y, fx, fy, cx, cy

    def _backproject(self, u, v, z_m, intrinsics):
        _, _, fx, fy, cx, cy = intrinsics
        x = (u - cx) * z_m / fx
        y = (v - cy) * z_m / fy
        return np.array([x, y, z_m], dtype=np.float32)

    def _build_rgbd_correspondences(self, prev_pts, curr_pts, prev_depth, curr_depth, frame_shape):
        prev_intr = self._depth_intrinsics(frame_shape, prev_depth.shape)
        curr_intr = self._depth_intrinsics(frame_shape, curr_depth.shape)

        prev_xyz = []
        curr_xyz = []
        for p_prev, p_curr in zip(prev_pts, curr_pts):
            prev_u = p_prev[0] * prev_intr[0]
            prev_v = p_prev[1] * prev_intr[1]
            curr_u = p_curr[0] * curr_intr[0]
            curr_v = p_curr[1] * curr_intr[1]

            z_prev = self._sample_depth_m(prev_depth, prev_u, prev_v)
            z_curr = self._sample_depth_m(curr_depth, curr_u, curr_v)
            if z_prev is None or z_curr is None:
                continue

            prev_xyz.append(self._backproject(prev_u, prev_v, z_prev, prev_intr))
            curr_xyz.append(self._backproject(curr_u, curr_v, z_curr, curr_intr))

        if len(prev_xyz) < 6:
            return None, None

        return np.asarray(prev_xyz, dtype=np.float32), np.asarray(curr_xyz, dtype=np.float32)

    @staticmethod
    def _fit_rigid_transform(prev_xyz, curr_xyz):
        if prev_xyz is None or curr_xyz is None or len(prev_xyz) < 6:
            return None, None, None

        def solve(src, dst):
            src_centroid = np.mean(src, axis=0)
            dst_centroid = np.mean(dst, axis=0)
            src_centered = src - src_centroid
            dst_centered = dst - dst_centroid
            H = src_centered.T @ dst_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1.0
                R = Vt.T @ U.T
            t = dst_centroid - (R @ src_centroid)
            transformed = src @ R.T + t
            residuals = np.linalg.norm(transformed - dst, axis=1)
            return R, t, residuals

        R, t, residuals = solve(prev_xyz, curr_xyz)
        if residuals is None:
            return None, None, None

        keep_mask = residuals <= max(0.05, float(np.percentile(residuals, 75)))
        if np.count_nonzero(keep_mask) >= 6 and np.count_nonzero(~keep_mask) > 0:
            R, t, residuals = solve(prev_xyz[keep_mask], curr_xyz[keep_mask])

        return R, t, residuals

    def _estimate_metric_motion_from_point_clouds(self, prev_xyz, curr_xyz):
        R_points, t_points, residuals = self._fit_rigid_transform(prev_xyz, curr_xyz)
        if R_points is None or t_points is None:
            return None

        camera_rotation = R_points.T
        camera_translation = -(camera_rotation @ t_points)
        yaw = math.atan2(float(camera_rotation[0, 2]), float(camera_rotation[2, 2]))

        motion = {
            "forward_m": self._clip_motion(camera_translation[2], 0.35),
            "lateral_m": self._clip_motion(camera_translation[0], 0.25),
            "yaw_rad": self._clip_motion(yaw, math.radians(30.0)),
            "point_count": int(len(prev_xyz)),
            "median_residual_m": float(np.median(residuals)) if residuals is not None and len(residuals) else None,
            "source": "rgbd",
        }

        if motion["median_residual_m"] is not None and motion["median_residual_m"] > 0.08:
            return None

        return motion

    def _estimate_metric_motion(self, prev_pts, curr_pts, prev_depth, curr_depth, frame_shape):
        prev_xyz, curr_xyz = self._build_rgbd_correspondences(prev_pts, curr_pts, prev_depth, curr_depth, frame_shape)
        if prev_xyz is None or curr_xyz is None:
            return None
        return self._estimate_metric_motion_from_point_clouds(prev_xyz, curr_xyz)

    def _estimate_visual_motion(self, good_prev, good_curr, throttle_val):
        flow_vecs = good_curr - good_prev
        mean_flow = np.mean(flow_vecs, axis=0) # [dx, dy]
        visual_rotation = self._clip_motion(-mean_flow[0] / self.focal_length, math.radians(20.0))

        speed = 0.0
        if throttle_val > 0:
            speed = (throttle_val - 0.15) * 5.0

        return {
            "forward_m": self._clip_motion(speed * 0.06, 0.25),
            "lateral_m": 0.0,
            "yaw_rad": visual_rotation,
            "point_count": int(len(good_curr)),
            "median_residual_m": None,
            "source": "rgb",
        }

    def _fuse_rotation(self, visual_rotation, dt_imu):
        rotation = visual_rotation
        if self.imu_data and self.imu_data.get('gyro'):
            gyro_y = self.imu_data['gyro'][1]
            gyro_rotation = -gyro_y * dt_imu
            if abs(visual_rotation) > 0.001:
                rotation = 0.6 * gyro_rotation + 0.4 * visual_rotation
            else:
                rotation = gyro_rotation
        return self._clip_motion(rotation, math.radians(30.0))

    def _integrate_body_motion(self, forward_m, lateral_m):
        # Standard planar pose integration with heading measured from +X.
        self.x += forward_m * math.cos(self.theta) + lateral_m * math.sin(self.theta)
        self.y += forward_m * math.sin(self.theta) - lateral_m * math.cos(self.theta)

    def update(self, img_bgr, depth_map=None, throttle_val=0.0, imu_data=None):
        try:
            self.imu_data = imu_data or self.imu_data
            timestamp = time.time()
            dt_imu = 0.0
            if self.last_imu_time:
                dt_imu = timestamp - self.last_imu_time
            self.last_imu_time = timestamp

            frame_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            depth_metric = self._normalize_depth_map(depth_map)
            
            if not self.initialized:
                self.prev_gray = frame_gray
                self.prev_pts = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
                self.prev_depth = depth_metric
                self.initialized = True
                self.last_motion_source = "bootstrap"
                return self._get_state()

            if self.prev_pts is None or len(self.prev_pts) < 10:
                 self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
                 if self.prev_pts is None:
                     self.prev_gray = frame_gray
                     self.prev_depth = depth_metric
                     return self._get_state()

            # Optical Flow
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_pts, None, **self.lk_params)
            if curr_pts is None or status is None:
                self.initialized = False
                self.prev_gray = frame_gray
                self.prev_depth = depth_metric
                self.prev_pts = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
                self.last_motion_source = "reseed"
                return self._get_state()
            
            # Filter valid points
            good_prev = self.prev_pts[status == 1]
            good_curr = curr_pts[status == 1]
            self.last_tracking_points = int(len(good_curr))
            self.last_rgbd_points = 0

            if len(good_curr) < 5:
                # Lost track, re-init next frame
                self.initialized = False
                self.prev_gray = frame_gray
                self.prev_depth = depth_metric
                self.prev_pts = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
                self.last_motion_source = "reseed"
                return self._get_state()
            
            # Calculate Motion
            visual_motion = None
            if depth_metric is not None and self.prev_depth is not None:
                visual_motion = self._estimate_metric_motion(
                    good_prev,
                    good_curr,
                    self.prev_depth,
                    depth_metric,
                    frame_gray.shape,
                )

            if visual_motion is None:
                visual_motion = self._estimate_visual_motion(good_prev, good_curr, throttle_val)

            self.last_motion_source = visual_motion["source"]
            self.last_rgbd_points = int(visual_motion.get("point_count") or 0) if visual_motion["source"] == "rgbd" else 0
            rotation = self._fuse_rotation(visual_motion["yaw_rad"], dt_imu)

            # Update State
            self.theta += rotation
            
            # Apply Translation
            self._integrate_body_motion(
                visual_motion["forward_m"],
                visual_motion["lateral_m"],
            )
            self.last_motion = {
                "forward_m": float(visual_motion["forward_m"]),
                "lateral_m": float(visual_motion["lateral_m"]),
                "yaw_rad": float(rotation),
            }
            
            # Store history periodically
            if len(self.trajectory) == 0 or np.linalg.norm([self.x - self.trajectory[-1][0], self.y - self.trajectory[-1][1]]) > 0.1:
                self.trajectory.append((self.x, self.y))
            
            # Prepare next
            self.prev_gray = frame_gray
            self.prev_depth = depth_metric
            self.prev_pts = good_curr.reshape(-1, 1, 2)
            
            return self._get_state()
            
        except Exception as e:
            logger.error(f"SLAM Error: {e}")
            return self._get_state()

    def _get_state(self):
        return {
            "x": float(self.x),
            "y": float(self.y),
            "theta": float(self.theta),
            "trajectory": self.trajectory[-50:], # send last 50 points
            "imu": self.imu_data,
            "tracking_points": int(self.last_tracking_points),
            "rgbd_points": int(self.last_rgbd_points),
            "motion_source": self.last_motion_source,
            "last_motion": dict(self.last_motion),
        }
