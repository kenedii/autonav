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

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.trajectory = []
        self.initialized = False
        self.prev_gray = None
        self.prev_pts = None
        self.last_imu_time = None

    def update(self, img_bgr, depth_map=None, throttle_val=0.0, imu_data=None):
        try:
            self.imu_data = imu_data or self.imu_data
            timestamp = time.time()
            dt_imu = 0.0
            if self.last_imu_time:
                dt_imu = timestamp - self.last_imu_time
            self.last_imu_time = timestamp

            frame_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            if not self.initialized:
                self.prev_gray = frame_gray
                self.prev_pts = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
                self.initialized = True
                return self._get_state()

            if self.prev_pts is None or len(self.prev_pts) < 10:
                 self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
                 if self.prev_pts is None:
                     return self._get_state()

            # Optical Flow
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_pts, None, **self.lk_params)
            
            # Filter valid points
            good_prev = self.prev_pts[status == 1]
            good_curr = curr_pts[status == 1]

            if len(good_curr) < 5:
                # Lost track, re-init next frame
                self.initialized = False
                return self._get_state()
            
            # Calculate Motion
            translation = 0.0
            rotation = 0.0

            visual_rotation = 0.0
            visual_translation = 0.0

            if depth_map is not None:
                # RGB-D Mode
                valid_depths = []
                for i, (p_cur, p_prev) in enumerate(zip(good_curr, good_prev)):
                    u, v = int(p_prev[0]), int(p_prev[1])
                    if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                        z = depth_map[v, u] 
                        if 100 < z < 5000:
                             valid_depths.append(z / 1000.0)

                if len(valid_depths) > 5:
                    E, mask = cv2.findEssentialMat(good_curr, good_prev, focal=self.focal_length, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    if E is not None:
                        _, R, t, mask = cv2.recoverPose(E, good_curr, good_prev, focal=self.focal_length, pp=self.pp)
                        scale = np.median(valid_depths) * np.linalg.norm(t) * 0.1
                        visual_translation = scale
                        # yaw approx from R[0,2] (-sin(theta))
                        visual_rotation = -math.asin(max(min(R[0,2], 1.0), -1.0))
            else:
                # RGB Only (Visual Compass)
                flow_vecs = good_curr - good_prev
                mean_flow = np.mean(flow_vecs, axis=0) # [dx, dy]
                visual_rotation = -mean_flow[0] / self.focal_length
                
                # Throttle based speed
                speed = 0.0
                if throttle_val > 0:
                    speed = (throttle_val - 0.15) * 5.0
                visual_translation = speed * 0.06

            # --- FENSOR FUSION (IMU + VISUAL) ---
            rotation = visual_rotation
            translation = visual_translation

            if self.imu_data and self.imu_data.get('gyro'):
                # Gyro Y is Yaw? No, usually Z is forward, X right, Y down. 
                # Rotation about Y is Yaw.
                # Check sensor frame: Realsense D435i:
                # Accel/Gyro axes are usually aligned with camera axes.
                # If camera is standard: X-Right, Y-Down, Z-Forward.
                # Yaw is rotation around Y-Axis.
                # Gyro returns rad/s.
                gyro_y = self.imu_data['gyro'][1]
                
                # Integration
                gyro_delta = -gyro_y * dt_imu # -ve because Y is down? 
                # Actually, positive rotation around Y (down) moves X (right) to Z (forward) -> Turning Left?
                # Right-Hand Rule: Thumb down. Fingers curl CW (viewed from top).
                # So +GyroY = Right Turn?
                # If +GyroY = Right Turn, then Angle increases (if CW positive).
                # My theta usually standard math: CCW positive.
                # So +GyroY (CW) -> -Theta.
                # Let's try: rotation = -gyro_y * dt
                
                # Complementary Filter
                # If visual is available, fuse. If not, pure gyro.
                if abs(visual_rotation) > 0.001:
                    rotation = 0.6 * (-gyro_y * dt_imu) + 0.4 * visual_rotation
                else:
                    rotation = -gyro_y * dt_imu

            # Update State
            self.theta += rotation
            
            # Apply Translation
            self.x += translation * math.cos(self.theta)
            self.y += translation * math.sin(self.theta)
            
            # Store history periodically
            if len(self.trajectory) == 0 or np.linalg.norm([self.x - self.trajectory[-1][0], self.y - self.trajectory[-1][1]]) > 0.1:
                self.trajectory.append((self.x, self.y))
            
            # Prepare next
            self.prev_gray = frame_gray
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
            "imu": self.imu_data
        }
