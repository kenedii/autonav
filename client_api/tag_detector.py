import cv2


class AprilTagDetector(object):
    def __init__(self):
        self.available = False
        self.status = "unavailable"
        self._aruco = None
        self._dictionary = None
        self._parameters = None
        self._detector = None
        self._max_working_width = 640
        self._minimum_area_ratio = 0.0008

        try:
            aruco = getattr(cv2, "aruco", None)
            if aruco is None:
                return

            dict_id = getattr(aruco, "DICT_APRILTAG_36h11", None)
            if dict_id is None:
                return

            self._aruco = aruco
            if hasattr(aruco, "getPredefinedDictionary"):
                self._dictionary = aruco.getPredefinedDictionary(dict_id)
            else:
                self._dictionary = aruco.Dictionary_get(dict_id)

            if hasattr(aruco, "DetectorParameters_create"):
                self._parameters = aruco.DetectorParameters_create()
            elif hasattr(aruco, "DetectorParameters"):
                self._parameters = aruco.DetectorParameters()

            if hasattr(aruco, "ArucoDetector"):
                self._detector = aruco.ArucoDetector(self._dictionary, self._parameters)

            self.available = True
            self.status = "available"
        except Exception:
            self.available = False
            self.status = "unavailable"

    def detect(self, frame_bgr):
        if not self.available or frame_bgr is None:
            return []

        try:
            scale = 1.0
            working = frame_bgr
            if len(frame_bgr.shape) == 3 and frame_bgr.shape[1] > self._max_working_width:
                scale = float(self._max_working_width) / float(frame_bgr.shape[1])
                working = cv2.resize(
                    frame_bgr,
                    (self._max_working_width, int(frame_bgr.shape[0] * scale)),
                )
            if len(frame_bgr.shape) == 3:
                gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
            else:
                gray = working
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            if self._detector is not None:
                corners, ids, _ = self._detector.detectMarkers(gray)
            else:
                corners, ids, _ = self._aruco.detectMarkers(
                    gray,
                    self._dictionary,
                    parameters=self._parameters,
                )

            results = []
            if ids is None:
                return results

            for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
                area = abs(cv2.contourArea(marker_corners.astype("float32")))
                if area < (gray.shape[0] * gray.shape[1] * self._minimum_area_ratio):
                    continue
                pts = []
                cx = 0.0
                cy = 0.0
                for point in marker_corners[0]:
                    px = float(point[0] / scale)
                    py = float(point[1] / scale)
                    pts.append([px, py])
                    cx += px
                    cy += py
                if pts:
                    cx /= float(len(pts))
                    cy /= float(len(pts))
                results.append({
                    "id": int(marker_id),
                    "center": [cx, cy],
                    "corners": pts,
                })
            return results
        except Exception:
            return []
