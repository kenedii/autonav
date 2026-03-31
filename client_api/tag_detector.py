import cv2


class AprilTagDetector(object):
    def __init__(self):
        self.available = False
        self.status = "unavailable"
        self._aruco = None
        self._dictionary = None
        self._parameters = None
        self._detector = None

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
            if len(frame_bgr.shape) == 3:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame_bgr

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
                pts = []
                cx = 0.0
                cy = 0.0
                for point in marker_corners[0]:
                    px = float(point[0])
                    py = float(point[1])
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
