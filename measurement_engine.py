import cv2
import numpy as np
import math

class MeasurementEngine:
    """
    Phase 3: Deterministic Measurement & Logic Pipeline.

    This engine receives keypoints predicted by the YOLO-Pose model.
    It performs rigid mathematical operations to compute real-world measurements (millimeters),
    alignment (asymmetry), and lighting validation (RGB/HSV color extraction) required for
    Industrial Automated Optical Inspection (AOI).
    """

    def __init__(self, camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = None, z_distance_mm: float = 500.0):
        """
        Initializes the measurement engine with OpenCV camera intrinsic and extrinsic parameters.
        These parameters are usually obtained via cv2.calibrateCamera using a checkerboard.
        """
        # Default placeholder intrinsic matrix for a 1080p camera if none provided
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [800.0, 0.0, 960.0],
                [0.0, 800.0, 540.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix

        # Default zero distortion if none provided
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1), dtype=np.float32)

        # The physical distance from the camera lens to the conveyor belt surface in millimeters.
        # This acts as our simplified extrinsic translation vector (Z-axis).
        self.z_distance_mm = z_distance_mm

    def pixel_to_world(self, u: float, v: float) -> tuple:
        """
        Converts a 2D pixel coordinate (u, v) into a 3D world coordinate (X, Y) in millimeters.
        Assumes the object is flat on the conveyor belt at `z_distance_mm`.
        First, we undistort the pixel points, then use the intrinsic matrix to find the real-world scale.
        """
        # Create a single-point array in the shape required by cv2.undistortPoints
        pts = np.array([[[float(u), float(v)]]], dtype=np.float32)

        # Undistort the point to normalized device coordinates
        undistorted = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs)

        # Extract normalized coordinates (x', y')
        x_norm = undistorted[0][0][0]
        y_norm = undistorted[0][0][1]

        # Project to world coordinates using the known Z distance
        X_world = x_norm * self.z_distance_mm
        Y_world = y_norm * self.z_distance_mm

        return (X_world, Y_world)

    def calculate_distance(self, p1: tuple, p2: tuple) -> float:
        """
        Calculates the true Euclidean distance in millimeters between two pixel points,
        accounting for camera lens distortion and perspective projection.
        """
        w1 = self.pixel_to_world(p1[0], p1[1])
        w2 = self.pixel_to_world(p2[0], p2[1])

        # Euclidean distance in the 3D world (Z is constant, so we only need X and Y)
        world_dist = math.sqrt((w2[0] - w1[0])**2 + (w2[1] - w1[1])**2)
        return world_dist

    def compute_dimensions(self, keypoints: list) -> dict:
        """
        Given the 5 keypoints (Top-Left, Top-Right, Bottom-Right, Bottom-Left, Center-Artwork),
        this function calculates the length and width of the package.

        Note: The points must be in the expected order, or mapped dynamically
        based on coordinate geometry. For this AOI pipeline, we assume
        a consistent left-to-right, top-to-bottom layout:
        idx 0: Top-Left Corner
        idx 1: Top-Right Corner
        idx 2: Bottom-Right Corner
        idx 3: Bottom-Left Corner
        """
        if len(keypoints) < 4:
            return {"width": 0.0, "height": 0.0}

        tl, tr, br, bl = keypoints[0], keypoints[1], keypoints[2], keypoints[3]

        # Calculate average Top and Bottom Width
        top_width = self.calculate_distance(tl, tr)
        bottom_width = self.calculate_distance(bl, br)
        avg_width = (top_width + bottom_width) / 2.0

        # Calculate average Left and Right Height/Length
        left_height = self.calculate_distance(tl, bl)
        right_height = self.calculate_distance(tr, br)
        avg_height = (left_height + right_height) / 2.0

        return {"width": avg_width, "height": avg_height}

    def compute_asymmetry(self, keypoints: list) -> float:
        """
        Calculates alignment/asymmetry of the product.
        The center artwork keypoint (idx 4) should theoretically be equidistant
        from the top-left and bottom-right corners (or equidistant from the edges).
        This function returns the deviation error in millimeters.
        0.0mm deviation means perfectly aligned artwork.
        """
        if len(keypoints) != 5:
            return 0.0

        tl, tr, br, bl, center = keypoints[0], keypoints[1], keypoints[2], keypoints[3], keypoints[4]

        # Measure distance from Center to Top-Left and Center to Bottom-Right
        dist_tl_to_center = self.calculate_distance(tl, center)
        dist_center_to_br = self.calculate_distance(center, br)

        # The difference in these distances indicates how "off-center" the artwork is.
        asymmetry_error = abs(dist_tl_to_center - dist_center_to_br)
        return asymmetry_error

    def extract_color(self, frame: np.ndarray, top_left_pt: tuple, bottom_right_pt: tuple) -> dict:
        """
        Accepts bounding box coordinates for a specific inspection zone (e.g., Gusset or Label)
        and extracts the average RGB and HSV values.
        This is crucial for detecting misprints or poor lighting conditions.
        """
        # Ensure coordinates are integers and within image bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(top_left_pt[0])), max(0, int(top_left_pt[1]))
        x2, y2 = min(w, int(bottom_right_pt[0])), min(h, int(bottom_right_pt[1]))

        if x1 >= x2 or y1 >= y2:
            # Invalid ROI size
            return {"rgb": (0, 0, 0), "hsv": (0, 0, 0)}

        # Extract the Region of Interest (ROI)
        roi = frame[y1:y2, x1:x2]

        # Calculate average BGR (OpenCV's default color space) and convert to RGB
        avg_bgr = cv2.mean(roi)[:3] # mean returns 4 elements (B,G,R,Alpha), we want first 3
        avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))

        # Convert ROI to HSV space to extract average Hue/Sat/Val
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_hsv_raw = cv2.mean(hsv_roi)[:3]
        avg_hsv = (int(avg_hsv_raw[0]), int(avg_hsv_raw[1]), int(avg_hsv_raw[2]))

        return {
            "rgb": avg_rgb,
            "hsv": avg_hsv
        }
