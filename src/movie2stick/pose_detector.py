"""Pose detection module using MediaPipe."""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class PoseKeypoints:
    """Container for pose keypoints of a single person."""
    
    # MediaPipe pose landmark indices
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    # Connections for drawing stickman
    BODY_CONNECTIONS = [
        # Head
        (NOSE, LEFT_EYE),
        (NOSE, RIGHT_EYE),
        (LEFT_EYE, LEFT_EAR),
        (RIGHT_EYE, RIGHT_EAR),
        # Torso
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
        # Left arm
        (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
        # Right arm
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST),
        # Left leg
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        # Right leg
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE),
    ]
    
    # Simplified connections for cleaner stickman
    STICKMAN_CONNECTIONS = [
        # Spine (center line from head to pelvis)
        (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER),  # Head to shoulders center
        # Shoulders
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        # Left arm
        (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
        # Right arm
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST),
        # Hips
        (LEFT_HIP, RIGHT_HIP),
        # Left leg
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        # Right leg
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE),
    ]
    
    def __init__(self, landmarks: np.ndarray, visibility: np.ndarray, image_shape: Tuple[int, int]):
        """Initialize pose keypoints.
        
        Args:
            landmarks: Array of shape (33, 2) or (33, 3) with x, y (, z) coordinates normalized [0, 1]
            visibility: Array of shape (33,) with visibility scores for each landmark
            image_shape: Tuple of (height, width) for converting normalized coordinates
        """
        self.landmarks = landmarks
        self.visibility = visibility
        self.image_height, self.image_width = image_shape
        
    def get_point(self, idx: int, pixel_coords: bool = True) -> Optional[Tuple[int, int]]:
        """Get a specific landmark point.
        
        Args:
            idx: Landmark index
            pixel_coords: If True, return pixel coordinates, else normalized
            
        Returns:
            Tuple of (x, y) coordinates or None if not visible
        """
        if self.visibility[idx] < 0.5:
            return None
            
        x, y = self.landmarks[idx, 0], self.landmarks[idx, 1]
        
        if pixel_coords:
            return (int(x * self.image_width), int(y * self.image_height))
        return (x, y)
    
    def get_center_point(self, idx1: int, idx2: int, pixel_coords: bool = True) -> Optional[Tuple[int, int]]:
        """Get the center point between two landmarks.
        
        Args:
            idx1: First landmark index
            idx2: Second landmark index
            pixel_coords: If True, return pixel coordinates
            
        Returns:
            Center point coordinates or None if either point not visible
        """
        if self.visibility[idx1] < 0.5 or self.visibility[idx2] < 0.5:
            return None
            
        x = (self.landmarks[idx1, 0] + self.landmarks[idx2, 0]) / 2
        y = (self.landmarks[idx1, 1] + self.landmarks[idx2, 1]) / 2
        
        if pixel_coords:
            return (int(x * self.image_width), int(y * self.image_height))
        return (x, y)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Get bounding box of visible landmarks.
        
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates
        """
        visible_mask = self.visibility >= 0.5
        if not np.any(visible_mask):
            return (0, 0, 0, 0)
            
        visible_landmarks = self.landmarks[visible_mask]
        x_min = int(np.min(visible_landmarks[:, 0]) * self.image_width)
        y_min = int(np.min(visible_landmarks[:, 1]) * self.image_height)
        x_max = int(np.max(visible_landmarks[:, 0]) * self.image_width)
        y_max = int(np.max(visible_landmarks[:, 1]) * self.image_height)
        
        return (x_min, y_min, x_max, y_max)
    
    def get_torso_center(self) -> Optional[Tuple[int, int]]:
        """Get the center of the torso for character tracking."""
        # Average of shoulders and hips
        visible_count = 0
        x_sum, y_sum = 0, 0
        
        for idx in [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP]:
            if self.visibility[idx] >= 0.5:
                x_sum += self.landmarks[idx, 0]
                y_sum += self.landmarks[idx, 1]
                visible_count += 1
        
        if visible_count == 0:
            return None
            
        x = x_sum / visible_count
        y = y_sum / visible_count
        return (int(x * self.image_width), int(y * self.image_height))


class PoseDetector:
    """Real-time pose detector using MediaPipe Pose."""
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
    ):
        """Initialize the pose detector.
        
        Args:
            model_complexity: Model complexity (0, 1, or 2). Higher = more accurate but slower.
            min_detection_confidence: Minimum confidence for pose detection.
            min_tracking_confidence: Minimum confidence for pose tracking.
            enable_segmentation: Whether to enable segmentation mask output.
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is required for pose detection. "
                "Install it with: pip install mediapipe"
            )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        # For multi-person detection
        self.mp_holistic = None  # Reserved for future multi-person support
        
    def detect(self, frame: np.ndarray) -> List[PoseKeypoints]:
        """Detect poses in a frame.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            List of PoseKeypoints objects for each detected person
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = frame[:, :, ::-1]
        
        results = self.pose.process(rgb_frame)
        
        poses = []
        if results.pose_landmarks:
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ])
            visibility = np.array([
                lm.visibility for lm in results.pose_landmarks.landmark
            ])
            
            pose = PoseKeypoints(
                landmarks=landmarks,
                visibility=visibility,
                image_shape=(frame.shape[0], frame.shape[1])
            )
            poses.append(pose)
        
        return poses
    
    def close(self):
        """Release resources."""
        self.pose.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class MultiPoseDetector:
    """Multi-person pose detector using MediaPipe Pose with tracking.
    
    Uses a combination of detection and tracking to handle multiple people.
    This is a workaround since MediaPipe Pose only detects one person at a time.
    """
    
    def __init__(
        self,
        max_people: int = 5,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize multi-person detector.
        
        Args:
            max_people: Maximum number of people to track
            model_complexity: Model complexity (0, 1, or 2)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required. Install with: pip install mediapipe")
        
        self.max_people = max_people
        
        # Use Holistic for multi-person (future enhancement)
        # For now, use single-person Pose detector
        self.detector = PoseDetector(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    
    def detect(self, frame: np.ndarray) -> List[PoseKeypoints]:
        """Detect multiple poses in a frame.
        
        Currently delegates to single-person detection.
        Multi-person support can be added using segmentation and ROI detection.
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of detected poses
        """
        # For now, use single-person detection
        # TODO: Implement multi-person detection using:
        # 1. Person detection (e.g., YOLO, SSD)
        # 2. ROI extraction for each person
        # 3. Pose estimation on each ROI
        return self.detector.detect(frame)
    
    def close(self):
        """Release resources."""
        self.detector.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
