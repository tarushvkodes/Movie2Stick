"""Stickman rendering module."""

from typing import List, Optional, Tuple, Dict
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from movie2stick.pose_detector import PoseKeypoints


class StickmanStyle:
    """Style configuration for stickman rendering."""
    
    # Predefined color palettes for different characters (BGR format)
    COLOR_PALETTES = [
        {"body": (0, 255, 0), "head": (0, 200, 0), "joints": (0, 180, 0)},      # Green
        {"body": (255, 0, 0), "head": (200, 0, 0), "joints": (180, 0, 0)},      # Blue
        {"body": (0, 0, 255), "head": (0, 0, 200), "joints": (0, 0, 180)},      # Red
        {"body": (0, 255, 255), "head": (0, 200, 200), "joints": (0, 180, 180)}, # Yellow
        {"body": (255, 0, 255), "head": (200, 0, 200), "joints": (180, 0, 180)}, # Magenta
        {"body": (255, 255, 0), "head": (200, 200, 0), "joints": (180, 180, 0)}, # Cyan
        {"body": (128, 0, 255), "head": (100, 0, 200), "joints": (80, 0, 180)},  # Orange
        {"body": (255, 128, 0), "head": (200, 100, 0), "joints": (180, 80, 0)},  # Purple
    ]
    
    def __init__(
        self,
        body_color: Tuple[int, int, int] = (0, 255, 0),
        head_color: Tuple[int, int, int] = (0, 255, 0),
        joint_color: Tuple[int, int, int] = (0, 200, 0),
        line_thickness: int = 3,
        joint_radius: int = 5,
        head_radius_factor: float = 0.15,
        draw_head_circle: bool = True,
        draw_joints: bool = True,
    ):
        """Initialize stickman style.
        
        Args:
            body_color: BGR color for body lines
            head_color: BGR color for head circle
            joint_color: BGR color for joint circles
            line_thickness: Thickness of lines in pixels
            joint_radius: Radius of joint circles in pixels
            head_radius_factor: Head circle radius as fraction of torso height
            draw_head_circle: Whether to draw a circle for the head
            draw_joints: Whether to draw circles at joints
        """
        self.body_color = body_color
        self.head_color = head_color
        self.joint_color = joint_color
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
        self.head_radius_factor = head_radius_factor
        self.draw_head_circle = draw_head_circle
        self.draw_joints = draw_joints
    
    @classmethod
    def from_palette(cls, palette_index: int, **kwargs) -> "StickmanStyle":
        """Create a style from a predefined palette.
        
        Args:
            palette_index: Index of the color palette to use
            **kwargs: Additional style parameters to override
            
        Returns:
            StickmanStyle instance
        """
        palette = cls.COLOR_PALETTES[palette_index % len(cls.COLOR_PALETTES)]
        return cls(
            body_color=palette["body"],
            head_color=palette["head"],
            joint_color=palette["joints"],
            **kwargs
        )


class StickmanRenderer:
    """Renders stickman figures from pose keypoints."""
    
    def __init__(
        self,
        background_color: Tuple[int, int, int] = (0, 0, 0),
        default_style: Optional[StickmanStyle] = None,
        draw_all_connections: bool = False,
    ):
        """Initialize the renderer.
        
        Args:
            background_color: BGR color for the background
            default_style: Default style for rendering (if not specified per-character)
            draw_all_connections: If True, draw all pose connections; if False, simplified stickman
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required. Install with: pip install opencv-python")
        
        self.background_color = background_color
        self.default_style = default_style or StickmanStyle()
        self.draw_all_connections = draw_all_connections
        
        # Simple stickman connections (more readable output)
        self.stickman_connections = [
            # Spine (head to torso center)
            ("head", "neck"),
            ("neck", "torso"),
            ("torso", "pelvis"),
            # Arms
            (PoseKeypoints.LEFT_SHOULDER, PoseKeypoints.LEFT_ELBOW),
            (PoseKeypoints.LEFT_ELBOW, PoseKeypoints.LEFT_WRIST),
            (PoseKeypoints.RIGHT_SHOULDER, PoseKeypoints.RIGHT_ELBOW),
            (PoseKeypoints.RIGHT_ELBOW, PoseKeypoints.RIGHT_WRIST),
            # Legs
            (PoseKeypoints.LEFT_HIP, PoseKeypoints.LEFT_KNEE),
            (PoseKeypoints.LEFT_KNEE, PoseKeypoints.LEFT_ANKLE),
            (PoseKeypoints.RIGHT_HIP, PoseKeypoints.RIGHT_KNEE),
            (PoseKeypoints.RIGHT_KNEE, PoseKeypoints.RIGHT_ANKLE),
        ]
    
    def render(
        self,
        frame_shape: Tuple[int, int, int],
        poses: List[PoseKeypoints],
        styles: Optional[List[StickmanStyle]] = None,
        character_ids: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Render stickman figures for all detected poses.
        
        Args:
            frame_shape: Shape of the output frame (height, width, channels)
            poses: List of PoseKeypoints objects
            styles: Optional list of styles for each pose (uses default if not provided)
            character_ids: Optional list of character IDs for consistent styling
            
        Returns:
            Rendered frame with stickman figures
        """
        # Create blank frame with background color
        output = np.full(frame_shape, self.background_color, dtype=np.uint8)
        
        for i, pose in enumerate(poses):
            # Determine style
            if styles and i < len(styles):
                style = styles[i]
            elif character_ids and i < len(character_ids):
                style = StickmanStyle.from_palette(character_ids[i])
            else:
                style = self.default_style
            
            self._draw_stickman(output, pose, style)
        
        return output
    
    def render_overlay(
        self,
        frame: np.ndarray,
        poses: List[PoseKeypoints],
        styles: Optional[List[StickmanStyle]] = None,
        character_ids: Optional[List[int]] = None,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Render stickman figures overlaid on the original frame.
        
        Args:
            frame: Original video frame
            poses: List of PoseKeypoints objects
            styles: Optional list of styles for each pose
            character_ids: Optional list of character IDs
            alpha: Opacity of the overlay (0.0 to 1.0)
            
        Returns:
            Frame with stickman overlay
        """
        output = frame.copy()
        
        for i, pose in enumerate(poses):
            if styles and i < len(styles):
                style = styles[i]
            elif character_ids and i < len(character_ids):
                style = StickmanStyle.from_palette(character_ids[i])
            else:
                style = self.default_style
            
            if alpha < 1.0:
                overlay = output.copy()
                self._draw_stickman(overlay, pose, style)
                output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
            else:
                self._draw_stickman(output, pose, style)
        
        return output
    
    def _draw_stickman(
        self,
        frame: np.ndarray,
        pose: PoseKeypoints,
        style: StickmanStyle,
    ):
        """Draw a single stickman figure on the frame.
        
        Args:
            frame: Frame to draw on (modified in-place)
            pose: Pose keypoints
            style: Style for rendering
        """
        # Calculate virtual points
        neck = pose.get_center_point(PoseKeypoints.LEFT_SHOULDER, PoseKeypoints.RIGHT_SHOULDER)
        pelvis = pose.get_center_point(PoseKeypoints.LEFT_HIP, PoseKeypoints.RIGHT_HIP)
        head = pose.get_point(PoseKeypoints.NOSE)
        
        # Calculate torso center for body scale reference
        if neck and pelvis:
            torso_height = abs(pelvis[1] - neck[1])
        else:
            torso_height = 100  # Default
        
        # Draw body lines
        # Spine: head -> neck -> pelvis
        if head and neck:
            cv2.line(frame, head, neck, style.body_color, style.line_thickness)
        if neck and pelvis:
            cv2.line(frame, neck, pelvis, style.body_color, style.line_thickness)
        
        # Shoulders line
        left_shoulder = pose.get_point(PoseKeypoints.LEFT_SHOULDER)
        right_shoulder = pose.get_point(PoseKeypoints.RIGHT_SHOULDER)
        if left_shoulder and right_shoulder:
            cv2.line(frame, left_shoulder, right_shoulder, style.body_color, style.line_thickness)
        
        # Hips line
        left_hip = pose.get_point(PoseKeypoints.LEFT_HIP)
        right_hip = pose.get_point(PoseKeypoints.RIGHT_HIP)
        if left_hip and right_hip:
            cv2.line(frame, left_hip, right_hip, style.body_color, style.line_thickness)
        
        # Arms
        limb_connections = [
            (PoseKeypoints.LEFT_SHOULDER, PoseKeypoints.LEFT_ELBOW),
            (PoseKeypoints.LEFT_ELBOW, PoseKeypoints.LEFT_WRIST),
            (PoseKeypoints.RIGHT_SHOULDER, PoseKeypoints.RIGHT_ELBOW),
            (PoseKeypoints.RIGHT_ELBOW, PoseKeypoints.RIGHT_WRIST),
            # Legs
            (PoseKeypoints.LEFT_HIP, PoseKeypoints.LEFT_KNEE),
            (PoseKeypoints.LEFT_KNEE, PoseKeypoints.LEFT_ANKLE),
            (PoseKeypoints.RIGHT_HIP, PoseKeypoints.RIGHT_KNEE),
            (PoseKeypoints.RIGHT_KNEE, PoseKeypoints.RIGHT_ANKLE),
        ]
        
        for start_idx, end_idx in limb_connections:
            start_point = pose.get_point(start_idx)
            end_point = pose.get_point(end_idx)
            if start_point and end_point:
                cv2.line(frame, start_point, end_point, style.body_color, style.line_thickness)
        
        # Draw head circle
        if style.draw_head_circle and head:
            head_radius = max(int(torso_height * style.head_radius_factor), 10)
            cv2.circle(frame, head, head_radius, style.head_color, style.line_thickness)
        
        # Draw joint circles
        if style.draw_joints:
            joint_indices = [
                PoseKeypoints.LEFT_SHOULDER, PoseKeypoints.RIGHT_SHOULDER,
                PoseKeypoints.LEFT_ELBOW, PoseKeypoints.RIGHT_ELBOW,
                PoseKeypoints.LEFT_WRIST, PoseKeypoints.RIGHT_WRIST,
                PoseKeypoints.LEFT_HIP, PoseKeypoints.RIGHT_HIP,
                PoseKeypoints.LEFT_KNEE, PoseKeypoints.RIGHT_KNEE,
                PoseKeypoints.LEFT_ANKLE, PoseKeypoints.RIGHT_ANKLE,
            ]
            
            for idx in joint_indices:
                point = pose.get_point(idx)
                if point:
                    cv2.circle(frame, point, style.joint_radius, style.joint_color, -1)
