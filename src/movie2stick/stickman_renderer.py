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
        eye_color: Tuple[int, int, int] = (255, 255, 255),
        mouth_color: Tuple[int, int, int] = (255, 255, 255),
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
        self.eye_color = eye_color
        self.mouth_color = mouth_color
    
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
        background_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render stickman figures for all detected poses.
        
        Args:
            frame_shape: Shape of the output frame (height, width, channels)
            poses: List of PoseKeypoints objects
            styles: Optional list of styles for each pose (uses default if not provided)
            character_ids: Optional list of character IDs for consistent styling
            background_image: Optional image to use as background
            
        Returns:
            Rendered frame with stickman figures
        """
        # Create frame
        if background_image is not None:
            output = background_image.copy()
            # Ensure it matches target shape if needed (though usually caller handles this)
            if output.shape != frame_shape:
                output = cv2.resize(output, (frame_shape[1], frame_shape[0]))
        else:
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
        # Draw head circle and face
        if style.draw_head_circle and head:
            head_radius = max(int(torso_height * style.head_radius_factor), 10)
            # Draw head background (filled)
            cv2.circle(frame, head, head_radius, style.head_color, -1)
            # Draw head outline
            cv2.circle(frame, head, head_radius, style.body_color, style.line_thickness)
            
            # Draw face features if available
            if hasattr(pose, 'face_landmarks') and pose.face_landmarks is not None:
                self._draw_face_features(frame, pose, style, head, head_radius)
        
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

    def _draw_face_features(self, frame, pose, style, head_center, head_radius):
        """Draw eyes and mouth based on face landmarks."""
        # Map normalized face landmarks to pixel coordinates relative to the head position
        # We use a simplified mapping assuming the face is centered on the nose keypoint
        
        # Landmarks indices (MediaPipe Face Mesh 468)
        # Left Eye: 33 (inner), 133 (outer), 159 (top), 145 (bottom)
        # Right Eye: 362 (inner), 263 (outer), 386 (top), 374 (bottom)
        # Mouth: 61 (left), 291 (right), 13 (upper), 14 (lower)
        
        # Scaling factor to fit face landmarks into stickman head
        # We can't map 1:1 because stickman head might be different size/orientation
        # So we estimate relative positions
        
        if pose.face_blendshapes:
            # unique blendshapes usage
            pass
            
        # Simplified drawing:
        # Calculate eye positions relative to nose (index 1 of face mesh usually)
        # But face_landmarks[1] is nose tip.
        
        landmarks = pose.face_landmarks
        h, w = frame.shape[:2]
        
        # Safe access helper
        def get_lm(idx):
             if idx < len(landmarks):
                 return landmarks[idx]
             return None

        # Nose tip
        nose = get_lm(1)
        if nose is None: return

        # Calculate bounding box of face to normalize
        # Or just project directly if aligned?
        # Let's project directly but scale to head_radius
        
        # Face bounding box approximation
        # We need to scale the face landmarks to fit inside our drawn head circle
        # The nose (head_center) matches landmarks[1]
        
        # Approx face scale from landmarks (e.g., eye distance)
        left_eye_outer = get_lm(33)
        right_eye_outer = get_lm(263)
        
        if left_eye_outer is None or right_eye_outer is None: return
        
        # Real distance
        eye_dist = np.linalg.norm(np.array([left_eye_outer.x, left_eye_outer.y]) - np.array([right_eye_outer.x, right_eye_outer.y]))
        
        # If eye_dist is 0 (impossible), skip
        if eye_dist < 0.001: return
        
        # Desired eye distance in stickman head (e.g., 40% of diameter)
        target_eye_dist = head_radius * 0.8
        
        scale = target_eye_dist / eye_dist
        
        # Transform function
        def transform(lm):
            # Center relative to nose
            rel_x = lm.x - nose.x
            rel_y = lm.y - nose.y
            
            # Scale
            # Correct aspect ratio
            scaled_x = rel_x * scale #* (w/h if normalized? No, x/y are normalized 0-1)
            # wait, x is * width, y is * height.
            # We need to handle aspect ratio
            
            # Convert to pixels relative to head center
            px = int(head_center[0] + rel_x * scale * w) # Rough approx
            # Better:
            # We want isotropic scaling in pixel space.
            # Convert lm to pixels first
            lm_px_x = lm.x * w
            lm_px_y = lm.y * h
            nose_px_x = nose.x * w
            nose_px_y = nose.y * h
            
            dx = lm_px_x - nose_px_x
            dy = lm_px_y - nose_px_y
            
            # Scale factor based on pixel distance
            pixel_eye_dist = eye_dist * w # approx
            scale_factor = target_eye_dist / pixel_eye_dist
            
            return (int(head_center[0] + dx * scale_factor), int(head_center[1] + dy * scale_factor))

        # Draw Left Eye
        # simple circle or ellipse based on open/closed
        l_eye_top = get_lm(159)
        l_eye_bot = get_lm(145)
        if l_eye_top and l_eye_bot:
             l_top = transform(l_eye_top)
             l_bot = transform(l_eye_bot)
             l_height = np.linalg.norm(np.array(l_top) - np.array(l_bot))
             l_center = ((l_top[0]+l_bot[0])//2, (l_top[1]+l_bot[1])//2)
             
             # If eye is open
             if l_height > 2:
                 cv2.circle(frame, l_center, max(int(head_radius * 0.15), 2), style.eye_color, -1)
             else:
                 cv2.line(frame, l_top, l_bot, style.eye_color, 2)
        
        # Draw Right Eye
        r_eye_top = get_lm(386)
        r_eye_bot = get_lm(374)
        if r_eye_top and r_eye_bot:
             r_top = transform(r_eye_top)
             r_bot = transform(r_eye_bot)
             r_height = np.linalg.norm(np.array(r_top) - np.array(r_bot))
             r_center = ((r_top[0]+r_bot[0])//2, (r_top[1]+r_bot[1])//2)
             
             if r_height > 2:
                 cv2.circle(frame, r_center, max(int(head_radius * 0.15), 2), style.eye_color, -1)
             else:
                 cv2.line(frame, r_top, r_bot, style.eye_color, 2)

        # Draw Mouth
        # Lips: 61, 146, 291, 375 (outer loop)
        mouth_indices = [61, 81, 178, 87, 14, 317, 402, 311, 291] # Lower lip line approx
        mouth_pts = []
        for idx in mouth_indices:
             lm = get_lm(idx)
             if lm:
                 mouth_pts.append(transform(lm))
        
        if len(mouth_pts) > 1:
            # Draw curve
            pts = np.array(mouth_pts, np.int32)
            cv2.polylines(frame, [pts], False, style.mouth_color, 2)

