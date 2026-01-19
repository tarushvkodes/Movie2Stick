"""Video processing module for input/output handling."""

from typing import Optional, Tuple, Generator, Callable
from enum import Enum
import time
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False

from movie2stick.pose_detector import PoseDetector, MultiPoseDetector
from movie2stick.stickman_renderer import StickmanRenderer, StickmanStyle
from movie2stick.character_tracker import CharacterTracker


class InputSource(Enum):
    """Video input source types."""
    WEBCAM = "webcam"
    SCREEN = "screen"
    FILE = "file"
    WINDOW = "window"


class VideoProcessor:
    """Main video processing pipeline.
    
    Coordinates:
    - Input capture (webcam, screen, file)
    - Pose detection
    - Character tracking
    - Stickman rendering
    - Virtual camera output
    """
    
    def __init__(
        self,
        input_source: InputSource = InputSource.SCREEN,
        input_path: Optional[str] = None,
        webcam_index: int = 0,
        screen_region: Optional[Tuple[int, int, int, int]] = None,
        output_width: int = 1280,
        output_height: int = 720,
        output_fps: int = 30,
        enable_virtual_camera: bool = True,
        show_preview: bool = True,
        overlay_mode: bool = False,
        model_complexity: int = 1,
    ):
        """Initialize the video processor.
        
        Args:
            input_source: Type of video input
            input_path: Path to video file (if input_source is FILE)
            webcam_index: Webcam device index (if input_source is WEBCAM)
            screen_region: (left, top, width, height) for screen capture
            output_width: Width of output video
            output_height: Height of output video
            output_fps: Target frames per second
            enable_virtual_camera: Whether to output as virtual camera
            show_preview: Whether to show preview window
            overlay_mode: If True, overlay stickman on original; if False, stickman only
            model_complexity: MediaPipe model complexity (0-2)
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required. Install with: pip install opencv-python")
        
        self.input_source = input_source
        self.input_path = input_path
        self.webcam_index = webcam_index
        self.screen_region = screen_region
        self.output_width = output_width
        self.output_height = output_height
        self.output_fps = output_fps
        self.enable_virtual_camera = enable_virtual_camera
        self.show_preview = show_preview
        self.overlay_mode = overlay_mode
        self.model_complexity = model_complexity
        
        # Components (initialized in start())
        self.pose_detector: Optional[PoseDetector] = None
        self.renderer: Optional[StickmanRenderer] = None
        self.tracker: Optional[CharacterTracker] = None
        self.virtual_camera = None
        self.capture = None
        self.screen_capture = None
        
        # State
        self.is_running = False
        self.frame_count = 0
        self.start_time = 0
        
    def start(self):
        """Initialize all components and start processing."""
        # Initialize pose detector
        self.pose_detector = PoseDetector(
            model_complexity=self.model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Initialize renderer
        self.renderer = StickmanRenderer(
            background_color=(0, 0, 0),  # Black background
            default_style=StickmanStyle(),
        )
        
        # Initialize tracker
        self.tracker = CharacterTracker(
            max_distance=200,
            max_frames_missing=30,
        )
        
        # Initialize input source
        self._init_input()
        
        # Initialize virtual camera
        if self.enable_virtual_camera:
            self._init_virtual_camera()
        
        self.is_running = True
        self.start_time = time.time()
        
    def _init_input(self):
        """Initialize video input source."""
        if self.input_source == InputSource.WEBCAM:
            self.capture = cv2.VideoCapture(self.webcam_index)
            if not self.capture.isOpened():
                raise RuntimeError(f"Failed to open webcam {self.webcam_index}")
            
            # Set capture resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.output_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.output_height)
            self.capture.set(cv2.CAP_PROP_FPS, self.output_fps)
            
        elif self.input_source == InputSource.FILE:
            if not self.input_path:
                raise ValueError("input_path is required for FILE input source")
            self.capture = cv2.VideoCapture(self.input_path)
            if not self.capture.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.input_path}")
                
        elif self.input_source == InputSource.SCREEN:
            if not MSS_AVAILABLE:
                raise ImportError("mss is required for screen capture. Install with: pip install mss")
            self.screen_capture = mss.mss()
            
            # Default to primary monitor if no region specified
            if self.screen_region is None:
                monitor = self.screen_capture.monitors[1]  # Primary monitor
                self.screen_region = {
                    "left": monitor["left"],
                    "top": monitor["top"],
                    "width": monitor["width"],
                    "height": monitor["height"],
                }
            else:
                left, top, width, height = self.screen_region
                self.screen_region = {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }
    
    def _init_virtual_camera(self):
        """Initialize virtual camera output."""
        if not PYVIRTUALCAM_AVAILABLE:
            print("Warning: pyvirtualcam not available. Virtual camera disabled.")
            print("Install with: pip install pyvirtualcam")
            print("Also install OBS Virtual Camera or v4l2loopback.")
            self.enable_virtual_camera = False
            return
        
        try:
            # Try to create virtual camera
            self.virtual_camera = pyvirtualcam.Camera(
                width=self.output_width,
                height=self.output_height,
                fps=self.output_fps,
                print_fps=False,
            )
            print(f"Virtual camera started: {self.virtual_camera.device}")
        except Exception as e:
            print(f"Warning: Failed to create virtual camera: {e}")
            print("Make sure OBS Virtual Camera or v4l2loopback is installed.")
            self.enable_virtual_camera = False
    
    def _get_frame(self) -> Optional[np.ndarray]:
        """Get the next frame from the input source."""
        if self.input_source in [InputSource.WEBCAM, InputSource.FILE]:
            ret, frame = self.capture.read()
            if not ret:
                return None
            return frame
            
        elif self.input_source == InputSource.SCREEN:
            screenshot = self.screen_capture.grab(self.screen_region)
            frame = np.array(screenshot)
            # Convert BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        
        return None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the pipeline.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed frame with stickman figures
        """
        # Resize input if needed
        if frame.shape[1] != self.output_width or frame.shape[0] != self.output_height:
            frame = cv2.resize(frame, (self.output_width, self.output_height))
        
        # Detect poses
        poses = self.pose_detector.detect(frame)
        
        # Update tracking
        character_ids = self.tracker.update(poses)
        
        # Get color indices for consistent character colors
        color_indices = [
            self.tracker.get_character_color_index(cid) if cid >= 0 else 0
            for cid in character_ids
        ]
        
        # Render stickman
        if self.overlay_mode:
            output = self.renderer.render_overlay(
                frame,
                poses,
                character_ids=color_indices,
            )
        else:
            output = self.renderer.render(
                frame.shape,
                poses,
                character_ids=color_indices,
            )
        
        self.frame_count += 1
        return output
    
    def run(self, callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None):
        """Main processing loop.
        
        Args:
            callback: Optional callback function called with (input_frame, output_frame)
        """
        self.start()
        
        target_frame_time = 1.0 / self.output_fps
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # Get input frame
                frame = self._get_frame()
                if frame is None:
                    if self.input_source == InputSource.FILE:
                        # End of file
                        break
                    continue
                
                # Process frame
                output = self.process_frame(frame)
                
                # Output to virtual camera
                if self.enable_virtual_camera and self.virtual_camera:
                    # Convert BGR to RGB for pyvirtualcam
                    rgb_output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                    self.virtual_camera.send(rgb_output)
                
                # Show preview
                if self.show_preview:
                    cv2.imshow("Movie2Stick Preview", output)
                    
                    # Handle key events
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        break
                    elif key == ord('o'):  # Toggle overlay mode
                        self.overlay_mode = not self.overlay_mode
                        print(f"Overlay mode: {self.overlay_mode}")
                    elif key == ord('r'):  # Reset tracking
                        self.tracker.reset()
                        print("Tracking reset")
                
                # Call user callback
                if callback:
                    callback(frame, output)
                
                # Frame rate control
                elapsed = time.time() - loop_start
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop processing and release resources."""
        self.is_running = False
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        if self.screen_capture:
            self.screen_capture.close()
            self.screen_capture = None
        
        if self.virtual_camera:
            self.virtual_camera.close()
            self.virtual_camera = None
        
        if self.pose_detector:
            self.pose_detector.close()
            self.pose_detector = None
        
        cv2.destroyAllWindows()
        
        # Print stats
        if self.frame_count > 0 and self.start_time > 0:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed
            print(f"Processed {self.frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class FrameProcessor:
    """Simplified frame processor for integration with other applications."""
    
    def __init__(
        self,
        model_complexity: int = 1,
        overlay_mode: bool = False,
    ):
        """Initialize the frame processor.
        
        Args:
            model_complexity: MediaPipe model complexity (0-2)
            overlay_mode: If True, overlay stickman on original
        """
        self.pose_detector = PoseDetector(model_complexity=model_complexity)
        self.renderer = StickmanRenderer()
        self.tracker = CharacterTracker()
        self.overlay_mode = overlay_mode
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed frame
        """
        poses = self.pose_detector.detect(frame)
        character_ids = self.tracker.update(poses)
        
        color_indices = [
            self.tracker.get_character_color_index(cid) if cid >= 0 else 0
            for cid in character_ids
        ]
        
        if self.overlay_mode:
            return self.renderer.render_overlay(frame, poses, character_ids=color_indices)
        else:
            return self.renderer.render(frame.shape, poses, character_ids=color_indices)
    
    def close(self):
        """Release resources."""
        self.pose_detector.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
