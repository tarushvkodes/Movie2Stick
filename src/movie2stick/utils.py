"""Utility functions for Movie2Stick."""

from typing import Tuple, Optional, List
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def resize_with_aspect_ratio(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    inter: int = cv2.INTER_LINEAR if CV2_AVAILABLE else 0,
) -> np.ndarray:
    """Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width (if None, calculated from height)
        height: Target height (if None, calculated from width)
        inter: Interpolation method
        
    Returns:
        Resized image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required")
    
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    
    return cv2.resize(image, dim, interpolation=inter)


def letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize image with letterboxing to fit target dimensions.
    
    Args:
        image: Input image
        target_width: Target width
        target_height: Target height
        color: Background color for letterbox bars
        
    Returns:
        Letterboxed image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required")
    
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_width / w, target_height / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create output image with background color
    output = np.full((target_height, target_width, 3), color, dtype=np.uint8)
    
    # Calculate position to paste resized image
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    
    # Paste resized image
    output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return output


def calculate_fps(timestamps: List[float], window: int = 30) -> float:
    """Calculate FPS from a list of frame timestamps.
    
    Args:
        timestamps: List of frame timestamps in seconds
        window: Number of recent frames to consider
        
    Returns:
        Frames per second
    """
    if len(timestamps) < 2:
        return 0.0
    
    recent = timestamps[-window:]
    if len(recent) < 2:
        return 0.0
    
    duration = recent[-1] - recent[0]
    if duration <= 0:
        return 0.0
    
    return (len(recent) - 1) / duration


def draw_fps(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 1.0,
) -> np.ndarray:
    """Draw FPS counter on image.
    
    Args:
        image: Input image
        fps: Frames per second value
        position: Position for FPS text
        color: Text color
        font_scale: Font scale
        
    Returns:
        Image with FPS overlay
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required")
    
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )
    return image


def create_color_swatch(
    colors: List[Tuple[int, int, int]],
    swatch_size: int = 50,
) -> np.ndarray:
    """Create a color swatch image for previewing character colors.
    
    Args:
        colors: List of BGR colors
        swatch_size: Size of each color square
        
    Returns:
        Image with color swatches
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required")
    
    n_colors = len(colors)
    width = swatch_size * n_colors
    height = swatch_size
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        x1 = i * swatch_size
        x2 = (i + 1) * swatch_size
        image[:, x1:x2] = color
    
    return image


def get_system_info() -> dict:
    """Get system information for debugging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    
    # Check for GPU support
    try:
        import cv2
        info["opencv_version"] = cv2.__version__
        info["opencv_build"] = cv2.getBuildInformation()[:200] + "..."
    except ImportError:
        info["opencv_version"] = "Not installed"
    
    try:
        import mediapipe
        info["mediapipe_version"] = mediapipe.__version__
    except ImportError:
        info["mediapipe_version"] = "Not installed"
    
    try:
        import pyvirtualcam
        info["pyvirtualcam_version"] = pyvirtualcam.__version__
    except ImportError:
        info["pyvirtualcam_version"] = "Not installed"
    
    return info
