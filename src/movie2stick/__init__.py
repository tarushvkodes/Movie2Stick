"""Movie2Stick - Real-time video to stickman converter."""

__version__ = "1.0.0"
__author__ = "Tarushv Kosgi"

from movie2stick.pose_detector import PoseDetector
from movie2stick.stickman_renderer import StickmanRenderer
from movie2stick.character_tracker import CharacterTracker
from movie2stick.video_processor import VideoProcessor

__all__ = [
    "PoseDetector",
    "StickmanRenderer", 
    "CharacterTracker",
    "VideoProcessor",
]
