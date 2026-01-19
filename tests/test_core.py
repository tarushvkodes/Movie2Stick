"""Tests for Movie2Stick core functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestPoseKeypoints:
    """Tests for PoseKeypoints class."""
    
    def test_pose_keypoints_initialization(self):
        """Test PoseKeypoints can be initialized with valid data."""
        from movie2stick.pose_detector import PoseKeypoints
        
        # Create mock landmarks (33 landmarks with x, y, z)
        landmarks = np.random.rand(33, 3)
        visibility = np.ones(33) * 0.9
        image_shape = (720, 1280)
        
        pose = PoseKeypoints(landmarks, visibility, image_shape)
        
        assert pose.landmarks.shape == (33, 3)
        assert pose.visibility.shape == (33,)
        assert pose.image_height == 720
        assert pose.image_width == 1280
    
    def test_get_point_visible(self):
        """Test getting a visible landmark point."""
        from movie2stick.pose_detector import PoseKeypoints
        
        landmarks = np.zeros((33, 3))
        landmarks[0] = [0.5, 0.5, 0.0]  # Nose at center
        visibility = np.ones(33) * 0.9
        
        pose = PoseKeypoints(landmarks, visibility, (720, 1280))
        
        point = pose.get_point(PoseKeypoints.NOSE)
        assert point == (640, 360)  # Center of 1280x720
    
    def test_get_point_not_visible(self):
        """Test getting a point that is not visible."""
        from movie2stick.pose_detector import PoseKeypoints
        
        landmarks = np.zeros((33, 3))
        landmarks[0] = [0.5, 0.5, 0.0]
        visibility = np.zeros(33)  # All invisible
        
        pose = PoseKeypoints(landmarks, visibility, (720, 1280))
        
        point = pose.get_point(PoseKeypoints.NOSE)
        assert point is None
    
    def test_get_center_point(self):
        """Test getting center point between two landmarks."""
        from movie2stick.pose_detector import PoseKeypoints
        
        landmarks = np.zeros((33, 3))
        landmarks[PoseKeypoints.LEFT_SHOULDER] = [0.4, 0.3, 0.0]
        landmarks[PoseKeypoints.RIGHT_SHOULDER] = [0.6, 0.3, 0.0]
        visibility = np.ones(33) * 0.9
        
        pose = PoseKeypoints(landmarks, visibility, (720, 1280))
        
        center = pose.get_center_point(
            PoseKeypoints.LEFT_SHOULDER,
            PoseKeypoints.RIGHT_SHOULDER
        )
        assert center == (640, 216)  # Midpoint
    
    def test_get_bounding_box(self):
        """Test getting bounding box of visible landmarks."""
        from movie2stick.pose_detector import PoseKeypoints
        
        landmarks = np.zeros((33, 3))
        landmarks[0] = [0.2, 0.1, 0.0]
        landmarks[1] = [0.8, 0.9, 0.0]
        visibility = np.zeros(33)
        visibility[0] = 0.9
        visibility[1] = 0.9
        
        pose = PoseKeypoints(landmarks, visibility, (720, 1280))
        
        bbox = pose.get_bounding_box()
        assert bbox == (256, 72, 1024, 648)
    
    def test_get_torso_center(self):
        """Test getting torso center."""
        from movie2stick.pose_detector import PoseKeypoints
        
        landmarks = np.zeros((33, 3))
        landmarks[PoseKeypoints.LEFT_SHOULDER] = [0.4, 0.2, 0.0]
        landmarks[PoseKeypoints.RIGHT_SHOULDER] = [0.6, 0.2, 0.0]
        landmarks[PoseKeypoints.LEFT_HIP] = [0.4, 0.6, 0.0]
        landmarks[PoseKeypoints.RIGHT_HIP] = [0.6, 0.6, 0.0]
        visibility = np.ones(33) * 0.9
        
        pose = PoseKeypoints(landmarks, visibility, (720, 1280))
        
        center = pose.get_torso_center()
        assert center == (640, 288)  # Average of all 4 points


class TestStickmanStyle:
    """Tests for StickmanStyle class."""
    
    def test_default_style(self):
        """Test default style initialization."""
        from movie2stick.stickman_renderer import StickmanStyle
        
        style = StickmanStyle()
        
        assert style.body_color == (0, 255, 0)
        assert style.line_thickness == 3
        assert style.draw_head_circle is True
        assert style.draw_joints is True
    
    def test_from_palette(self):
        """Test creating style from color palette."""
        from movie2stick.stickman_renderer import StickmanStyle
        
        style = StickmanStyle.from_palette(0)
        assert style.body_color == (0, 255, 0)  # Green
        
        style = StickmanStyle.from_palette(1)
        assert style.body_color == (255, 0, 0)  # Blue
        
        # Test wrapping
        style = StickmanStyle.from_palette(8)
        assert style.body_color == (0, 255, 0)  # Wraps to green


class TestStickmanRenderer:
    """Tests for StickmanRenderer class."""
    
    def test_render_empty_poses(self):
        """Test rendering with no poses."""
        from movie2stick.stickman_renderer import StickmanRenderer
        
        renderer = StickmanRenderer(background_color=(0, 0, 0))
        
        output = renderer.render(
            frame_shape=(720, 1280, 3),
            poses=[],
        )
        
        assert output.shape == (720, 1280, 3)
        assert np.all(output == 0)  # All black
    
    def test_render_with_pose(self):
        """Test rendering with a pose."""
        from movie2stick.stickman_renderer import StickmanRenderer
        from movie2stick.pose_detector import PoseKeypoints
        
        renderer = StickmanRenderer(background_color=(0, 0, 0))
        
        # Create a mock pose
        landmarks = np.zeros((33, 3))
        landmarks[PoseKeypoints.NOSE] = [0.5, 0.2, 0.0]
        landmarks[PoseKeypoints.LEFT_SHOULDER] = [0.4, 0.3, 0.0]
        landmarks[PoseKeypoints.RIGHT_SHOULDER] = [0.6, 0.3, 0.0]
        landmarks[PoseKeypoints.LEFT_HIP] = [0.4, 0.6, 0.0]
        landmarks[PoseKeypoints.RIGHT_HIP] = [0.6, 0.6, 0.0]
        visibility = np.ones(33) * 0.9
        
        pose = PoseKeypoints(landmarks, visibility, (720, 1280))
        
        output = renderer.render(
            frame_shape=(720, 1280, 3),
            poses=[pose],
        )
        
        assert output.shape == (720, 1280, 3)
        # Should have some non-black pixels (stickman drawn)
        assert np.any(output != 0)


class TestCharacterTracker:
    """Tests for CharacterTracker class."""
    
    def test_update_creates_new_character(self):
        """Test that update creates new characters for new detections."""
        from movie2stick.character_tracker import CharacterTracker
        from movie2stick.pose_detector import PoseKeypoints
        
        tracker = CharacterTracker()
        
        # Create a mock pose
        landmarks = np.zeros((33, 3))
        landmarks[PoseKeypoints.LEFT_SHOULDER] = [0.4, 0.3, 0.0]
        landmarks[PoseKeypoints.RIGHT_SHOULDER] = [0.6, 0.3, 0.0]
        landmarks[PoseKeypoints.LEFT_HIP] = [0.4, 0.6, 0.0]
        landmarks[PoseKeypoints.RIGHT_HIP] = [0.6, 0.6, 0.0]
        visibility = np.ones(33) * 0.9
        
        pose = PoseKeypoints(landmarks, visibility, (720, 1280))
        
        character_ids = tracker.update([pose])
        
        assert len(character_ids) == 1
        assert character_ids[0] == 0  # First character ID
        assert len(tracker.characters) == 1
    
    def test_update_tracks_same_character(self):
        """Test that same character is tracked across frames."""
        from movie2stick.character_tracker import CharacterTracker
        from movie2stick.pose_detector import PoseKeypoints
        
        tracker = CharacterTracker(max_distance=100)
        
        def create_pose(x_offset=0.5):
            landmarks = np.zeros((33, 3))
            landmarks[PoseKeypoints.LEFT_SHOULDER] = [x_offset - 0.1, 0.3, 0.0]
            landmarks[PoseKeypoints.RIGHT_SHOULDER] = [x_offset + 0.1, 0.3, 0.0]
            landmarks[PoseKeypoints.LEFT_HIP] = [x_offset - 0.1, 0.6, 0.0]
            landmarks[PoseKeypoints.RIGHT_HIP] = [x_offset + 0.1, 0.6, 0.0]
            visibility = np.ones(33) * 0.9
            return PoseKeypoints(landmarks, visibility, (720, 1280))
        
        # Frame 1
        ids1 = tracker.update([create_pose(0.5)])
        assert ids1[0] == 0
        
        # Frame 2 - slight movement
        ids2 = tracker.update([create_pose(0.51)])
        assert ids2[0] == 0  # Same character
    
    def test_update_tracks_multiple_characters(self):
        """Test tracking multiple characters with different colors."""
        from movie2stick.character_tracker import CharacterTracker
        from movie2stick.pose_detector import PoseKeypoints
        
        tracker = CharacterTracker()
        
        def create_pose(x_offset):
            landmarks = np.zeros((33, 3))
            landmarks[PoseKeypoints.LEFT_SHOULDER] = [x_offset - 0.05, 0.3, 0.0]
            landmarks[PoseKeypoints.RIGHT_SHOULDER] = [x_offset + 0.05, 0.3, 0.0]
            landmarks[PoseKeypoints.LEFT_HIP] = [x_offset - 0.05, 0.6, 0.0]
            landmarks[PoseKeypoints.RIGHT_HIP] = [x_offset + 0.05, 0.6, 0.0]
            visibility = np.ones(33) * 0.9
            return PoseKeypoints(landmarks, visibility, (720, 1280))
        
        # Two people in frame
        poses = [create_pose(0.3), create_pose(0.7)]
        ids = tracker.update(poses)
        
        assert len(ids) == 2
        assert ids[0] != ids[1]  # Different IDs
        
        # Check different colors assigned
        color1 = tracker.get_character_color_index(ids[0])
        color2 = tracker.get_character_color_index(ids[1])
        assert color1 != color2
    
    def test_reset(self):
        """Test reset clears all tracking state."""
        from movie2stick.character_tracker import CharacterTracker
        from movie2stick.pose_detector import PoseKeypoints
        
        tracker = CharacterTracker()
        
        landmarks = np.zeros((33, 3))
        landmarks[PoseKeypoints.LEFT_SHOULDER] = [0.4, 0.3, 0.0]
        landmarks[PoseKeypoints.RIGHT_SHOULDER] = [0.6, 0.3, 0.0]
        landmarks[PoseKeypoints.LEFT_HIP] = [0.4, 0.6, 0.0]
        landmarks[PoseKeypoints.RIGHT_HIP] = [0.6, 0.6, 0.0]
        visibility = np.ones(33) * 0.9
        
        pose = PoseKeypoints(landmarks, visibility, (720, 1280))
        tracker.update([pose])
        
        assert len(tracker.characters) == 1
        
        tracker.reset()
        
        assert len(tracker.characters) == 0
        assert tracker.frame_count == 0
        assert tracker.next_character_id == 0


class TestTrackedCharacter:
    """Tests for TrackedCharacter class."""
    
    def test_update_calculates_velocity(self):
        """Test that velocity is calculated correctly."""
        from movie2stick.character_tracker import TrackedCharacter
        
        char = TrackedCharacter(
            character_id=0,
            last_position=(100, 100),
            last_seen_frame=0,
            color_index=0,
        )
        
        char.update((110, 100), 1)  # Moved 10 pixels right in 1 frame
        
        assert char.velocity == (10.0, 0.0)
    
    def test_predict_position(self):
        """Test position prediction based on velocity."""
        from movie2stick.character_tracker import TrackedCharacter
        
        char = TrackedCharacter(
            character_id=0,
            last_position=(100, 100),
            last_seen_frame=0,
            color_index=0,
            velocity=(5.0, 2.0),
        )
        
        predicted = char.predict_position(2)  # 2 frames later
        
        assert predicted == (110, 104)  # 100 + 5*2, 100 + 2*2


class TestConfig:
    """Tests for configuration handling."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from movie2stick.config import Config
        
        config = Config()
        
        assert config.output.width == 1280
        assert config.output.height == 720
        assert config.processing.model_complexity == 1
        assert config.stickman.line_thickness == 3
    
    def test_to_dict_and_from_dict(self):
        """Test config serialization."""
        from movie2stick.config import Config
        
        config = Config()
        config.output.width = 1920
        
        data = config.to_dict()
        assert data["output"]["width"] == 1920
        
        loaded = Config.from_dict(data)
        assert loaded.output.width == 1920


class TestUtils:
    """Tests for utility functions."""
    
    def test_calculate_fps(self):
        """Test FPS calculation."""
        from movie2stick.utils import calculate_fps
        
        timestamps = [0.0, 0.033, 0.066, 0.1]
        fps = calculate_fps(timestamps)
        
        assert 29 < fps < 31  # ~30 FPS
    
    def test_calculate_fps_empty(self):
        """Test FPS calculation with insufficient data."""
        from movie2stick.utils import calculate_fps
        
        assert calculate_fps([]) == 0.0
        assert calculate_fps([0.0]) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
