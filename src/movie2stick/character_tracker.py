"""Character tracking module for consistent stickman identification."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

from movie2stick.pose_detector import PoseKeypoints


@dataclass
class TrackedCharacter:
    """Represents a tracked character across frames."""
    
    character_id: int
    last_position: Tuple[int, int]
    last_seen_frame: int
    color_index: int
    positions_history: List[Tuple[int, int]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    def update(self, position: Tuple[int, int], frame_number: int):
        """Update character position and calculate velocity."""
        if self.last_position:
            dx = position[0] - self.last_position[0]
            dy = position[1] - self.last_position[1]
            dt = frame_number - self.last_seen_frame
            if dt > 0:
                self.velocity = (dx / dt, dy / dt)
        
        self.last_position = position
        self.last_seen_frame = frame_number
        self.positions_history.append(position)
        
        # Keep only recent history (last 30 frames)
        if len(self.positions_history) > 30:
            self.positions_history.pop(0)
    
    def predict_position(self, frame_number: int) -> Tuple[int, int]:
        """Predict position based on velocity."""
        dt = frame_number - self.last_seen_frame
        predicted_x = int(self.last_position[0] + self.velocity[0] * dt)
        predicted_y = int(self.last_position[1] + self.velocity[1] * dt)
        return (predicted_x, predicted_y)


class CharacterTracker:
    """Tracks characters across frames for consistent identification.
    
    Uses position-based tracking to maintain character identity across frames.
    Each tracked character gets a consistent color/style for rendering.
    """
    
    def __init__(
        self,
        max_distance: int = 200,
        max_frames_missing: int = 30,
        max_characters: int = 10,
    ):
        """Initialize the character tracker.
        
        Args:
            max_distance: Maximum distance (pixels) for matching detections to tracks
            max_frames_missing: Number of frames before a track is removed
            max_characters: Maximum number of characters to track
        """
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        self.max_characters = max_characters
        
        self.characters: Dict[int, TrackedCharacter] = {}
        self.next_character_id = 0
        self.frame_count = 0
        
        # Available color indices (cycle through palette)
        self.available_colors: List[int] = list(range(8))
        self.used_colors: Dict[int, int] = {}  # character_id -> color_index
    
    def update(self, poses: List[PoseKeypoints]) -> List[int]:
        """Update tracking with new pose detections.
        
        Args:
            poses: List of detected poses in the current frame
            
        Returns:
            List of character IDs corresponding to each pose
        """
        self.frame_count += 1
        
        # Get positions for all detected poses
        detected_positions = []
        for pose in poses:
            center = pose.get_torso_center()
            if center:
                detected_positions.append(center)
            else:
                # Fallback to bounding box center
                bbox = pose.get_bounding_box()
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    detected_positions.append(center)
                else:
                    detected_positions.append(None)
        
        # Match detections to existing tracks
        character_ids = self._match_detections(detected_positions)
        
        # Remove stale tracks
        self._cleanup_stale_tracks()
        
        return character_ids
    
    def _match_detections(self, positions: List[Optional[Tuple[int, int]]]) -> List[int]:
        """Match detected positions to existing character tracks.
        
        Uses Hungarian algorithm-like greedy matching based on distance.
        
        Args:
            positions: List of detected positions (may contain None)
            
        Returns:
            List of character IDs for each position
        """
        character_ids = [-1] * len(positions)
        
        if not positions:
            return character_ids
        
        # Calculate distance matrix
        valid_positions = [(i, pos) for i, pos in enumerate(positions) if pos is not None]
        
        if not valid_positions:
            return character_ids
        
        # Get active character tracks
        active_characters = list(self.characters.values())
        
        if not active_characters:
            # No existing tracks, create new ones
            for i, pos in valid_positions:
                char_id = self._create_character(pos)
                character_ids[i] = char_id
            return character_ids
        
        # Calculate distances from each detection to each track
        distances = np.zeros((len(valid_positions), len(active_characters)))
        
        for i, (det_idx, pos) in enumerate(valid_positions):
            for j, char in enumerate(active_characters):
                predicted_pos = char.predict_position(self.frame_count)
                dist = np.sqrt(
                    (pos[0] - predicted_pos[0]) ** 2 + 
                    (pos[1] - predicted_pos[1]) ** 2
                )
                distances[i, j] = dist
        
        # Greedy matching (could use Hungarian algorithm for optimality)
        matched_chars = set()
        matched_dets = set()
        
        # Sort all pairs by distance
        pairs = []
        for i in range(len(valid_positions)):
            for j in range(len(active_characters)):
                pairs.append((distances[i, j], i, j))
        pairs.sort(key=lambda x: x[0])
        
        # Match greedily
        for dist, det_i, char_j in pairs:
            if det_i in matched_dets or char_j in matched_chars:
                continue
            
            if dist <= self.max_distance:
                det_idx, pos = valid_positions[det_i]
                char = active_characters[char_j]
                char.update(pos, self.frame_count)
                character_ids[det_idx] = char.character_id
                matched_chars.add(char_j)
                matched_dets.add(det_i)
        
        # Create new tracks for unmatched detections
        for i, (det_idx, pos) in enumerate(valid_positions):
            if i not in matched_dets:
                if len(self.characters) < self.max_characters:
                    char_id = self._create_character(pos)
                    character_ids[det_idx] = char_id
        
        return character_ids
    
    def _create_character(self, position: Tuple[int, int]) -> int:
        """Create a new tracked character.
        
        Args:
            position: Initial position
            
        Returns:
            Character ID
        """
        # Assign color index
        if self.available_colors:
            color_index = self.available_colors.pop(0)
        else:
            color_index = self.next_character_id % 8
        
        char_id = self.next_character_id
        self.next_character_id += 1
        
        character = TrackedCharacter(
            character_id=char_id,
            last_position=position,
            last_seen_frame=self.frame_count,
            color_index=color_index,
            positions_history=[position],
        )
        
        self.characters[char_id] = character
        self.used_colors[char_id] = color_index
        
        return char_id
    
    def _cleanup_stale_tracks(self):
        """Remove tracks that haven't been seen for too long."""
        stale_ids = []
        
        for char_id, char in self.characters.items():
            frames_missing = self.frame_count - char.last_seen_frame
            if frames_missing > self.max_frames_missing:
                stale_ids.append(char_id)
        
        for char_id in stale_ids:
            # Return color to available pool
            color_index = self.used_colors.pop(char_id, None)
            if color_index is not None and color_index not in self.available_colors:
                self.available_colors.append(color_index)
            
            del self.characters[char_id]
    
    def get_character_color_index(self, character_id: int) -> int:
        """Get the color index for a character.
        
        Args:
            character_id: Character ID
            
        Returns:
            Color index for the character's palette
        """
        if character_id in self.characters:
            return self.characters[character_id].color_index
        return 0  # Default color
    
    def reset(self):
        """Reset all tracking state."""
        self.characters.clear()
        self.next_character_id = 0
        self.frame_count = 0
        self.available_colors = list(range(8))
        self.used_colors.clear()
