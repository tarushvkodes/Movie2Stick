"""Configuration handling for Movie2Stick."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any
import os
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class StickmanConfig:
    """Configuration for stickman appearance."""
    
    line_thickness: int = 3
    joint_radius: int = 5
    head_radius_factor: float = 0.15
    draw_head_circle: bool = True
    draw_joints: bool = True
    background_color: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    
    model_complexity: int = 1  # 0=lite, 1=full, 2=heavy
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    overlay_mode: bool = False


@dataclass
class TrackingConfig:
    """Configuration for character tracking."""
    
    max_distance: int = 200
    max_frames_missing: int = 30
    max_characters: int = 10


@dataclass
class OutputConfig:
    """Configuration for video output."""
    
    width: int = 1280
    height: int = 720
    fps: int = 30
    enable_virtual_camera: bool = True
    show_preview: bool = True


@dataclass 
class InputConfig:
    """Configuration for video input."""
    
    source: str = "screen"  # screen, webcam, file
    file_path: Optional[str] = None
    webcam_index: int = 0
    screen_region: Optional[Tuple[int, int, int, int]] = None


@dataclass
class Config:
    """Complete configuration for Movie2Stick."""
    
    stickman: StickmanConfig = field(default_factory=StickmanConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    input: InputConfig = field(default_factory=InputConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "stickman": asdict(self.stickman),
            "processing": asdict(self.processing),
            "tracking": asdict(self.tracking),
            "output": asdict(self.output),
            "input": asdict(self.input),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(
            stickman=StickmanConfig(**data.get("stickman", {})),
            processing=ProcessingConfig(**data.get("processing", {})),
            tracking=TrackingConfig(**data.get("tracking", {})),
            output=OutputConfig(**data.get("output", {})),
            input=InputConfig(**data.get("input", {})),
        )
    
    def save(self, path: str):
        """Save configuration to file.
        
        Args:
            path: Path to save configuration file (JSON or YAML)
        """
        data = self.to_dict()
        
        if path.endswith(".yaml") or path.endswith(".yml"):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML files. Install with: pip install pyyaml")
            with open(path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from file.
        
        Args:
            path: Path to configuration file (JSON or YAML)
            
        Returns:
            Config instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required for YAML files. Install with: pip install pyyaml")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data or {})
    
    @classmethod
    def get_default_config_path(cls) -> str:
        """Get the default configuration file path."""
        config_dir = os.path.expanduser("~/.config/movie2stick")
        return os.path.join(config_dir, "config.yaml")
    
    @classmethod
    def load_or_create_default(cls) -> "Config":
        """Load config from default location or create default."""
        default_path = cls.get_default_config_path()
        
        if os.path.exists(default_path):
            return cls.load(default_path)
        
        # Create default config
        config = cls()
        
        # Create config directory if needed
        config_dir = os.path.dirname(default_path)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        
        # Save default config
        config.save(default_path)
        
        return config


# Default configuration instance
DEFAULT_CONFIG = Config()
