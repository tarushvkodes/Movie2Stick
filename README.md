# Movie2Stick

Real-time video filter that converts videos into uncopyrighted stickman figures for avoiding copyright infringement when making reaction content, or for obfuscating the original content for any other reason.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## Features

- **Real-time Processing**: Converts video to stickman figures in real-time on consumer hardware
- **Multiple Input Sources**: Support for screen capture, webcam, and video files
- **Virtual Camera Output**: Works with OBS and other streaming software via virtual webcam
- **Character Tracking**: Maintains consistent character identity across frames with unique colors
- **Overlay Mode**: Optional overlay of stickman on original video
- **Customizable Appearance**: Configurable stickman style, colors, and line thickness

## How It Works

1. **Pose Detection**: Uses MediaPipe Pose to detect human body keypoints in real-time
2. **Character Tracking**: Tracks individuals across frames to maintain identity
3. **Stickman Rendering**: Draws simplified stickman figures based on detected poses
4. **Virtual Camera Output**: Outputs the processed video as a virtual webcam for use in OBS

## Requirements

- Python 3.8+
- Webcam or screen to capture
- For virtual camera output:
  - **Windows**: OBS Virtual Camera (comes with OBS Studio)
  - **macOS**: OBS Virtual Camera
  - **Linux**: v4l2loopback kernel module

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tarushvkodes/Movie2Stick.git
cd Movie2Stick
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pose Model

On first run, Movie2Stick will automatically download the MediaPipe pose model. If automatic download fails, you can download manually:

```bash
# Create model directory
mkdir -p ~/.cache/movie2stick/models

# Download lite model (faster, less accurate)
curl -o ~/.cache/movie2stick/models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

# OR download full model (recommended)
curl -o ~/.cache/movie2stick/models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task

# OR download heavy model (most accurate, slowest)
curl -o ~/.cache/movie2stick/models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

You can also set a custom model path:
```bash
export MOVIE2STICK_MODEL_PATH=/path/to/your/pose_landmarker.task
```

### 4. Set Up Virtual Camera (Optional)

**Windows/macOS:**
- Install [OBS Studio](https://obsproject.com/)
- Start OBS Virtual Camera at least once

**Linux:**
```bash
# Install v4l2loopback
sudo apt-get install v4l2loopback-dkms

# Load the kernel module
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Movie2Stick" exclusive_caps=1
```

### 5. Install the Package

```bash
pip install -e .
```

## Usage

### Basic Usage

```bash
# Capture screen and output as virtual camera
movie2stick

# Capture from webcam
movie2stick --source webcam

# Process a video file
movie2stick --source file --file /path/to/video.mp4
```

### Command Line Options

```
Usage: movie2stick [OPTIONS]

Options:
  -s, --source [screen|webcam|file]
                                  Video input source (default: screen)
  -f, --file PATH                 Path to video file (required if source is
                                  'file')
  -w, --webcam INTEGER            Webcam device index (default: 0)
  --width INTEGER                 Output width in pixels (default: 1280)
  --height INTEGER                Output height in pixels (default: 720)
  --fps INTEGER                   Target frames per second (default: 30)
  --virtual-camera / --no-virtual-camera
                                  Enable/disable virtual camera output
                                  (default: enabled)
  --preview / --no-preview        Show preview window (default: enabled)
  --overlay / --no-overlay        Overlay stickman on original video
                                  (default: disabled)
  --model-complexity [0|1|2]      MediaPipe model complexity: 0=lite, 1=full,
                                  2=heavy (default: 1)
  --region TEXT                   Screen region to capture:
                                  'left,top,width,height' (default: full screen)
  --help                          Show this message and exit.
```

### Keyboard Controls

While the preview window is active:
- **Q** or **ESC**: Quit the application
- **O**: Toggle overlay mode
- **R**: Reset character tracking

### Using with OBS

1. Start Movie2Stick with virtual camera enabled (default)
2. In OBS, add a new "Video Capture Device" source
3. Select "Movie2Stick" (or "OBS Virtual Camera") as the device
4. The stickman video will appear in your OBS scene

## Python API

For integration with other applications:

```python
from movie2stick import VideoProcessor, InputSource

# Create processor
processor = VideoProcessor(
    input_source=InputSource.SCREEN,
    output_width=1280,
    output_height=720,
    enable_virtual_camera=True,
    overlay_mode=False,
)

# Run processing loop
processor.run()
```

### Frame-by-Frame Processing

```python
import cv2
from movie2stick import FrameProcessor

# Initialize processor
with FrameProcessor(model_complexity=1) as processor:
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame to stickman
        output = processor.process(frame)
        
        cv2.imshow("Stickman", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
```

## Configuration

Create a configuration file at `~/.config/movie2stick/config.yaml`:

```yaml
stickman:
  line_thickness: 3
  joint_radius: 5
  head_radius_factor: 0.15
  draw_head_circle: true
  draw_joints: true
  background_color: [0, 0, 0]

processing:
  model_complexity: 1
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  overlay_mode: false

tracking:
  max_distance: 200
  max_frames_missing: 30
  max_characters: 10

output:
  width: 1280
  height: 720
  fps: 30
  enable_virtual_camera: true
  show_preview: true
```

## Performance Tips

1. **Model Complexity**: Use `--model-complexity 0` for faster but less accurate detection
2. **Resolution**: Lower `--width` and `--height` for better performance
3. **FPS**: Reduce `--fps` if experiencing lag
4. **Preview**: Use `--no-preview` if you only need virtual camera output

## Troubleshooting

### Virtual Camera Not Working

**Windows/macOS:**
- Make sure OBS Studio is installed
- Start OBS and enable Virtual Camera at least once

**Linux:**
- Check if v4l2loopback is loaded: `lsmod | grep v4l2loopback`
- If not loaded, run: `sudo modprobe v4l2loopback`

### Low Frame Rate

- Reduce model complexity: `--model-complexity 0`
- Lower output resolution: `--width 640 --height 480`
- Close other resource-intensive applications

### Pose Detection Issues

- Ensure good lighting
- Ensure the person is fully visible in frame
- Increase detection confidence: may need code modification

## Architecture

```
movie2stick/
├── pose_detector.py      # MediaPipe pose detection
├── stickman_renderer.py  # Stickman drawing logic
├── character_tracker.py  # Character identity tracking
├── video_processor.py    # Main processing pipeline
├── config.py             # Configuration handling
├── utils.py              # Utility functions
└── main.py               # CLI entry point
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose detection
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) for virtual camera output
- [OpenCV](https://opencv.org/) for video processing
