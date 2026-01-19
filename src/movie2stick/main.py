"""Command-line interface for Movie2Stick."""

import sys
from typing import Optional

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

from movie2stick.video_processor import VideoProcessor, InputSource


def get_input_source(source: str) -> InputSource:
    """Convert string to InputSource enum."""
    source_map = {
        "screen": InputSource.SCREEN,
        "webcam": InputSource.WEBCAM,
        "file": InputSource.FILE,
    }
    return source_map.get(source.lower(), InputSource.SCREEN)


if CLICK_AVAILABLE:
    @click.command()
    @click.option(
        "--source", "-s",
        type=click.Choice(["screen", "webcam", "file"]),
        default="screen",
        help="Video input source (default: screen)"
    )
    @click.option(
        "--file", "-f",
        type=click.Path(exists=True),
        default=None,
        help="Path to video file (required if source is 'file')"
    )
    @click.option(
        "--webcam", "-w",
        type=int,
        default=0,
        help="Webcam device index (default: 0)"
    )
    @click.option(
        "--width",
        type=int,
        default=1280,
        help="Output width in pixels (default: 1280)"
    )
    @click.option(
        "--height",
        type=int,
        default=720,
        help="Output height in pixels (default: 720)"
    )
    @click.option(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second (default: 30)"
    )
    @click.option(
        "--virtual-camera/--no-virtual-camera",
        default=True,
        help="Enable/disable virtual camera output (default: enabled)"
    )
    @click.option(
        "--preview/--no-preview",
        default=True,
        help="Show preview window (default: enabled)"
    )
    @click.option(
        "--overlay/--no-overlay",
        default=False,
        help="Overlay stickman on original video (default: disabled)"
    )
    @click.option(
        "--stylized-background/--no-stylized-background",
        default=False,
        help="Show stylized edge-detected background (default: disabled)"
    )
    @click.option(
        "--model-complexity",
        type=click.Choice(["0", "1", "2"]),
        default="1",
        help="MediaPipe model complexity: 0=lite, 1=full, 2=heavy (default: 1)"
    )
    @click.option(
        "--region",
        type=str,
        default=None,
        help="Screen region to capture: 'left,top,width,height' (default: full screen)"
    )
    def main(
        source: str,
        file: Optional[str],
        webcam: int,
        width: int,
        height: int,
        fps: int,
        virtual_camera: bool,
        preview: bool,
        overlay: bool,
        stylized_background: bool,
        model_complexity: str,
        region: Optional[str],
    ):
        """Movie2Stick - Real-time video to stickman converter.
        
        Captures video from screen, webcam, or file, converts detected people
        into stickman figures, and outputs as a virtual camera for use in OBS
        or other streaming software.
        
        CONTROLS:
        
        \b
        - Press 'q' or ESC to quit
        - Press 'o' to toggle overlay mode
        - Press 'r' to reset character tracking
        """
        click.echo("=" * 50)
        click.echo("Movie2Stick - Real-time Stickman Video Filter")
        click.echo("=" * 50)
        
        input_source = get_input_source(source)
        
        if input_source == InputSource.FILE and not file:
            click.echo("Error: --file is required when source is 'file'", err=True)
            sys.exit(1)
        
        # Parse region if provided
        screen_region = None
        if region:
            try:
                parts = [int(x.strip()) for x in region.split(",")]
                if len(parts) != 4:
                    raise ValueError("Region must have 4 values")
                screen_region = tuple(parts)
            except ValueError as e:
                click.echo(f"Error parsing region: {e}", err=True)
                sys.exit(1)
        
        click.echo(f"\nConfiguration:")
        click.echo(f"  Input source: {source}")
        if input_source == InputSource.FILE:
            click.echo(f"  File: {file}")
        elif input_source == InputSource.WEBCAM:
            click.echo(f"  Webcam index: {webcam}")
        elif input_source == InputSource.SCREEN:
            if screen_region:
                click.echo(f"  Screen region: {screen_region}")
            else:
                click.echo(f"  Screen region: Full screen")
        click.echo(f"  Output: {width}x{height} @ {fps} FPS")
        click.echo(f"  Virtual camera: {'Enabled' if virtual_camera else 'Disabled'}")
        click.echo(f"  Preview: {'Enabled' if preview else 'Disabled'}")
        click.echo(f"  Overlay mode: {'Enabled' if overlay else 'Disabled'}")
        click.echo(f"  Stylized background: {'Enabled' if stylized_background else 'Disabled'}")
        click.echo(f"  Model complexity: {model_complexity}")
        
        click.echo("\nStarting... Press 'q' or ESC to quit.\n")
        
        try:
            processor = VideoProcessor(
                input_source=input_source,
                input_path=file,
                webcam_index=webcam,
                screen_region=screen_region,
                output_width=width,
                output_height=height,
                output_fps=fps,
                enable_virtual_camera=virtual_camera,
                show_preview=preview,
                overlay_mode=overlay,
                stylized_background=stylized_background,
                model_complexity=int(model_complexity),
            )
            processor.run()
        except ImportError as e:
            click.echo(f"\nError: Missing dependency - {e}", err=True)
            click.echo("\nPlease install all requirements:", err=True)
            click.echo("  pip install -r requirements.txt", err=True)
            sys.exit(1)
        except RuntimeError as e:
            click.echo(f"\nError: {e}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"\nUnexpected error: {e}", err=True)
            raise

else:
    def main():
        """Fallback main function without Click."""
        print("=" * 50)
        print("Movie2Stick - Real-time Stickman Video Filter")
        print("=" * 50)
        print("\nNote: Install 'click' for full CLI support:")
        print("  pip install click")
        print("\nStarting with default settings...")
        print("Press 'q' or ESC to quit.\n")
        
        try:
            processor = VideoProcessor(
                input_source=InputSource.SCREEN,
                output_width=1280,
                output_height=720,
                output_fps=30,
                enable_virtual_camera=True,
                show_preview=True,
                overlay_mode=False,
                model_complexity=1,
            )
            processor.run()
        except ImportError as e:
            print(f"\nError: Missing dependency - {e}")
            print("\nPlease install all requirements:")
            print("  pip install -r requirements.txt")
            sys.exit(1)


if __name__ == "__main__":
    main()
