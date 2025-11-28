"""
Camera Display Utility for multi-camera visualization.

This module provides a unified way to display multiple camera feeds
in a single window with configurable layout.
"""

import cv2
import numpy as np
from typing import Optional
from functools import cache


@cache
def is_headless() -> bool:
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa
        return False
    except Exception:
        return True


class CameraDisplay:
    """
    Unified camera display utility for multi-camera visualization.

    Combines multiple camera feeds into a single window with
    horizontal, vertical, or grid layout.

    Example:
        display = CameraDisplay(window_name="Cameras", layout="horizontal")

        while recording:
            images = {"image_top": img1, "image_wrist": img2}
            display.show(images)

        display.close()
    """

    def __init__(
        self,
        window_name: str = "Camera View",
        layout: str = "horizontal",
        show_labels: bool = True,
        label_height: int = 25,
        label_font_scale: float = 0.6,
        target_height: Optional[int] = None,
    ):
        """
        Initialize camera display.

        Args:
            window_name: Name of the OpenCV window
            layout: How to arrange cameras - "horizontal", "vertical", or "grid"
            show_labels: Whether to show camera name labels on images
            label_height: Height of label bar in pixels
            label_font_scale: Font scale for labels
            target_height: If set, resize all images to this height (maintains aspect ratio)
        """
        self.window_name = window_name
        self.layout = layout
        self.show_labels = show_labels
        self.label_height = label_height
        self.label_font_scale = label_font_scale
        self.target_height = target_height

        self._window_created = False
        self._last_size = None

    def _setup_window(self) -> None:
        """Create and position the display window."""
        if self._window_created:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Position window at a reasonable location (top-left area)
        cv2.moveWindow(self.window_name, 50, 50)
        self._window_created = True

    def _add_label(self, image: np.ndarray, label: str) -> np.ndarray:
        """
        Add a label bar to the top of an image.

        Args:
            image: Input image (BGR format)
            label: Text to display

        Returns:
            Image with label bar added
        """
        h, w = image.shape[:2]

        # Create label bar (dark background)
        label_bar = np.zeros((self.label_height, w, 3), dtype=np.uint8)
        label_bar[:] = (40, 40, 40)  # Dark gray background

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, self.label_font_scale, 1)[0]
        text_x = (w - text_size[0]) // 2  # Center text
        text_y = (self.label_height + text_size[1]) // 2
        cv2.putText(
            label_bar, label, (text_x, text_y),
            font, self.label_font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Stack label on top of image
        return np.vstack([label_bar, image])

    def _resize_to_height(self, image: np.ndarray, target_height: int) -> np.ndarray:
        """Resize image to target height while maintaining aspect ratio."""
        h, w = image.shape[:2]
        if h == target_height:
            return image
        scale = target_height / h
        new_w = int(w * scale)
        return cv2.resize(image, (new_w, target_height))

    def _combine_images(self, images: dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine multiple images into a single frame.

        Args:
            images: Dict mapping camera name to image array (RGB format)

        Returns:
            Combined image in BGR format for OpenCV display
        """
        if not images:
            # Return blank image if no cameras
            return np.zeros((480, 640, 3), dtype=np.uint8)

        processed = []

        # Sort by camera name for consistent ordering
        for name in sorted(images.keys()):
            img = images[name]

            # Convert RGB to BGR for OpenCV
            if img.shape[2] == 3:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img

            # Resize if target height specified
            if self.target_height:
                bgr = self._resize_to_height(bgr, self.target_height)

            # Add label if enabled
            if self.show_labels:
                # Clean up camera name for display
                display_name = name.replace("observation.images.", "").replace("_", " ").title()
                bgr = self._add_label(bgr, display_name)

            processed.append(bgr)

        # Handle different layouts
        if len(processed) == 1:
            return processed[0]

        if self.layout == "horizontal":
            # Make all images same height before horizontal stack
            max_h = max(img.shape[0] for img in processed)
            padded = []
            for img in processed:
                if img.shape[0] < max_h:
                    pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                    img = np.vstack([img, pad])
                padded.append(img)
            return np.hstack(padded)

        elif self.layout == "vertical":
            # Make all images same width before vertical stack
            max_w = max(img.shape[1] for img in processed)
            padded = []
            for img in processed:
                if img.shape[1] < max_w:
                    pad = np.zeros((img.shape[0], max_w - img.shape[1], 3), dtype=np.uint8)
                    img = np.hstack([img, pad])
                padded.append(img)
            return np.vstack(padded)

        else:  # grid layout
            return self._create_grid(processed)

    def _create_grid(self, images: list[np.ndarray]) -> np.ndarray:
        """Create a grid layout for multiple images."""
        n = len(images)
        if n <= 2:
            return np.hstack(images)

        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # Find max dimensions
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)

        # Create grid
        grid = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            h, w = img.shape[:2]
            grid[row*max_h:row*max_h+h, col*max_w:col*max_w+w] = img

        return grid

    def show(self, images: dict[str, np.ndarray]) -> int:
        """
        Display multiple camera images in a single window.

        Args:
            images: Dict mapping camera name to image array (RGB format)

        Returns:
            Key code from cv2.waitKey (for input handling)
        """
        if is_headless():
            return -1

        if not images:
            return cv2.waitKey(10)

        # Filter to only image keys
        image_dict = {k: v for k, v in images.items() if "image" in k.lower()}

        if not image_dict:
            return cv2.waitKey(10)

        # Setup window on first call
        self._setup_window()

        # Combine images
        combined = self._combine_images(image_dict)

        # Resize window if image size changed
        h, w = combined.shape[:2]
        if self._last_size != (w, h):
            cv2.resizeWindow(self.window_name, w, h)
            self._last_size = (w, h)

        # Display
        cv2.imshow(self.window_name, combined)

        return cv2.waitKey(10)

    def close(self) -> None:
        """Close the display window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False


# Convenience function for simple use cases
_default_display: Optional[CameraDisplay] = None


def show_cameras(
    images: dict[str, np.ndarray],
    window_name: str = "Camera View",
    layout: str = "horizontal",
) -> int:
    """
    Convenience function to display cameras without managing a CameraDisplay instance.

    Args:
        images: Dict mapping camera name to image array (RGB format)
        window_name: Name of the window
        layout: Layout type

    Returns:
        Key code from cv2.waitKey
    """
    global _default_display

    if _default_display is None or _default_display.window_name != window_name:
        if _default_display is not None:
            _default_display.close()
        _default_display = CameraDisplay(window_name=window_name, layout=layout)

    return _default_display.show(images)


def close_camera_display() -> None:
    """Close the default camera display."""
    global _default_display
    if _default_display is not None:
        _default_display.close()
        _default_display = None
