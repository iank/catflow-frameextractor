from frameextractor.utils.image_processing import yolo_to_ls
import pytest


def test_yolo_to_ls():
    original_width, original_height = 200, 400

    # Center of bounding box
    x_px, y_px = 100, 200

    # Size of bounding box
    width_px, height_px = 50, 50

    x, y, width, height = yolo_to_ls(
        x_px, y_px, width_px, height_px, original_width, original_height
    )
    assert x == pytest.approx(37.50)
    assert y == pytest.approx(43.75)
    assert width == pytest.approx(25.00)
    assert height == pytest.approx(12.50)
