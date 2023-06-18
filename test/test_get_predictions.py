from frameextractor.utils.image_processing import get_predictions
from frameextractor.utils.model import Model

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent.resolve() / "test_files"


@pytest.mark.datafiles(
    FIXTURE_DIR / "car.png",
    FIXTURE_DIR / "yolov5n.pt",
)
def test_get_predictions_structure(datafiles):
    """Verify that get_predictions produces the correct output"""
    img_path = datafiles / "car.png"
    assert img_path.is_file()

    model_name = str(datafiles / "yolov5n")  # load_model adds the .pt
    model = Model(model_name, 0.2)

    predictions = get_predictions(model, str(img_path))
    assert predictions["model_version"] == model_name
    assert "score" in predictions
    assert len(predictions["result"]) == 1

    prediction = predictions["result"][0]
    assert "id" in prediction
    assert prediction["type"] == "rectanglelabels"
    assert prediction["from_name"] == "label"
    assert prediction["to_name"] == "image"
    assert prediction["original_width"] == 640
    assert prediction["original_height"] == 452
    assert prediction["image_rotation"] == 0
    assert "value" in prediction

    pv = prediction["value"]
    assert pv["rotation"] == 0
    assert pv["x"] >= 3.20 and pv["x"] <= 3.21
    assert pv["y"] >= 25.60 and pv["y"] <= 25.64
    assert pv["width"] >= 96.79 and pv["width"] <= 96.80
    assert pv["height"] >= 63.93 and pv["height"] <= 63.94
    assert pv["rectanglelabels"] == ["car"]


@pytest.mark.datafiles(
    FIXTURE_DIR / "cat_and_dog.png",
    FIXTURE_DIR / "yolov5n.pt",
)
def test_get_predictions_multiple(datafiles):
    """Verify multiple detections"""
    img_path = datafiles / "cat_and_dog.png"
    assert img_path.is_file()

    model_name = str(datafiles / "yolov5n")  # load_model adds the .pt
    model = Model(model_name, 0.2)

    predictions = get_predictions(model, str(img_path))
    assert len(predictions["result"]) == 2

    labels = [x["value"]["rectanglelabels"][0] for x in predictions["result"]]
    assert "cat" in labels
    assert "dog" in labels


@pytest.mark.datafiles(
    FIXTURE_DIR / "cat_and_dog.png",
    FIXTURE_DIR / "yolov5n.pt",
)
def test_get_predictions_threshold(datafiles):
    """Verify confidence threshold"""
    img_path = datafiles / "cat_and_dog.png"
    assert img_path.is_file()

    model_name = str(datafiles / "yolov5n")  # load_model adds the .pt
    model = Model(model_name, 0.44)

    predictions = get_predictions(model, str(img_path))
    assert len(predictions["result"]) == 1

    assert predictions["result"][0]["value"]["rectanglelabels"] == ["dog"]
