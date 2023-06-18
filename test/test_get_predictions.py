from frameextractor.utils.image_processing import get_predictions
from frameextractor.utils.model import load_model

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
    model = load_model(model_name)

    predictions = get_predictions(model, model_name, str(img_path), 0.2)
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
    assert pv["x"] >= 0 and pv["x"] <= 100
    assert pv["y"] >= 0 and pv["y"] <= 100
    assert pv["width"] >= 0 and pv["width"] <= prediction["original_width"]
    assert pv["height"] >= 0 and pv["height"] <= prediction["original_height"]
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
    model = load_model(model_name)

    predictions = get_predictions(model, model_name, str(img_path), 0.2)
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
    model = load_model(model_name)

    predictions = get_predictions(model, model_name, str(img_path), 0.44)
    assert len(predictions["result"]) == 1

    assert predictions["result"][0]["value"]["rectanglelabels"] == ["dog"]
