from frameextractor.utils.image_processing import extract_frames

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent.resolve() / "test_files"


@pytest.mark.datafiles(
    FIXTURE_DIR / "car.mp4",
)
def test_extract_frames(datafiles):
    """Verify that extract_frames can open a video and return a list of frames"""
    video = str(next(datafiles.iterdir()))
    frames = extract_frames(video)
    assert len(frames) == 11
