import numpy as np

from src.chess_camera.chess_position import standard_position
from src.chess_camera.reference_store import ReferenceImageStore


def test_reference_store_round_trip_and_matching(tmp_path) -> None:
    store = ReferenceImageStore(tmp_path / "reference.npz")
    image = np.full((800, 800, 3), 127, dtype=np.uint8)
    position = standard_position()

    store.save(
        image,
        camera_id="camera",
        camera_resolution=(1920, 1080),
        orientation=2,
        position=position,
    )

    loaded = store.load_matching(
        camera_id="camera",
        camera_resolution=(1920, 1080),
        orientation=2,
        position=position,
    )
    assert loaded is not None
    np.testing.assert_array_equal(loaded, image)
    assert (
        store.load_matching(
            camera_id="other-camera",
            camera_resolution=(1920, 1080),
            orientation=2,
            position=position,
        )
        is None
    )
    assert (
        store.load_matching(
            camera_id="camera",
            camera_resolution=(1280, 720),
            orientation=2,
            position=position,
        )
        is None
    )
    assert (
        store.load_matching(
            camera_id="camera",
            camera_resolution=(1920, 1080),
            orientation=1,
            position=position,
        )
        is None
    )
    moved_position = position.copy()
    moved_position[(4, 4)] = moved_position.pop((6, 4))
    assert (
        store.load_matching(
            camera_id="camera",
            camera_resolution=(1920, 1080),
            orientation=2,
            position=moved_position,
        )
        is None
    )
