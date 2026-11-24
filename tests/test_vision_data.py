import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pytest
import cv2

from src.chess_camera.chess_position import Position
from src.chess_camera.vision_collection import (
    Capture, HandCapture, VisionCollector, occupancy_samples,
)
from src.chess_camera.vision_data import (
    VisionDataset, VisionSample, demonstration_samples, normalize_view,
)


def recorded_board():
    image = np.zeros((300, 300, 3), np.uint8)
    image[:, :, 0] = np.arange(300, dtype=np.uint16)[:, None] % 256
    image[:, :, 1] = np.arange(300, dtype=np.uint16)[None, :] % 256
    grid = np.zeros((9, 9, 2), np.float32)
    grid[:, :, 0] = np.linspace(50, 250, 9)[None, :]
    grid[:, :, 1] = np.linspace(50, 250, 9)[:, None]
    return image, grid


def sample(session="session", *, demo=False):
    return VisionSample(
        "occupancy", "empty",
        {"crop": np.zeros((32, 32, 3), np.uint8),
         "valid": np.full((32, 32), 255, np.uint8)},
        {"label_source": "test"}, session, demo,
    )


def hand_sample(label="unknown", *, baseline=True, baseline_label="unknown", demo=False):
    image = np.full((48, 48, 3), 80, np.uint8)
    mask = np.full((48, 48), 255, np.uint8)
    image[12:36, 12:36] = 0
    mask[12:36, 12:36] = 0
    images = {"current": image.copy(), "current_valid": mask.copy()}
    if baseline:
        images.update({"baseline": image, "baseline_valid": mask})
    return VisionSample(
        "hand", label, images, {"baseline_compatible": baseline}, "hand", demo,
        baseline_label,
    )


def wait_for(predicate):
    deadline = monotonic() + 10
    while not predicate():
        if monotonic() > deadline:
            pytest.fail("Vision worker did not reach expected state.")
        sleep(0.01)


def test_normalized_board_halo_crop_and_ring_coordinates():
    image, grid = recorded_board()
    view = normalize_view(image, grid, 20)
    assert view.image.shape == (240, 240, 3)
    assert np.array_equal(view.image[40, 40], image[50, 50])
    assert np.array_equal(view.image[200, 200], image[250, 250])
    for row, column in ((0, 0), (3, 4), (7, 7)):
        crop, valid = view.square(row, column)
        assert crop.shape == (40, 40, 3)
        assert valid.shape == (40, 40)
        assert np.all(valid == 255)
        y, x = int((row + 2.5) * 20), int((column + 2.5) * 20)
        assert np.array_equal(crop[20, 20], view.image[y, x])
    ring, mask = view.ring()
    assert np.all(ring[60:180, 60:180] == 0)
    assert np.all(mask[60:180, 60:180] == 0)
    assert mask[50, 50] == 255


def test_normalized_invalid_pixels_and_degenerate_grid():
    image, grid = recorded_board()
    shifted = grid - 80
    view = normalize_view(image, shifted, 20)
    assert view.valid[0, 0] == 0
    assert np.all(view.image[view.valid == 0] == 0)
    with pytest.raises(ValueError, match="nondegenerate"):
        normalize_view(image, np.zeros_like(grid))


def test_normalized_perspective_grid_maps_to_exact_square_coordinates():
    image, grid = recorded_board()
    projective = np.array([
        [0.8, 0.1, 10], [0.05, 0.75, 12], [0.0004, 0.0002, 1],
    ], np.float64)
    distorted = cv2.perspectiveTransform(grid, projective)
    view = normalize_view(image, distorted, 20)
    mapped = cv2.perspectiveTransform(distorted, view.transform)
    expected = grid * 0.8
    assert np.max(np.abs(mapped - expected)) < 0.001


def test_dataset_transaction_hash_review_and_reference_safe_discard(tmp_path):
    dataset = VisionDataset.create(tmp_path)
    identifiers = dataset.save([sample(), sample("second")], 1_000_000)
    dataset.review(identifiers[0], "occupied")
    reloaded = VisionDataset(tmp_path)
    record = next(item for item in reloaded.records() if item.sample_id == identifiers[0])
    assert record.label == "occupied"
    assert record.original_label == "empty"
    assert record.reviewed
    reloaded.discard(sample_id=identifiers[0], reason="test")
    assert reloaded.images(identifiers[1])["crop"].shape == (32, 32, 3)
    with sqlite3.connect(dataset.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 2
        connection.execute("UPDATE assets SET png = ?", (b"corrupt",))
    with pytest.raises(ValueError, match="Corrupt"):
        reloaded.images(identifiers[1])


def test_quota_rolls_back_whole_batch_and_missing_drive_is_not_recreated(tmp_path):
    dataset = VisionDataset.create(tmp_path)
    with pytest.raises(OSError, match="quota"):
        dataset.save([sample(), sample()], 1)
    assert dataset.records() == []
    dataset.path.unlink()
    with pytest.raises(sqlite3.Error):
        dataset.save([sample()], 1_000_000)
    assert not dataset.path.exists()


def test_piece_colour_correction_preserves_provenance_and_reopens(tmp_path):
    dataset = VisionDataset.create(tmp_path)
    original = replace(sample(), metadata={
        "piece_colour": "white", "piece_colour_source": "logical-position",
    })
    identifier = dataset.save([original], 1_000_000)[0]
    dataset.review(identifier, "occupied", piece_colour="white")
    accepted = dataset.records()[0]
    assert accepted.metadata["piece_colour_source"] == "logical-position"
    assert "piece_colour_revisions" not in accepted.metadata
    dataset.review(identifier, "occupied", piece_colour="black")
    reopened = VisionDataset(tmp_path)
    stored = reopened.records()[0]
    assert stored.label == "occupied"
    assert stored.original_label == "empty"
    assert stored.metadata["piece_colour"] == "black"
    assert stored.metadata["piece_colour_source"] == "reviewer"
    history = stored.metadata["piece_colour_revisions"]
    assert len(history) == 1
    assert history[0]["previous"] == "white"
    assert history[0]["previous_source"] == "logical-position"
    assert history[0]["label"] == "black"
    assert history[0]["created"]
    reopened.review(identifier, "occupied", piece_colour="black")
    assert reopened.records()[0].metadata["piece_colour_revisions"] == history


@pytest.mark.parametrize("label, colour", [
    ("empty", "white"), ("empty", "black"), ("occupied", "none"),
    ("occupied", "invalid"),
])
def test_invalid_piece_colour_review_does_not_partially_save(tmp_path, label, colour):
    dataset = VisionDataset.create(tmp_path)
    identifier = dataset.save([sample()], 1_000_000)[0]
    before = dataset.records()[0]
    with pytest.raises(ValueError):
        dataset.review(identifier, label, piece_colour=colour)
    assert dataset.records()[0] == before
    with sqlite3.connect(dataset.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 0


def test_hand_review_rejects_piece_colour(tmp_path):
    image, grid = recorded_board()
    _, samples = demonstration_samples(image, grid)
    dataset = VisionDataset.create(tmp_path)
    identifier = dataset.save([samples[-1]], 20_000_000)[0]
    with pytest.raises(ValueError, match="piece colour"):
        dataset.review(identifier, "clear", piece_colour="none")
    assert not dataset.records()[0].reviewed


def test_shared_baseline_review_is_atomic_and_preserves_other_current_labels(tmp_path):
    dataset = VisionDataset.create(tmp_path)
    identifiers = dataset.save([
        hand_sample("clear"), hand_sample("crossing"),
    ], 1_000_000)
    members = dataset.baseline_members(identifiers[0])
    assert set(members) == set(identifiers)
    before = dataset.records()
    with pytest.raises(ValueError, match="confirmation"):
        dataset.review(identifiers[0], "crossing", baseline_label="invalid")
    assert dataset.records() == before
    with pytest.raises(ValueError, match="piece colour"):
        dataset.review(
            identifiers[0], "crossing", baseline_label="invalid", piece_colour="white",
            confirmed_baseline_members=members,
        )
    assert dataset.records() == before
    dataset.review(
        identifiers[0], "clear", baseline_label="invalid",
        confirmed_baseline_members=members,
    )
    reopened = VisionDataset(tmp_path)
    records = {item.sample_id: item for item in reopened.records()}
    assert records[identifiers[0]].label == "clear"
    assert records[identifiers[1]].label == "crossing"
    assert not records[identifiers[1]].reviewed
    assert len({item.baseline_id for item in records.values()}) == 1
    for item in records.values():
        assert item.baseline_label == "invalid" and item.baseline_reviewed
        assert item.metadata["baseline_label_strength"] == "reviewed"
        assert item.training_targets["current_obstruction"]
        assert item.training_targets["baseline_validity"]
        assert not item.training_targets["comparison"]
        assert item.metadata["original_baseline_label"] == "unknown"
        assert item.metadata["baseline_revisions"][0]["previous"] == "unknown"
    dataset.discard(sample_id=identifiers[0], reason="test")
    assert "baseline" in dataset.images(identifiers[1])
    assert dataset.records()[0].baseline_label == "invalid"


def test_shared_baseline_link_changes_require_fresh_confirmation(tmp_path):
    dataset = VisionDataset.create(tmp_path)
    identifiers = dataset.save([hand_sample(), hand_sample()], 1_000_000)
    confirmed = dataset.baseline_members(identifiers[0])
    dataset.save([hand_sample()], 1_000_000)
    with pytest.raises(ValueError, match="confirmation"):
        dataset.review(
            identifiers[0], "clear", baseline_label="valid",
            confirmed_baseline_members=confirmed,
        )
    assert all(item.label == "unknown" and item.baseline_label == "unknown"
               for item in dataset.records())


def test_invalid_baseline_masks_only_unreviewed_dependent_current_targets(tmp_path):
    dataset = VisionDataset.create(tmp_path)
    automatic = replace(hand_sample("crossing"), metadata={
        "label_source": "deterministic-black-board-gate", "baseline_compatible": True,
    })
    identifiers = dataset.save([automatic, automatic], 1_000_000)
    dataset.review(
        identifiers[0], "clear", baseline_label="invalid",
        confirmed_baseline_members=dataset.baseline_members(identifiers[0]),
    )
    records = {item.sample_id: item for item in dataset.records()}
    assert records[identifiers[0]].training_targets["current_obstruction"]
    assert records[identifiers[1]].label == "crossing"
    assert not records[identifiers[1]].training_targets["current_obstruction"]
    assert records[identifiers[1]].training_targets["baseline_validity"]
    dataset.review(identifiers[1], "crossing")
    assert all(item.training_targets["current_obstruction"] for item in dataset.records())


def test_new_observation_inherits_reviewed_baseline_not_automatic_proposal(tmp_path):
    dataset = VisionDataset.create(tmp_path)
    identifier = dataset.save([hand_sample("clear")], 1_000_000)[0]
    dataset.review(identifier, "clear", baseline_label="invalid")
    automatic = replace(
        hand_sample("crossing", baseline_label="valid"),
        metadata={"baseline_label_source": "deterministic-baseline-proposal"},
    )
    new_id = dataset.save([automatic], 1_000_000)[0]
    new = next(item for item in dataset.records() if item.sample_id == new_id)
    assert new.baseline_label == "invalid"
    assert new.metadata["baseline_label_source"] == "reviewer"
    assert new.metadata["baseline_revisions"][0]["label"] == "invalid"
    with pytest.raises(ValueError, match="Conflicting"):
        dataset.save([hand_sample(baseline_label="valid")], 1_000_000)
    assert len(dataset.records()) == 2


def test_independent_hand_targets_and_manifest_support_missing_or_bad_baselines(tmp_path):
    dataset = VisionDataset.create(tmp_path)
    current_only = hand_sample("clear", baseline=False)
    invalid_reference = hand_sample(baseline_label="invalid")
    unknown = replace(hand_sample(), session_id="unlabelled", images={
        **hand_sample().images,
        "baseline": np.full((48, 48, 3), 100, np.uint8),
    })
    identifiers = dataset.save([
        current_only, invalid_reference, unknown, replace(current_only, demo=True),
    ], 1_000_000)
    eligible = {item.sample_id: item for item in dataset.records(training_only=True)}
    assert set(eligible) == set(identifiers[:2])
    assert eligible[identifiers[0]].baseline_id is None
    assert eligible[identifiers[0]].training_targets == {
        "current_obstruction": True, "baseline_validity": False, "comparison": False,
    }
    assert eligible[identifiers[1]].training_targets == {
        "current_obstruction": False, "baseline_validity": True, "comparison": False,
    }
    destination = tmp_path / "manifest.jsonl"
    dataset.export_manifest(destination)
    exported = {row["id"]: row for row in map(json.loads, destination.read_text().splitlines())}
    assert exported[identifiers[0]]["schema_version"] == 2
    assert set(exported[identifiers[0]]["assets"]) == {"current", "current_valid"}
    assert exported[identifiers[1]]["baseline_split_group"] == (
        eligible[identifiers[1]].baseline_id
    )
    assert exported[identifiers[1]]["training_targets"]["baseline_validity"]


def test_legacy_baseline_is_unknown_without_rewriting_dataset_on_open(tmp_path):
    dataset = VisionDataset.create(tmp_path)
    dataset.save([hand_sample("clear")], 1_000_000)
    with sqlite3.connect(dataset.path) as connection:
        connection.execute("UPDATE samples SET metadata='{}'")
    reopened = VisionDataset(tmp_path)
    stored = reopened.records()[0]
    assert stored.baseline_id is not None
    assert stored.baseline_label == "unknown"
    assert stored.training_targets["current_obstruction"]
    assert not stored.training_targets["comparison"]
    with sqlite3.connect(dataset.path) as connection:
        assert connection.execute("SELECT metadata FROM samples").fetchone()[0] == "{}"
        assert connection.execute("SELECT version FROM dataset").fetchone()[0] == 1


def test_hand_asset_validation_and_isolated_current_reads(tmp_path):
    dataset = VisionDataset.create(tmp_path)
    missing_mask = hand_sample(baseline=False)
    missing_mask.images["baseline"] = np.zeros((48, 48, 3), np.uint8)
    with pytest.raises(ValueError, match="roles"):
        dataset.save([missing_mask], 1_000_000)
    with pytest.raises(ValueError, match="Baseline labels"):
        dataset.save([hand_sample(baseline=False, baseline_label="valid")], 1_000_000)
    original = hand_sample("clear")
    original.images["baseline"] = np.full((48, 48, 3), 120, np.uint8)
    identifier = dataset.save([original], 1_000_000)[0]
    with sqlite3.connect(dataset.path) as connection:
        connection.execute(
            "UPDATE assets SET png=? WHERE hash=(SELECT hash FROM sample_assets "
            "WHERE sample=? AND role='baseline')", (b"bad baseline", identifier),
        )
    assert set(dataset.images(identifier, roles=("current", "current_valid"))) == {
        "current", "current_valid",
    }
    with pytest.raises(ValueError, match="Corrupt"):
        dataset.images(identifier, roles=("baseline", "baseline_valid"))


def test_dataset_identity_rejects_replaced_database(tmp_path):
    old = VisionDataset.create(tmp_path)
    old.path.unlink()
    new = VisionDataset.create(tmp_path)
    assert new.identity != old.identity
    with pytest.raises(ValueError, match="identity"):
        old.records()


def test_recorded_demos_are_excluded_even_after_manual_label_review(tmp_path):
    image, grid = recorded_board()
    view, samples = demonstration_samples(image, grid)
    assert view.image.shape == (1200, 1200, 3)
    assert len(samples) == 8
    assert sum(item.kind == "occupancy" for item in samples) == 6
    dataset = VisionDataset.create(tmp_path)
    identifiers = dataset.save(samples, 20_000_000)
    dataset.review(identifiers[0], "occupied")
    assert dataset.records(training_only=True) == []
    hand = [item for item in dataset.records() if item.kind == "hand"]
    assert any(item.metadata["synthetic_overlay"] for item in hand)
    for item in hand:
        images = dataset.images(item.sample_id)
        assert np.all(images["current"][300:900, 300:900] == 0)


@pytest.mark.parametrize("turns", range(4))
def test_occupancy_sampling_balances_and_includes_unchanged_occupancy_capture(turns):
    image, grid = recorded_board()
    before: Position = {(6, 0): "P", (6, 1): "P", (7, 4): "K", (0, 4): "k"}
    after = {**before, (4, 0): "P"}
    del after[(6, 0)]
    capture = Capture(image, grid, after, before, turns, "game", "move")
    samples = occupancy_samples(capture, "session", 1)
    assert len(samples) == 6
    assert sum(item.label == "empty" for item in samples) == 3
    assert sum(item.label == "occupied" for item in samples) == 3
    capture = replace(capture, previous={(4, 0): "p"}, position={(4, 0): "P"})
    assert any(item.metadata["algebraic_square"] == "a4"
               for item in occupancy_samples(capture, "session", 2))


def test_delayed_commit_and_final_pending_discard(tmp_path):
    dataset = VisionDataset.create(tmp_path / "data")
    collector = VisionCollector(dataset, 20_000_000, tmp_path / "control")
    image, grid = recorded_board()
    first = Capture(image, grid, {(4, 0): "P"}, {(6, 0): "P"},
                    0, "game", "first")
    try:
        collector.accepted(first)
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        assert dataset.records() == []
        collector.accepted(replace(
            first, previous=first.position, position={(3, 0): "P"},
            move_id="second", finished=True,
        ))
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        records = dataset.records()
        assert records
        assert all(item.metadata["confirmed_by"] == "second" for item in records)
        assert all(item.metadata["move"] == "first" for item in records)
    finally:
        collector.close(wait=True)


def test_correction_discards_stored_session_without_quarantine(tmp_path):
    dataset = VisionDataset.create(tmp_path / "data")
    collector = VisionCollector(dataset, 20_000_000, tmp_path / "control")
    image, grid = recorded_board()
    first = Capture(image, grid, {(4, 0): "P"}, {(6, 0): "P"},
                    0, "game", "first")
    try:
        collector.accepted(first)
        collector.accepted(replace(
            first, previous=first.position, position={(3, 0): "P"}, move_id="second",
        ))
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        assert dataset.records()
        old_session = collector.session
        collector.invalidate("reconciliation")
        wait_for(lambda: collector.session != old_session)
        assert dataset.records() == []
        assert not collector.journal.exists()
    finally:
        collector.close(wait=True)


def test_worker_failure_is_explicit_and_never_escapes_into_gameplay(tmp_path):
    dataset = VisionDataset.create(tmp_path / "data")
    collector = VisionCollector(dataset, 1, tmp_path / "control")
    image, grid = recorded_board()
    collector.recorded_demo(image, grid)
    wait_for(collector.closed.is_set)
    assert collector.failed.is_set()
    assert not collector.active
    assert "quota" in collector.messages.get_nowait()
    assert dataset.records() == []
    collector.close(wait=True)


def test_correction_after_pause_still_discards_that_sessions_samples(tmp_path):
    dataset = VisionDataset.create(tmp_path / "data")
    collector = VisionCollector(dataset, 20_000_000, tmp_path / "control")
    dataset.save([sample(collector.session)], 1_000_000)
    collector.close(wait=True)
    collector.invalidate("late correction")
    wait_for(collector.closed.is_set)
    assert dataset.records() == []
    assert not collector.journal.exists()
    collector.close(wait=True)


def test_unavailable_drive_keeps_durable_deletion_intent_until_reopened(tmp_path):
    root = tmp_path / "data"
    control = tmp_path / "control"
    dataset = VisionDataset.create(root)
    collector = VisionCollector(dataset, 20_000_000, control)
    dataset.save([sample(collector.session)], 1_000_000)
    collector.close(wait=True)
    unavailable = root / "unmounted.sqlite3"
    dataset.path.rename(unavailable)
    collector.invalidate("correction while disconnected")
    wait_for(collector.closed.is_set)
    assert collector.journal.exists()
    assert not dataset.path.exists()
    unavailable.rename(dataset.path)
    replacement = VisionCollector(VisionDataset(root), 20_000_000, control)
    try:
        wait_for(lambda: not replacement.journal.exists())
        assert dataset.records() == []
    finally:
        replacement.close(wait=True)


def test_invalidation_prevents_queued_old_history_from_reentering_dataset(tmp_path):
    dataset = VisionDataset.create(tmp_path / "data")
    collector = VisionCollector(dataset, 20_000_000, tmp_path / "control")
    image, grid = recorded_board()
    first = Capture(image, grid, {(4, 0): "P"}, {(6, 0): "P"},
                    0, "game", "first")
    try:
        collector.accepted(first)
        collector.invalidate("history repair")
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        collector.accepted(replace(
            first, previous=first.position, position={(3, 0): "P"}, move_id="second",
        ))
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        assert dataset.records() == []
    finally:
        collector.close(wait=True)


def test_end_game_discards_pending_without_deleting_confirmed_examples(tmp_path):
    dataset = VisionDataset.create(tmp_path / "data")
    collector = VisionCollector(dataset, 20_000_000, tmp_path / "control")
    image, grid = recorded_board()
    first = Capture(image, grid, {(4, 0): "P"}, {(6, 0): "P"},
                    0, "game", "first")
    try:
        collector.accepted(first)
        second = replace(
            first, previous=first.position, position={(3, 0): "P"}, move_id="second",
        )
        collector.accepted(second)
        collector.end_game()
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        confirmed_count = len(dataset.records())
        assert confirmed_count > 0
        collector.accepted(replace(
            first, game_key="new-game", move_id="new-first",
        ))
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        assert len(dataset.records()) == confirmed_count
    finally:
        collector.close(wait=True)


def test_dataset_cannot_use_diagnostic_retention_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside diagnostics"):
        VisionDataset.create(tmp_path / "diagnostics" / "dataset")
    assert not (tmp_path / "diagnostics").exists()


def test_periodic_complete_snapshot_retains_all_64_squares():
    image, grid = recorded_board()
    capture = Capture(image, grid, {(4, 0): "P"}, {(6, 0): "P"}, 0, "game", "move")
    samples = occupancy_samples(capture, "session", 50)
    assert len(samples) == 64
    assert all(item.metadata["full_position_snapshot"] for item in samples)


def test_hand_transitions_and_stillness_release_remain_reviewable(tmp_path):
    dataset = VisionDataset.create(tmp_path / "data")
    collector = VisionCollector(dataset, 20_000_000, tmp_path / "control")
    image, grid = recorded_board()
    image = np.pad(image, ((0, 10), (0, 10), (0, 0)))
    try:
        collector.hand(HandCapture(image, grid, False, baseline=True))
        collector.hand(HandCapture(image, grid, True))
        collector.hand(HandCapture(image, grid, False, uncertain=True))
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        records = dataset.records()
        assert {item.label for item in records} == {"clear", "crossing", "unknown"}
        uncertain = next(item for item in records if item.label == "unknown")
        assert uncertain.metadata["stillness_release"] is True
        assert all(item.metadata["label_strength"] == "weak" for item in records)
        assert all(item.metadata["baseline_label_strength"] == "weak" for item in records)
    finally:
        collector.close(wait=True)


def test_hand_collection_explicitly_rejects_insufficient_halo(tmp_path):
    dataset = VisionDataset.create(tmp_path / "data")
    collector = VisionCollector(dataset, 20_000_000, tmp_path / "control")
    image, grid = recorded_board()
    collector.hand(HandCapture(image, grid, False, baseline=True))
    wait_for(collector.closed.is_set)
    assert collector.failed.is_set()
    assert "99% edge coverage" in collector.messages.get_nowait()
    assert dataset.records() == []
    collector.close(wait=True)


def test_hand_collection_keeps_current_observations_without_a_baseline(tmp_path):
    dataset = VisionDataset.create(tmp_path / "data")
    collector = VisionCollector(dataset, 20_000_000, tmp_path / "control")
    image, grid = recorded_board()
    image = np.pad(image, ((10, 10), (10, 10), (0, 0)))
    grid = grid + 10
    try:
        collector.hand(HandCapture(image, grid, True))
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        first = dataset.records()[0]
        assert first.label == "crossing"
        assert first.baseline_id is None
        assert first.training_targets["current_obstruction"]
        collector.hand(HandCapture(image, grid, False, baseline=True))
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        collector.hand(HandCapture(image, grid + 1, True))
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        records = dataset.records()
        assert len(records) == 3
        assert records[-1].baseline_id is None
        assert records[-1].label == "crossing"
        collector.hand(HandCapture(image, grid + 1, False, baseline=True))
        collector.hand(HandCapture(image[:280, :280], grid + 1, False, baseline=True))
        collector.hand(HandCapture(image, grid + 1, True))
        wait_for(lambda: collector.jobs.unfinished_tasks == 0)
        assert dataset.records()[-1].baseline_id is None
        assert dataset.records()[-1].label == "crossing"
        assert not collector.failed.is_set()
    finally:
        collector.close(wait=True)
