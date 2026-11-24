"""Normalized camera images and an isolated, transactional vision dataset."""

import hashlib
import json
import shutil
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Literal
from uuid import UUID, uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from .board_detector import FloatArray, ImageArray

SampleKind = Literal["occupancy", "hand"]
Mask = NDArray[np.uint8]
LABELS: dict[str, tuple[str, ...]] = {
    "occupancy": ("empty", "occupied", "unknown"),
    "hand": ("clear", "crossing", "unknown"),
}
PIECE_COLOURS = ("white", "black", "none", "unknown")
BASELINE_LABELS = ("valid", "invalid", "unknown")
DATASET_FILENAME = "vision.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class NormalizedView:
    image: ImageArray
    valid: Mask
    transform: NDArray[np.float64]
    pixels_per_square: int

    def square(self, row: int, column: int) -> tuple[ImageArray, Mask]:
        if not (0 <= row < 8 and 0 <= column < 8):
            raise ValueError("Target square must be inside the board.")
        p = self.pixels_per_square
        x, y = round((column + 1.5) * p), round((row + 1.5) * p)
        return (
            self.image[y : y + 2 * p, x : x + 2 * p].copy(),
            self.valid[y : y + 2 * p, x : x + 2 * p].copy(),
        )

    def ring(self) -> tuple[ImageArray, Mask]:
        p = self.pixels_per_square
        image, mask = self.image.copy(), self.valid.copy()
        image[3 * p : 9 * p, 3 * p : 9 * p] = 0
        mask[3 * p : 9 * p, 3 * p : 9 * p] = 0
        return image, mask


def normalize_view(
    frame: ImageArray, grid: FloatArray, pixels_per_square: int = 100
) -> NormalizedView:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("A uint8 BGR camera image is required.")
    if grid.shape != (9, 9, 2) or not np.isfinite(grid).all():
        raise ValueError("A finite 9x9 camera grid is required.")
    if pixels_per_square < 16 or pixels_per_square > 200:
        raise ValueError("Pixels per square must be between 16 and 200.")
    source = np.asarray(grid[[0, 0, 8, 8], [0, 8, 8, 0]], np.float32)
    if not cv2.isContourConvex(source) or abs(cv2.contourArea(source)) < 64:
        raise ValueError("Board corners must define a nondegenerate convex board.")
    p = pixels_per_square
    destination = np.asarray(
        [[2 * p, 2 * p], [10 * p, 2 * p], [10 * p, 10 * p], [2 * p, 10 * p]],
        np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    size = 12 * p
    image = cv2.warpPerspective(
        frame, transform, (size, size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    # Linear interpolation requires a completely observed source footprint.
    valid = cv2.warpPerspective(
        np.full(frame.shape[:2], 255, np.uint8),
        transform, (size, size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = np.where(valid == 255, 255, 0).astype(np.uint8)
    image[valid == 0] = 0
    return NormalizedView(
        np.asarray(image, dtype=np.uint8), valid,
        np.asarray(transform, dtype=np.float64), p,
    )


def coverage(mask: Mask) -> float:
    return float(np.count_nonzero(mask) / mask.size)


def ring_coverage(view: NormalizedView) -> dict[str, float]:
    p = view.pixels_per_square
    return {
        "top": coverage(view.valid[: 3 * p]),
        "bottom": coverage(view.valid[9 * p :]),
        "left": coverage(view.valid[:, : 3 * p]),
        "right": coverage(view.valid[:, 9 * p :]),
    }


@dataclass(frozen=True)
class VisionSample:
    kind: SampleKind
    label: str
    images: dict[str, ImageArray | Mask]
    metadata: dict[str, object]
    session_id: str
    demo: bool = False
    baseline_label: str = "unknown"


@dataclass(frozen=True)
class StoredSample:
    sample_id: str
    kind: SampleKind
    label: str
    original_label: str
    session_id: str
    demo: bool
    metadata: dict[str, object]
    reviewed: bool

    @property
    def baseline_id(self) -> str | None:
        value = self.metadata.get("baseline_id")
        return value if isinstance(value, str) else None

    @property
    def baseline_label(self) -> str:
        return str(self.metadata.get("baseline_label", "unknown"))

    @property
    def baseline_reviewed(self) -> bool:
        return self.metadata.get("baseline_reviewed") is True

    @property
    def training_targets(self) -> dict[str, bool]:
        known = not self.demo and self.label != "unknown"
        if self.kind == "occupancy":
            return {"occupancy": known}
        if (
            self.baseline_label == "invalid" and not self.reviewed
            and self.metadata.get("label_source") == "deterministic-black-board-gate"
        ):
            known = False
        return {
            "current_obstruction": known,
            "baseline_validity": (
                not self.demo and self.baseline_id is not None
                and self.baseline_label != "unknown"
            ),
            "comparison": (
                known and self.baseline_id is not None
                and self.baseline_label == "valid"
                and self.metadata.get("baseline_compatible") is True
            ),
        }


def baseline_identity(image_hash: str, mask_hash: str) -> str:
    return hashlib.sha256(f"{image_hash}:{mask_hash}".encode("ascii")).hexdigest()


def baseline_state(metadata: dict[str, object], identity: str) -> dict[str, object]:
    label = metadata.get("baseline_label", "unknown")
    original = metadata.get("original_baseline_label", label)
    history = metadata.get("baseline_revisions", [])
    if metadata.get("baseline_schema_version", 1) != 1:
        raise ValueError("Unsupported baseline metadata version.")
    if label not in BASELINE_LABELS or original not in BASELINE_LABELS:
        raise ValueError("Invalid baseline label.")
    if not isinstance(history, list):
        raise ValueError("Invalid baseline revision history.")
    return {
        "baseline_schema_version": 1, "baseline_id": identity, "baseline_label": label,
        "original_baseline_label": original,
        "baseline_label_source": metadata.get("baseline_label_source", "not-recorded"),
        "baseline_label_strength": metadata.get("baseline_label_strength", "unknown"),
        "baseline_reviewed": metadata.get("baseline_reviewed") is True,
        "baseline_revisions": history,
    }


class VisionDataset:
    """Connections are short lived; never recreate a missing external dataset."""

    def __init__(self, root: Path) -> None:
        self.path = root.resolve() / DATASET_FILENAME
        with self._connect() as connection:
            row = connection.execute(
                "SELECT version, identity FROM dataset"
            ).fetchone()
            if row is None or row[0] != 1:
                raise ValueError("Unsupported or incomplete vision dataset.")
            self.identity = UUID(str(row[1])).hex
            if self.identity != row[1]:
                raise ValueError("Invalid dataset identity.")

    @classmethod
    def create(cls, root: Path) -> "VisionDataset":
        if root.resolve().is_relative_to((Path.cwd() / "diagnostics").resolve()):
            raise ValueError(
                "Choose a directory outside diagnostics; diagnostic retention can delete it."
            )
        root.mkdir(parents=True, exist_ok=True)
        path = root / DATASET_FILENAME
        if path.exists():
            return cls(root)
        # Exclusive file creation avoids overwriting an existing dataset.
        with path.open("xb"):
            pass
        try:
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.executescript(
                    """
                    PRAGMA foreign_keys = ON;
                    CREATE TABLE dataset(version INTEGER, identity TEXT);
                    CREATE TABLE assets(
                        hash TEXT PRIMARY KEY, png BLOB NOT NULL);
                    CREATE TABLE samples(
                        id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                        label TEXT NOT NULL, original_label TEXT NOT NULL,
                        session TEXT NOT NULL, demo INTEGER NOT NULL,
                        metadata TEXT NOT NULL, created TEXT NOT NULL,
                        reviewed INTEGER NOT NULL DEFAULT 0);
                    CREATE TABLE sample_assets(
                        sample TEXT REFERENCES samples(id) ON DELETE CASCADE,
                        role TEXT NOT NULL, hash TEXT REFERENCES assets(hash),
                        PRIMARY KEY(sample, role));
                    CREATE TABLE revisions(
                        sample TEXT REFERENCES samples(id) ON DELETE CASCADE,
                        label TEXT NOT NULL, created TEXT NOT NULL);
                    CREATE TABLE deletions(
                        sample TEXT PRIMARY KEY, reason TEXT, created TEXT);
                    CREATE INDEX samples_session ON samples(session);
                    """
                )
                connection.execute(
                    "INSERT INTO dataset VALUES (1, ?)", (uuid4().hex,)
                )
        except (OSError, sqlite3.Error):
            path.unlink(missing_ok=True)
            raise
        return cls(root)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(
            self.path.as_uri() + "?mode=rw", uri=True, timeout=0.2
        )
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            identity = getattr(self, "identity", None)
            if identity is not None:
                row = connection.execute("SELECT identity FROM dataset").fetchone()
                if row is None or row[0] != identity:
                    raise ValueError("The selected dataset identity changed.")
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, samples: list[VisionSample], quota_bytes: int) -> list[str]:
        encoded: list[tuple[VisionSample, dict[str, tuple[str, bytes]]]] = []
        for sample in samples:
            if sample.kind not in LABELS or sample.label not in LABELS[sample.kind]:
                raise ValueError("Invalid sample label.")
            roles = set(sample.images)
            expected = {"crop", "valid"} if sample.kind == "occupancy" else {
                "current", "current_valid"
            }
            if sample.kind == "hand" and "baseline" in roles:
                expected |= {"baseline", "baseline_valid"}
            if roles != expected:
                raise ValueError("Missing or unexpected sample image roles.")
            if sample.baseline_label not in BASELINE_LABELS or (
                sample.baseline_label != "unknown"
                and (sample.kind != "hand" or "baseline" not in roles)
            ):
                raise ValueError("Baseline labels require a baseline image.")
            for role, image in sample.images.items():
                if image.dtype != np.uint8:
                    raise ValueError("Sample images must use uint8 pixels.")
                if "valid" in role:
                    if image.ndim != 2:
                        raise ValueError("Validity masks must be two dimensional.")
                elif image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError("Colour assets must have three channels.")
            pairs = (
                [("crop", "valid")] if sample.kind == "occupancy" else
                [("current", "current_valid")]
            )
            if sample.kind == "hand" and "baseline" in roles:
                pairs.append(("baseline", "baseline_valid"))
            for image_role, mask_role in pairs:
                if sample.images[image_role].shape[:2] != sample.images[mask_role].shape:
                    raise ValueError("Image and validity mask dimensions differ.")
            if sample.kind == "hand" and "baseline" in roles and (
                sample.images["baseline"].shape != sample.images["current"].shape
            ):
                raise ValueError("Baseline and current dimensions differ.")
            assets: dict[str, tuple[str, bytes]] = {}
            for role, image in sample.images.items():
                success, png = cv2.imencode(".png", image)
                if not success:
                    raise ValueError(f"Could not encode {role}.")
                data = png.tobytes()
                assets[role] = (hashlib.sha256(data).hexdigest(), data)
            encoded.append((sample, assets))
        estimated = sum(
            len(data) for _, assets in encoded for _, data in assets.values()
        )
        if shutil.disk_usage(self.path.parent).free < estimated + 8_000_000:
            raise OSError("Insufficient free space on the dataset drive.")
        identifiers: list[str] = []
        with self._connect() as connection:
            for sample, assets in encoded:
                identifier = uuid4().hex
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    (identifier, sample.kind, sample.label, sample.label,
                     sample.session_id, int(sample.demo),
                     json.dumps({
                         **sample.metadata,
                         "asset_shapes": {
                             role: list(image.shape)
                             for role, image in sample.images.items()
                         },
                     }, allow_nan=False), utc_now()),
                )
                for role, (digest, data) in assets.items():
                    connection.execute(
                        "INSERT OR IGNORE INTO assets VALUES (?, ?)", (digest, data)
                    )
                    connection.execute(
                        "INSERT INTO sample_assets VALUES (?, ?, ?)",
                        (identifier, role, digest),
                    )
                if sample.kind == "hand":
                    metadata = dict(sample.metadata)
                    if "baseline" in assets:
                        identity = baseline_identity(
                            assets["baseline"][0], assets["baseline_valid"][0],
                        )
                        metadata["baseline_label"] = sample.baseline_label
                        state = baseline_state(metadata, identity)
                        members = self._baseline_members(connection, identifier)
                        others = [member for member in members if member != identifier]
                        if others:
                            previous = json.loads(connection.execute(
                                "SELECT metadata FROM samples WHERE id = ?", (others[0],),
                            ).fetchone()[0])
                            if not isinstance(previous, dict):
                                raise ValueError("Invalid shared baseline metadata.")
                            existing = baseline_state(previous, identity)
                            automatic = metadata.get("baseline_label_source") == (
                                "deterministic-baseline-proposal"
                            )
                            if (
                                sample.baseline_label != "unknown"
                                and existing["baseline_label"] != "unknown"
                                and sample.baseline_label != existing["baseline_label"]
                                and not (existing["baseline_reviewed"] and automatic)
                            ):
                                raise ValueError("Conflicting labels for a shared baseline.")
                            state = existing
                        self._write_baseline_state(connection, members, state)
                    elif (
                        metadata.get("baseline_label", "unknown") != "unknown"
                        or metadata.get("baseline_id") is not None
                    ):
                        raise ValueError("Missing baseline cannot have a validity label.")
                identifiers.append(identifier)
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            pages = connection.execute("PRAGMA page_count").fetchone()[0]
            free_pages = connection.execute("PRAGMA freelist_count").fetchone()[0]
            if (pages - free_pages) * page_size > quota_bytes:
                raise OSError("Dataset quota reached; collection paused.")
        return identifiers

    def records(self, *, training_only: bool = False) -> list[StoredSample]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT s.id, kind, label, original_label, session, demo, "
                "metadata, reviewed, b.hash, m.hash FROM samples s "
                "LEFT JOIN sample_assets b ON b.sample=s.id AND b.role='baseline' "
                "LEFT JOIN sample_assets m ON m.sample=s.id AND m.role='baseline_valid' "
                "ORDER BY created, s.rowid"
            ).fetchall()
        result: list[StoredSample] = []
        for row in rows:
            metadata = json.loads(row[6])
            if (
                row[1] not in LABELS or row[2] not in LABELS[row[1]]
                or row[3] not in LABELS[row[1]] or not isinstance(metadata, dict)
            ):
                raise ValueError("Invalid stored sample schema or label.")
            if row[1] == "hand":
                if (row[8] is None) != (row[9] is None):
                    raise ValueError("Incomplete stored baseline assets.")
                if row[8] is not None:
                    metadata.update(baseline_state(
                        metadata, baseline_identity(row[8], row[9]),
                    ))
                elif (
                    metadata.get("baseline_label", "unknown") != "unknown"
                    or metadata.get("baseline_id") is not None
                ):
                    raise ValueError("Missing baseline has a validity label.")
            item = StoredSample(
                row[0], row[1], row[2], row[3], row[4], bool(row[5]),
                metadata, bool(row[7]),
            )
            if not training_only or any(item.training_targets.values()):
                result.append(item)
        return result

    def images(
        self, sample_id: str, *, roles: tuple[str, ...] | None = None,
    ) -> dict[str, ImageArray | Mask]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, assets.hash, png FROM sample_assets "
                "JOIN assets ON assets.hash = sample_assets.hash WHERE sample = ?",
                (sample_id,),
            ).fetchall()
        if roles is not None:
            rows = [row for row in rows if row[0] in roles]
            if {row[0] for row in rows} != set(roles):
                raise ValueError("Missing requested sample image assets.")
        if not rows:
            raise ValueError("Sample has no image assets.")
        result: dict[str, ImageArray | Mask] = {}
        for role, digest, data in rows:
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Corrupt image asset: {role}.")
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Unreadable image asset: {role}.")
            result[role] = np.asarray(image, dtype=np.uint8)
        return result

    def _baseline_members(
        self, connection: sqlite3.Connection, sample_id: str,
    ) -> tuple[str, ...]:
        rows = connection.execute(
            "SELECT other.sample FROM sample_assets b "
            "JOIN sample_assets m ON m.sample=b.sample AND m.role='baseline_valid' "
            "JOIN sample_assets other ON other.hash=b.hash AND other.role='baseline' "
            "JOIN sample_assets om ON om.sample=other.sample "
            "AND om.role='baseline_valid' AND om.hash=m.hash "
            "JOIN samples s ON s.id=other.sample AND s.kind='hand' "
            "WHERE b.sample=? AND b.role='baseline' ORDER BY other.sample",
            (sample_id,),
        ).fetchall()
        return tuple(row[0] for row in rows)

    def baseline_members(self, sample_id: str) -> tuple[str, ...]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT kind FROM samples WHERE id=?", (sample_id,),
            ).fetchone()
            if row is None or row[0] != "hand":
                raise ValueError("A hand observation is required.")
            return self._baseline_members(connection, sample_id)

    def _write_baseline_state(
        self, connection: sqlite3.Connection, members: tuple[str, ...],
        state: dict[str, object],
    ) -> None:
        for member in members:
            metadata = json.loads(connection.execute(
                "SELECT metadata FROM samples WHERE id=?", (member,),
            ).fetchone()[0])
            if not isinstance(metadata, dict):
                raise ValueError("Invalid shared baseline metadata.")
            metadata.update(state)
            connection.execute(
                "UPDATE samples SET metadata=? WHERE id=?",
                (json.dumps(metadata, allow_nan=False), member),
            )

    def review(
        self, sample_id: str, label: str, *, piece_colour: str | None = None,
        baseline_label: str | None = None,
        confirmed_baseline_members: tuple[str, ...] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT kind, metadata FROM samples WHERE id = ?", (sample_id,)
            ).fetchone()
            if row is None or label not in LABELS[row[0]]:
                raise ValueError("Missing sample or invalid review label.")
            metadata = json.loads(row[1])
            if not isinstance(metadata, dict):
                raise ValueError("Invalid sample metadata.")
            if baseline_label is not None:
                if row[0] != "hand" or baseline_label not in BASELINE_LABELS:
                    raise ValueError("Invalid baseline review label.")
                members = self._baseline_members(connection, sample_id)
                if not members:
                    raise ValueError("Cannot review a missing baseline.")
                hashes = connection.execute(
                    "SELECT hash FROM sample_assets WHERE sample=? "
                    "AND role IN ('baseline', 'baseline_valid') ORDER BY role",
                    (sample_id,),
                ).fetchall()
                state = baseline_state(
                    metadata, baseline_identity(hashes[0][0], hashes[1][0]),
                )
                changed = baseline_label != state["baseline_label"] or (
                    baseline_label != "unknown" and not state["baseline_reviewed"]
                )
                if changed:
                    if (
                        confirmed_baseline_members is not None
                        and members != confirmed_baseline_members
                        or len(members) > 1 and confirmed_baseline_members is None
                    ):
                        raise ValueError(
                            "Shared baseline links need confirmation; reload and confirm "
                            "the linked observations before saving."
                        )
                    history = state["baseline_revisions"]
                    if not isinstance(history, list):
                        raise ValueError("Invalid baseline revision history.")
                    history.append({
                        "previous": state["baseline_label"],
                        "previous_source": state["baseline_label_source"],
                        "label": baseline_label, "source": "reviewer",
                        "created": utc_now(), "observation": sample_id,
                    })
                    state.update({
                        "baseline_label": baseline_label,
                        "baseline_label_source": "reviewer", "baseline_reviewed": True,
                        "baseline_label_strength": "reviewed",
                    })
                    self._write_baseline_state(connection, members, state)
                    metadata.update(state)
            if row[0] == "hand":
                metadata["current_label_source"] = "reviewer"
                metadata["current_label_strength"] = "reviewed"
            if piece_colour is not None:
                if row[0] != "occupancy" or piece_colour not in PIECE_COLOURS:
                    raise ValueError("Invalid piece colour for this example.")
                if (
                    label == "empty" and piece_colour in ("white", "black")
                    or label == "occupied" and piece_colour == "none"
                ):
                    raise ValueError(
                        "Conflicting labels: empty needs none/unknown; "
                        "occupied needs white/black/unknown."
                    )
                if piece_colour != metadata.get("piece_colour", "unknown"):
                    history = metadata.setdefault("piece_colour_revisions", [])
                    if not isinstance(history, list):
                        raise ValueError("Invalid piece-colour revision history.")
                    history.append({
                        "previous": metadata.get("piece_colour"),
                        "previous_source": metadata.get("piece_colour_source"),
                        "label": piece_colour, "source": "reviewer",
                        "created": utc_now(),
                    })
                    metadata["piece_colour"] = piece_colour
                    metadata["piece_colour_source"] = "reviewer"
            connection.execute(
                "UPDATE samples SET label = ?, metadata = ?, reviewed = 1 WHERE id = ?",
                (label, json.dumps(metadata, allow_nan=False), sample_id),
            )
            connection.execute(
                "INSERT INTO revisions VALUES (?, ?, ?)", (sample_id, label, utc_now())
            )

    def summary(self) -> str:
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            pages = connection.execute("PRAGMA page_count").fetchone()[0]
            free_pages = connection.execute("PRAGMA freelist_count").fetchone()[0]
        return (
            f"{count} examples; {(pages - free_pages) * page_size / 1_000_000:.1f} MB "
            f"active DB pages ({self.path.stat().st_size / 1_000_000:.1f} MB file)."
        )

    def discard(
        self, *, sample_id: str | None = None,
        session_id: str | None = None, reason: str,
    ) -> int:
        if (sample_id is None) == (session_id is None):
            raise ValueError("Choose exactly one sample or session to discard.")
        field, value = ("id", sample_id) if sample_id else ("session", session_id)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT id FROM samples WHERE {field} = ?", (value,)
            ).fetchall()
            connection.executemany(
                "INSERT OR REPLACE INTO deletions VALUES (?, ?, ?)",
                [(row[0], reason, utc_now()) for row in rows],
            )
            connection.execute(f"DELETE FROM samples WHERE {field} = ?", (value,))
            connection.execute(
                "DELETE FROM assets WHERE hash NOT IN (SELECT hash FROM sample_assets)"
            )
        return len(rows)

    def export_manifest(self, destination: Path) -> None:
        if destination.resolve() == self.path:
            raise ValueError("The manifest must not overwrite the dataset.")
        records = self.records()
        with self._connect() as connection:
            assets = connection.execute(
                "SELECT sample, role, sample_assets.hash, length(png) "
                "FROM sample_assets JOIN assets ON assets.hash = sample_assets.hash"
            ).fetchall()
        hashes: dict[str, dict[str, object]] = {}
        for sample_id, role, digest, size in assets:
            hashes.setdefault(sample_id, {})[role] = {
                "sha256": digest, "png_bytes": size,
            }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            temporary.write_text(
                "\n".join(
                    json.dumps({
                        "schema_version": 2, "id": item.sample_id,
                        "kind": item.kind, "label": item.label,
                        "original_label": item.original_label,
                        "session": item.session_id, "demo": item.demo,
                        "reviewed": item.reviewed, "metadata": item.metadata,
                        "assets": hashes.get(item.sample_id, {}),
                        "baseline_id": item.baseline_id,
                        "baseline_present": item.baseline_id is not None,
                        "baseline_label": item.baseline_label if item.kind == "hand" else None,
                        "baseline_reviewed": item.baseline_reviewed,
                        "training_targets": item.training_targets,
                        "baseline_split_group": item.baseline_id,
                    }, allow_nan=False)
                    for item in records
                ) + "\n", encoding="utf-8",
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def demonstration_samples(
    frame: ImageArray, grid: FloatArray,
) -> tuple[NormalizedView, list[VisionSample]]:
    view = normalize_view(frame, grid)
    session = "demo-" + uuid4().hex
    source_hash = hashlib.sha256(frame.tobytes()).hexdigest()
    common: dict[str, object] = {
        "source": "recorded-image-demo", "source_hash": source_hash,
        "label_source": "unreviewed-demo", "training_excluded": True,
        "pixels_per_square": view.pixels_per_square,
        "transform": view.transform.tolist(),
        "camera_grid": grid.tolist(),
        "ring_coverage": ring_coverage(view),
        "baseline_compatible": True,
    }
    samples: list[VisionSample] = []
    for row, column in ((0, 0), (0, 7), (3, 3), (4, 4), (7, 0), (7, 7)):
        crop, valid = view.square(row, column)
        samples.append(VisionSample(
            "occupancy", "unknown", {"crop": crop, "valid": valid},
            {**common, "physical_square": [row, column],
             "parity": (row + column) % 2, "valid_fraction": coverage(valid)},
            session, True,
        ))
    baseline, valid = view.ring()
    current = baseline.copy()
    # Deliberately artificial, for exercising review rather than training.
    p = view.pixels_per_square
    cv2.rectangle(current, (5 * p, 0), (6 * p, 3 * p - 1), (80, 140, 190), -1)
    current[valid == 0] = 0
    for image, label, synthetic in (
        (baseline, "clear", False), (current, "crossing", True),
    ):
        samples.append(VisionSample(
            "hand", label,
            {"baseline": baseline, "current": image,
             "baseline_valid": valid, "current_valid": valid},
            {**common, "synthetic_overlay": synthetic,
             "label_source": "demo-proposal-not-ground-truth"},
            session, True,
        ))
    return view, samples
