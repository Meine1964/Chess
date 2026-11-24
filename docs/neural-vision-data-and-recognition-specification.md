# Neural Vision Data and Recognition Specification

**Version:** 0.12
**Status:** Data foundation implemented and automated-test validated.
Neural training/inference, the shadow-mode contract, conditional test selection,
and live hardware acceptance remain pending implementation or validation.
**Platform:** Python on Windows, with optional GPU-accelerated training and
inference

**Related specifications:**

- [Chess-square occupancy and UI marking](./chess-square-occupancy-ui-specification.md)
- [Reference-image move tracking](./reference-image-move-tracking-specification.md)
- [Projector/camera chessboard system](./projector-camera-chessboard-specification.md)
- [Piece-aware board localization](./piece-aware-board-localization-specification.md)
- [Move-detection performance](./move-detection-performance-specification.md)

## Initial Data-Foundation Implementation Contract

The approved first implementation is limited to the normalized preview, opt-in
data collection, image-and-label review for both datasets, and recorded-image
demonstration examples. Neural training, inference, model activation, automatic
retraining, advanced dataset splitting, and model-release management remain
later milestones.

The following restrictions apply:

- Existing gameplay decisions, detector thresholds, calibration, projection,
  move acceptance, and timing policies shall remain unchanged. Collection
  observers shall never provide input to those decisions.
- Put normalization, persistence, asynchronous collection, and review in new
  modules. Existing application changes shall be limited to Tools-menu wiring,
  read-only snapshots, lifecycle invalidation, and post-decision notifications.
- Collection defaults to off for every application run. Opening the preview or
  reviewer shall not start collection or calibration, modify saved game state,
  or cause startup dialogs.
- No new ML dependencies, downloads, uploads, GPU setup, settings migration, or
  camera/projector access shall be required to review an existing dataset.
- Use a selected-directory SQLite dataset with schema version, atomic sample
  and image commits, content hashes, deduplicated image assets, and label
  revision history. Reopening shall verify dataset identity; missing external
  storage shall never silently create a replacement dataset.
- Encoding, normalized collection crops, and persistence shall run on a bounded
  worker queue. Failure or overflow shall pause collection visibly, not delay
  move verification. Tools windows shall be modeless.
- Without a trustworthy independent verification checkpoint, correction
  invalidation may conservatively discard the current collection session's
  occupancy samples. It shall never claim an inferred checkpoint is verified.
- The initial live hand collector may use black-board gate observations and
  clean-baseline updates; projected-guidance and scripted stationary-hand
  collection remain explicitly separate coverage limitations.
- Recorded-image examples shall be marked demonstration-only, have honest
  proposed/unknown labels, and be excluded from default training selection.
  Artificial crossing overlays shall be identified as synthetic, not real hand
  evidence. Source screenshots and calibration files shall remain read-only.
- Add **Tools > Vision data...** and **Tools > Review vision labels...**.
  The former provides normalized preview, explicit dataset selection,
  start/pause, quota, status, and recorded-image example generation; the latter
  provides label acceptance/correction, skip, discard, and both image types.
- Keep datasets outside the application's diagnostic-retention directory;
  diagnostic cleanup must never delete collected training data.
- Automated tests shall cover geometric extraction, delayed commitment,
  invalidation, quota/failure isolation, persisted review/discard, demo
  exclusion, and menu integration without physical hardware. Record
  hardware-dependent validation as pending rather than claiming live success.

### Using the data foundation without hardware

1. Open **Tools > Review vision labels...**, choose **Open dataset...**, and
   select the directory `vision-data\demo` when using the generated
   demonstration set, not the `vision.sqlite3` file inside it. After a successful
   selection the tools remember this directory in local application settings
   across restarts. Opening the reviewer restores the last dataset; the dataset
   chooser starts there. On first use no dataset is discovered automatically.
2. Select Occupancy or Hand. The target occupancy square is outlined in yellow;
   one hand observation shows its linked baseline on the left and current image
   on the right, with captions and a shared observation identity.
3. Use dropdowns to choose the occupancy/hand label and, for occupancy, piece
   colour. Hand examples have independent **Current obstruction** and
   **Baseline validity** dropdowns. Shared baseline changes request confirmation
   with the linked count. **Save and next (Enter)** saves applicable labels
   together. **1**, **2**, or **3**
   saves the corresponding occupancy/hand label with the selected piece
   colour. **Right/Left** navigates without saving; **Delete** discards.
   Corrections persist while original labels and colour provenance remain
   available.
   **Previous/Next** (Left/Right) stop at the ends of the filtered list and
   never wrap. At the final example the save button reads **Save (Enter)** and
   saves without advancing.
4. Demo examples remain excluded from default training selection even after
   review. The crossing overlay is deliberately synthetic.
5. **Tools > Vision data...** provides full-board, hand-ring, and selected-square
   preview modes. Load a recorded camera image and matching grid JSON. This
   never starts live collection.

The initially generated local demo directory contains six occupancy examples
and two paired hand examples. Its images are stored inside `vision.sqlite3`,
not as individual crop files. Normalized previews and reviewer screenshots
alongside the database are illustrations, not additional review records.

To generate another demonstration dataset from a recorded image in the project
root, run:

```powershell
.venv\Scripts\python.exe -m scripts.generate_vision_examples --image 'diagnostics\runtime-camera.png' --detect-grid --output 'vision-data\demo'
```

Alternatively supply `--grid 'calibrations\latest.json'` instead of
`--detect-grid` only when that calibration actually matches the image.
Detection runs only on the supplied recorded image and never updates saved
calibration. The command writes a sanitized `recorded-grid.json` that can be
loaded with that same image in the preview, plus normalized image/mask previews.
Repeating generation appends new demo examples; it does not overwrite existing
labels. `vision-data` is ignored by Git.

### Current implementation boundaries

- The new modules are `vision_data.py` (normalization and dataset),
  `vision_collection.py` (worker and delayed commit), and `vision_tools.py`
  (modeless tools and passive notifications).
- Collection starts only through the explicit button, requires a verified live
  grid, and defaults off on every launch. The operator selects an anonymous
  setup alias and phase before starting a collection session.
- Initial validity thresholds are 99% observed pixels per occupancy crop and
  99% per evaluated current hand edge. Insufficient current framing pauses
  collection explicitly. A baseline also needs 99% per evaluated edge to be
  included; insufficient baseline coverage is reported and the valid current
  observation is retained without it. Demo data may illustrate invalid pixels
  without claiming training eligibility.
- Ordinary accepted positions select changed squares plus two unchanged empty
  and two unchanged occupied squares. Every fiftieth accepted position prepares
  a complete 64-square snapshot, also subject to next-move confirmation.
- Live hand collection retains clean-baseline and crossing/clearing transitions
  on the black board, not continuous video. A deterministic five-second
  stillness release is labeled `unknown`, not assumed clear.
- Quota counts active SQLite pages including indexes/metadata. Deletion frees
  pages for reuse but need not shrink the file immediately; status distinguishes
  active page use from allocated file size. Automatic vacuuming and redundant
  near-duplicate pruning are not implemented.
- Known correction/reset/reconciliation events conservatively discard affected
  collection sessions; no independently verified checkpoint is inferred.
  Deletion intent is journaled on local storage, separately from the external
  dataset, and replayed before further collection. Short local control-journal
  updates are the only synchronous persistence in a correction notification;
  image processing and external-drive deletion remain asynchronous.
- Dataset review, explicit sample/session deletion, manifest export, PNG asset
  hashing, and exact-asset deduplication are implemented. Setup-level split
  tooling, automatic review prioritization, training snapshots/caches,
  cross-process game-history linkage, trained-model impact tracking, backups,
  and all neural training/inference are deferred. No training entry point may
  be added later without honoring outstanding deletion intents.
- The reviewer uses dropdowns for the primary label and occupancy piece colour,
  saving both atomically with colour revision provenance. Its semantic summary
  and controls fit at 960x640; raw metadata opens in a separate read-only
  window. Titled groups distinguish read-only model inputs from editable
  output/training labels, with only hand-relevant fields for hand examples.
  Projector state and colour provenance remain in technical details. Missing
  recorded input facts remain explicit rather than being inferred.
- Both dataset choosers remember the last selected directory through local
  application settings. Invoking the reviewer restores it when no explicit
  dataset is supplied; unavailable storage is reported without recreation or
  starting collection.
- Hand storage/review supports independent current-obstruction and shared
  baseline-validity labels, optional references, explicit preview captions,
  shared-revision confirmation, and per-output training masks. Existing
  records without a baseline-validity label mean unknown validity, not an
  implicitly verified-clear baseline. Opening them does not migrate or rewrite
  their metadata.
- The redundant **Images** row is absent for both datasets. Hand examples hide
  the now-empty input-metadata group; the actual linked image previews remain.
  Previous/Next buttons and keyboard navigation stop at filtered-list
  boundaries, with unavailable directions disabled. Saving the final example
  does not wrap to the first.
- A paused writer can still process correction deletions. The final pending
  position and unprocessed captures are discarded on stop/shutdown rather than
  blocking gameplay to flush them.
- Hardware-free tests and recorded-image rendering do not establish live camera,
  projector, external-HDD throughput, or real-time latency acceptance. Those
  checks remain required when the hardware is available.

### Validation of the initial implementation

- The repository quality entry point,
  `.venv\Scripts\python.exe scripts\run_quality.py`, passed compilation and all
  605 tests.
- Focused vision/menu validation passed 28 tests, including delayed commitment,
  correction after pause, unavailable-drive deletion replay, image geometry,
  incomplete-halo rejection, persisted edits for both datasets, keyboard
  controls, demo exclusion, and unchanged menu commands.
- Recorded-image normalization and both reviewer layouts were rendered and
  visually checked without camera/projector access. Demo geometry was detected
  on the source image itself after discovering that the latest calibration and
  latest screenshot did not match.
- Live gameplay with collection enabled, external-HDD performance, actual
  disconnect/reconnect behavior on that HDD, and GPU/CPU model performance
  remain hardware acceptance tasks. No neural model is trained or activated.

### Validation of the label-review refinement

- Focused validation of `tests/test_vision_data.py` and
  `tests/test_vision_tools.py` passed all 39 tests after the final code change.
  Coverage includes atomic colour/primary-label corrections, preserved
  provenance, invalid-combination rejection, unsaved drafts, failed writes,
  both layouts at 960x640, and separate technical details.
- Both reviewer layouts were rendered from the existing local recorded demo
  dataset at 960x640 and visually inspected. The deliberately excluded centre
  of a hand ring does not count as missing camera coverage; focused tests
  distinguish that exclusion from missing edge pixels.
- Pylance reported no errors in the changed implementation modules and found
  all seven resolved review call sites compatible with the extended signature.
  No new dependencies, dataset migration, gameplay changes, or demo-label edits
  were required. The earlier full-suite result above belongs to the initial
  implementation; this refinement used focused validation, not a new full run.

### Validation of remembered dataset locations

- All 16 tests in `tests/test_vision_tools.py` passed after the final change,
  using isolated INI settings rather than the operator's application settings.
- Tests cover reopening through a fresh settings instance, explicit-dataset
  precedence, initial chooser directories, shared collection/review selection,
  cancellation and invalid paths, unavailable storage without recreation,
  settings-write error reporting, and collection remaining off.
- Pylance reported no errors in `vision_tools.py`; all 16 resolved reviewer
  constructor call sites remain compatible. Gameplay and dataset formats are
  unchanged. No full-suite or live-hardware validation was needed for this
  settings-only refinement.

### Validation of model input/output presentation

- All 17 tests in `tests/test_vision_tools.py` passed after the final UI change.
  Tests verify prefilled stored inputs/targets in the correct groups, read-only
  input fields, hand-specific visibility, unchanged label saving, and retained
  provenance/projector metadata in technical details.
- Both layouts were rendered from the recorded demo dataset and visually
  inspected at 960x640 with all relevant labels visible without scrolling.
  Pylance reported no errors or syntax issues in `vision_tools.py`.
- This change implements presentation only. The joint three-class recognition
  contract is specified, not trained or integrated into gameplay. No existing
  demo labels or dataset formats were changed.

### Validation of bounded reviewer navigation

- All 20 tests in `tests/test_vision_tools.py` passed after removing the Images
  row and replacing circular navigation with bounded selection.
- Coverage includes both filtered datasets, button states, Left/Right
  shortcuts, draft preservation at boundaries, saving the final/single
  example, empty lists, deletion of the last example, and 960x640 field layout.
- Pylance reported no errors or syntax issues in `vision_tools.py`. No model,
  dataset-format, or gameplay changes were made.

### Validation of completed hand-observation review/data support

- After the final implementation change, all 60 tests in
  `tests/test_vision_data.py` and `tests/test_vision_tools.py` passed.
- Focused cases cover independent outputs, atomic shared-baseline edits and
  confirmation cancellation, stale link-set rejection, retained current labels,
  reviewed-baseline inheritance, legacy read compatibility without writes,
  missing/corrupt references, current-only collection after geometry changes
  or insufficient reference coverage, and version-2 export eligibility.
- Unreviewed current labels dependent on a disputed deterministic reference
  are retained but masked until reviewed; independently reviewed current labels
  remain eligible. Invalid baseline labels are retained as useful validity
  targets, not quarantined or automatically discarded.
- Square, paired-hand, and missing-baseline layouts were rendered from recorded
  data and visually inspected at 960x640. Existing demo labels were not changed.
  Baseline reuse counts, captions, both output fields, and bounded navigation
  remain visible without scrolling.
- Pylance reported no errors in all three implementation modules, no syntax
  issues, and compatible resolved calls for the extended image/review APIs.
  SQLite schema remains version 1; manifests use version 2. No dependencies
  or gameplay code were changed in this refinement.
- The required pre-push entry point,
  `.venv\Scripts\python.exe scripts\run_quality.py`, passed compilation and all
  638 tests after the final implementation change.
- Neural training/inference, automated split construction, live collection
  acceptance, and external-HDD performance remain pending. No models are
  trained or activated.

## 1. Purpose

Define a normalized camera view, an opt-in training-data collection pipeline,
and two independent neural vision models:

1. A square-recognition model that jointly predicts occupancy and piece colour
   for one physical square.
2. A board-edge hand model that decides whether a hand, arm, or other temporary
   obstruction crosses the playing-board boundary.

The two models solve different problems and shall not share an output head or
runtime decision. They may use the same normalized-view implementation and
dataset infrastructure.

The initial work shall collect trustworthy, reviewable data before replacing
any existing detector. A neural model shall enter normal operation only after
offline evaluation, recorded-game replay, and live shadow-mode comparison show
that it satisfies the acceptance criteria in this specification.

## 2. Goals

This work shall:

1. Produce a top-down normalized view containing the whole 8x8 board and a
   two-square exterior halo on all four sides.
2. Save compact, versioned training examples rather than unrestricted camera
   surroundings.
3. Derive square-occupancy labels from settled positions, distinguishing
   implicit player acceptance from independent verification.
4. Preserve label provenance and uncertainty instead of converting every
   inference into ground truth.
5. Collect hard negative examples, not only squares that changed.
6. Recognize current board-edge obstruction independently, assess baseline
   validity separately, and use compatible baseline comparison as additional
   evidence rather than a prerequisite for every hand prediction.
7. Support explicit review of ambiguous, novel, and disagreement examples.
8. Prevent near-duplicate frames from leaking across training, validation, and
   test sets.
9. Train on an available GPU while retaining a defined CPU inference path.
10. Preserve the existing deterministic detectors as fallback and comparison
    paths until neural behavior is proven.

## 3. Non-Goals

The first version does not require:

- Piece-type recognition.
- Reconstruction of an arbitrary chess position from occupancy alone.
- Replacing legal-move validation.
- Using inferred piece colour as unquestioned ground truth.
- Identifying a person whose hand appears in a frame.
- Saving the room or unrestricted image content outside the normalized halo.
- Training continuously during a game.
- Uploading images, metadata, or models to a remote service.
- Replacing calibration, board localization, or orientation ownership.
- A single model that combines occupancy and hand detection.

## 4. Authoritative State and Safety Boundaries

Board geometry remains owned by the existing calibration and localization
subsystems. The neural pipeline shall consume a verified camera grid and shall
not modify it.

The logical chess position remains authoritative for game state. Neural
occupancy is evidence used by initialization, reset, reconciliation, and
diagnostics; it shall not independently invent piece identities or bypass
legal-move validation.

Hand detection is a safety gate. An unavailable model, invalid input, model
load failure, or low-confidence result shall not be interpreted as "no hand."
Before neural-only operation is approved, the existing deterministic
board-edge gate remains authoritative. During shadow mode, neural output is
diagnostic only.

## 5. Coordinate Systems and Terminology

### 5.1 Physical board coordinates

Physical rows and columns use the calibrated camera-grid ordering:

- Rows `0` through `7`.
- Columns `0` through `7`.
- Grid intersections `0` through `8` on each axis.

Dataset records shall also store the mapped algebraic square and current UI
orientation for diagnostics. Those values shall not replace the physical
row/column key.

### 5.2 Square unit

One normalized square unit is the side length of one physical chess square.
All crops and margins shall be defined first in square units and only then
converted to pixels.

### 5.3 Normalized canvas

The normalized canvas covers coordinates `[-2, 10] x [-2, 10]` in square
units:

- The physical board occupies `[0, 8] x [0, 8]`.
- A two-square exterior halo occupies the remaining area.
- The complete canvas is therefore 12x12 square units.

The canonical camera-grid orientation shall be retained in stored source
images. UI rotation may be applied to a preview, but shall be represented as a
view transform rather than silently changing dataset coordinates.

## 6. Normalized 12x12 Camera View

### 6.1 Geometry

The existing four outer board corners define a projective transform from the
camera image to the normalized board plane. The same homography shall be
extrapolated by two square widths on every side to produce the 12x12 canvas.
The extrapolation does not claim that the physical surface outside the board is
coplanar; it provides a consistent board-relative image coordinate system.

The initial recommended resolution is 1200x1200 pixels, or 100 pixels per
square unit. Resolution shall be configurable as a pixels-per-square value,
and every stored record shall declare the actual value.

The warp shall preserve colour and use one documented interpolation and border
policy. Pixels whose source coordinates are outside the camera frame shall be
marked invalid by a separate binary validity mask; synthetic border pixels
shall never be treated as observed image content.

### 6.2 Framing validation

Before collection, the application shall calculate how much of the requested
halo is visible on each side. A collection record shall contain the visible
halo width and valid-pixel fraction for each edge.

An occupancy crop may be retained only when its required pixels meet the
configured validity threshold. A hand-ring example requires usable coverage of
all evaluated edges; otherwise it shall be rejected or explicitly labeled
`insufficient-framing`, never silently resized from incomplete content.

The existing requirement that the camera see a two-square board-edge band
therefore becomes a measurable collection precondition.

### 6.3 Preview

The normalized view shall be available as an optional diagnostic preview
independently of neural-model enablement. It shall show:

- The complete 12x12 canvas.
- The 8x8 board boundary.
- Square coordinates in physical or selected UI orientation.
- Invalid or camera-clipped areas.
- The occupancy crop for a selected square.
- The retained hand-model ring and its masked regions.

The preview shall not itself cause samples to be stored.

## 7. Occupancy Examples

### 7.1 Visual input

For a target square, the saved visual crop shall contain:

- The complete target square.
- A margin of half a square on every side.
- A fixed 2x2-square extent centered on the target square.

At a board edge, the margin naturally uses the normalized exterior halo rather
than padding from a neighboring board location. The crop shall include a
validity-mask channel or paired mask file so the model can distinguish
camera-observed pixels from unavailable pixels.

The initial recommended image size is 200x200 pixels at 100 pixels per square.
Training may resample this image, but the source dataset should retain the
declared normalized resolution.

### 7.2 What to collect

For every accepted new position, the collector shall prepare the following
samples, subject to the delayed commit rule in Section 7.3:

1. Every square whose authoritative occupancy or piece assignment changed from
   the preceding accepted position. This includes an occupied capture
   destination or promotion square even when its binary occupancy is unchanged.
2. A configurable sample of unchanged occupied squares.
3. A configurable sample of unchanged empty squares.
4. Hard negatives or disagreements selected by the current deterministic and
   shadow neural detectors.
5. Periodic complete 64-square snapshots at a much lower frequency.

Collecting only changed squares would overrepresent move origins and
destinations, underrepresent stable edge cases, and provide too few examples of
ordinary empty squares. Sampling shall keep empty and occupied examples, light
and dark squares, board edges, corners, piece colours, and physical setups
measurably represented.

Multiple frames from one settling burst may be retained for robustness
research, but they shall share one correlation-group identifier and shall not
be counted as independent positions in dataset statistics.

The initial sampling policy shall retain all changed squares plus two
unchanged empty squares and two unchanged occupied squares per accepted
position, where available. Unchanged-square selection shall rotate across the
board and prioritize underrepresented square colours, edges, corners, and
shadow conditions. Exact duplicates shall be referenced rather than repeatedly
stored solely to increase sample counts.

An ordinary non-capturing move therefore produces six post-move crops: three
empty and three occupied. Captures, special moves, sparse positions, additional
hard examples, and complete snapshots change this count. Training batches
should initially balance empty and occupied examples approximately 1:1;
collection shall not discard valuable rare examples merely to enforce that
ratio. Evaluation on complete positions shall retain their natural class
proportions, including empty-heavy endgames.

### 7.3 Capture point

An occupancy example shall be captured only from a frame that:

- Uses a verified current camera grid.
- Is stable and accepted by the existing motion/obstruction policy.
- Has no calibration or alignment pattern projected.
- Has a declared projector state and illumination level.
- Corresponds to a position whose label provenance is known.
- Passes crop-validity checks.

The preferred records are the last clean accepted frame before a physical move
and the first clean settled frame after that move. These linked records provide
before/after evidence without saving frames containing the moving hand.

Capture and dataset commitment are separate events. The collector shall hold
the latest eligible settled position, its crops, labels, and provenance in a
bounded in-memory pending buffer. These pending assets shall not yet be written
as training examples.

When the application accepts the next physical move in the same game and
uninterrupted position history, commit the preceding pending position. Record
the confirming move identifier and acceptance time. Mere hand activity, a
candidate move, or a server update without accepted physical execution shall
not trigger commitment. Prepare the newly accepted position as the next pending
record only if it meets the capture requirements.

This treats continued play as implicit player acceptance of the preceding
position, not independent proof of correct recognition. The final pending
position shall be discarded when the game ends, collection stops, or the
application exits. A crash may lose pending examples; it shall not promote
them into the dataset. A correction, takeback, history discontinuity, or
calibration invalidation shall discard the pending record rather than attach it
to an unrelated subsequent move.

Hand examples have their own collection and label rules; acceptance of a later
move shall not automatically establish a crossing label.

### 7.4 Labels

The primary label is one of:

- `empty`
- `occupied`
- `unknown`

`unknown` records may be kept for later review but shall not enter supervised
empty/occupied training.

Strong labels may be produced from:

- A position independently confirmed to match the physical board, not merely
  inferred to match because a legal move was accepted.
- A successful reconciliation proving that physical occupancy matches the
  authoritative position.
- A human-reviewed correction.

Weak labels include:

- `implicit-player-acceptance`: a settled position committed after the next
  accepted physical move. This is the default practical gameplay label source.
- Existing deterministic occupancy output without authoritative-position
  confirmation.
- An inferred piece colour.
- A candidate move that was not accepted.
- A frame associated with ambiguous compound physical handling.

Weak labels shall be identified as such and shall not be silently promoted to
strong labels. Training configuration shall explicitly choose whether a weak
label source is included and with what weight.

Initial training may include `implicit-player-acceptance` labels without
requiring review of every example. Its use shall be explicit in the training
configuration and metrics. The image-and-label review tool shall also support
occupancy corrections. Independent evaluation labels shall be reviewed or
otherwise independently verified; continued play alone is not sufficient.

### 7.5 Discarding examples after a gameplay correction

When reconciliation, a user correction, or another authoritative history
repair casts doubt on collected occupancy labels:

1. Discard the pending occupancy record.
2. Exclude from training immediately and delete the stored occupancy examples
   after the last independently verified position and up to the correction
   event. If there is no such checkpoint in the game, discard all occupancy
   examples collected for that game so far.
3. Preserve only minimal deletion provenance: affected sample identifiers,
   interval, reason, and affected training-run identifiers. There shall be no
   image quarantine, restoration queue, or mandatory review of this interval.
4. Preserve shared assets only where another valid record still requires them.
   Apply exclusions to registered training snapshots and caches as well as the
   live dataset; no later run may reintroduce a deleted example from a stale
   manifest or backup.
5. Flag models trained on affected examples for reassessment. Deleting source
   images does not undo their effect on weights; subsequent clean training and
   the normal promotion gates are required before claiming that influence is
   removed. Never replace or reset the operational model mid-game merely
   because examples were discarded.

A checkpoint requires independent confirmation of the physical position;
accepting another model-inferred move shall not advance it. The correction
result starts a new checkpoint only if it independently verifies the physical
position. Correction-triggered deletion overrides routine retention protection;
if evaluation examples are affected, invalidate and version the affected
evaluation set and results rather than continuing to rely on them.

If storage is unavailable, record the deletion intent in local control state,
exclude those identifiers from every training selection, and report physical
deletion as pending until the drive is available. Report completion only after
the affected dataset assets have actually been removed.

Correction events shall be consumable when their application workflows exist;
this specification does not assume that every desired correction UI is already
implemented. Hand labels that depend on the invalidated interval shall also be
discarded when no trustworthy independent label remains. A bad baseline alone
does not invalidate an independently reviewed current-image label; retain
usable per-output evidence under Section 9.5. Unrelated hand examples need not
be deleted.

### 7.6 Piece colour

Piece colour is a required output target of the same model that predicts
occupancy, not an optional auxiliary experiment. Stored review values are:

- `white`
- `black`
- `none`
- `unknown`

Prefill occupancy and piece colour from the accepted logical position matching
the capture: an empty square gives `empty`/`none`; a piece gives
`occupied`/`white` or `occupied`/`black`. Preserve the existing delayed-confirmation
and correction/discard rules. Store provenance and available confidence
separately; inferred labels are not independently verified ground truth.
Neither logical occupancy nor logical piece colour shall be a model input.

Record light/dark square colour from the verified logical-to-physical mapping
and location from physical coordinates. For historical/demo images without a
matching position and orientation, explicitly show missing labels rather than
using the application's present position. Such examples require matching
capture context or manual annotation before they can supply those targets.

## 8. Occupancy Metadata

Metadata shall be separated into three categories.

### 8.1 Required model inputs and optional extensions

Only values reliably available at runtime and not derived from the desired
occupancy or piece-colour answer may be model inputs. The agreed square model
shall receive:

- The square image and its validity mask.
- Known light/dark square colour.
- Location: interior, edge, or corner and which physical edge(s) are adjacent,
  using a versioned numeric encoding rather than display strings.

These are inputs, whereas occupancy and piece colour are output/training labels.
Additional candidates, requiring measured benefit, are:

- Distances to rank/file edges and valid-pixel fraction.
- Local projective geometry features described below.
- Camera exposure, gain, white-balance mode, and projector illumination state
  when those values are available and stable across supported cameras.

Train an image-only ablation for comparison, not as a redefinition of the
agreed model contract. Report held-out-setup results to detect memorization of
equipment or location. Do not silently substitute guessed square colour when
required metadata is missing; the future trainer/runtime must reject or mark
that input unavailable under its versioned validation policy.

### 8.2 Geometry metadata

Useful local geometry includes:

- The four source-image corners of the square.
- Projected square area in camera pixels.
- Four projected edge lengths.
- Opposite-edge length ratios.
- Diagonal lengths and ratio.
- Local homography Jacobian or equivalent horizontal/vertical scale and skew.
- Distance from the square center to the camera-image center, normalized by
  image dimensions.
- Valid halo width on each side of the board.

These values capture foreshortening without requiring a physical square size.
A metric camera-to-board angle requires camera intrinsics and a board-plane
pose. Physical square size is useful for metric distance but is not by itself
sufficient to recover a trustworthy angle.

Projector-to-board angle is not a primary occupancy feature. Projector
brightness, colour, pattern state, and whether the board is black are more
directly relevant to what the camera observes. If projector pose later proves
useful, it shall be added as versioned optional metadata rather than assumed to
be known.

### 8.3 Audit-only metadata

The following fields support tracing, balancing, and dataset splitting but
shall not be model inputs by default:

- Dataset schema version and collector software version.
- Sample, game, move, position, capture-burst, and session identifiers.
- Anonymous user-declared board, piece-set, camera-setup, projector-setup, and
  room/lighting setup identifiers.
- Calibration identifier and normalized-view transform.
- Camera device identifier hash, capture resolution, and frame sequence.
- Timestamp and elapsed game time.
- Board orientation and mapped algebraic square.
- Before/after role for a linked move.
- Authoritative position hash or FEN when permitted by the storage setting.
- Label, label strength, source, confidence, and review history.
- Deterministic-detector result and score.
- Neural model version, output, confidence, and disagreement reason.
- Motion, settling, obstruction, alignment, and projector states.

Identifiers intended only for grouping or audit shall not be fed to a model;
otherwise it could memorize a board, game, camera, or location.

## 9. Board-Edge Hand Examples

### 9.1 Visual input

A hand observation is anchored to one current image and normally links:

1. A baseline reference, intended to be clear of temporary obstructions.
2. The current normalized view being labelled.

A baseline can be incorrectly accepted, missing, or incompatible. The model
shall support current-image-only recognition; do not make current obstruction
depend on baseline availability or correctness. When comparison is used, both
images shall have compatible verified camera-grid geometry. The retained
region in each available image is a ring consisting of:

- The outermost one-square band inside the 8x8 board.
- The two-square exterior halo.

The central 6x6 board region shall be masked, and image content beyond the
two-square halo shall not be included. The corresponding validity masks shall
be supplied. This region preserves the visual continuity of an arm across the
board boundary while limiting unrelated board and room content.

Keep current and baseline assets/masks separately linked, with observation and
baseline identities, timestamps, and geometry. Several current observations
may share one baseline. Use an explicit availability indicator for a missing
baseline; an ordinary black image shall never masquerade as a real baseline.
Separate current and baseline encodings shall permit independent outputs;
difference or fused features are optional additional comparison evidence.

### 9.2 Label definition

There are two independently reviewed output labels.

**Current obstruction**, describing the current image rather than the
difference from the baseline:

- `clear`: no temporary obstruction crosses a board boundary.
- `crossing`: a hand, arm, sleeve, or other temporary obstruction crosses a
  board boundary and can invalidate move observation.
- `unknown`: evidence is insufficient or disputed.

The model is not a generic hand detector. A hand visible only outside the
board, a board-confined chess piece, and a person elsewhere in the halo are not
positive unless they cross the board boundary in a way that can contaminate
board observation.

**Baseline validity**, describing the baseline image's obstruction content:

- `valid`: the monitored boundary is clear of temporary obstructions and the
  image is a candidate reference; chess pieces may still occupy the board.
- `invalid`: a hand, arm, sleeve, or other relevant temporary obstruction
  contaminates the reference.
- `unknown`: validity is unreviewed, disputed, or inconclusive.

A missing/unreadable baseline is unavailable, not a learned `invalid` example.
Geometry mismatch, age, or projection incompatibility can prevent comparison
even when the baseline's content label is `valid`; validate these separately.
An invalid baseline and a clear current image are a legitimate combination,
as are an invalid baseline and a crossing current image. Neither output shall
be inferred from the other.

Optional review tags may distinguish:

- Hand/arm/sleeve.
- Deliberately placed non-hand object.
- Edge piece or captured piece.
- Piece shadow.
- Projected light or clock content.
- Camera/grid movement.
- Framing failure.
- Other obstruction.

These tags support error analysis and need not be separate runtime classes.

### 9.3 Label sources

Hand labels shall use mixed automatic and review-based collection:

- Candidate current `clear` and baseline `valid` labels may come from accepted
  settled observations and baseline acquisition. Passing an existing gate
  alone is weak evidence, not independent proof.
- Candidate `crossing` labels may be generated from deterministic gate
  activation, physical-move activity phases, or a user correction.
- Automatically generated positives remain weak until their source proves the
  boundary-crossing definition or a reviewer confirms them.
- Disagreements, low-confidence examples, novel setups, and periodic random
  samples shall enter a review queue.

The deterministic gate's output shall not be copied as unquestioned neural
ground truth; doing so would teach the model to reproduce its errors.
Store each output's source, confidence when available, original label, and
review revisions separately. A reviewer may confidently label one output
while leaving the other unknown.

### 9.4 Hand metadata

Potential model inputs are:

- Valid-pixel masks and visible halo width per edge.
- Local board-boundary geometry in the source camera image.
- Camera exposure, gain, and white-balance state.
- Projector state, illumination intensity, and active overlay kind.
- Time delta between baseline and current image.
- Whether baseline and current use the same camera grid and calibration.
- Baseline age and the event that established it.

Baseline-specific features belong only to baseline assessment or the optional
comparison branch, not the independent current-image obstruction head.
Useful audit-only metadata includes deterministic connected-component
measurements, tracker state, move phase, active edge, stillness duration,
review tags, and model/detector disagreement.

Camera-to-board viewing geometry may be useful. Projector-to-board angle is
secondary to the directly observed projector state and should not be required
for the initial model.

### 9.5 Retention and per-output training eligibility

Baseline invalidity is a reference-quality signal, not a reason to discard an
otherwise useful observation. In particular:

- Retain reviewed contaminated baselines as negative examples for baseline
  validity. These have known labels and are not an image quarantine.
- Retain correctly reviewed current-image labels regardless of whether their
  linked baseline is valid, invalid, unknown, or absent.
- Exclude invalid/incompatible references from a training objective that
  assumes a valid clean baseline, not from all training objectives.
- If a current label came only from a detector using a now-disputed baseline,
  do not call it independently reliable; review it or leave that target unknown.
- Use per-output target masks: an unknown target contributes no supervised
  loss for that output, without suppressing a known target on the other output.
- Discard genuinely ambiguous/unusable images when no useful trustworthy
  target remains. Missing/corrupt pixels do not become usable solely because
  another target has a label.

Baseline content labels shall be associated with the shared baseline identity
and exposed consistently to its linked observations. Correcting baseline
validity shall not rewrite current-image labels or delete all linked pairs.
Preserve provenance and reference-safe deletion. Keep observations sharing a
baseline within one split and control repeated-baseline sampling so reuse does
not inflate baseline-validity metrics or dominate its training loss.

Schema-1 storage now carries versioned baseline metadata and independent review
history; records and version-2 manifests expose per-output target masks.
Training/split generation remains deferred. Eligibility flags describe usable
labels, not proof of asset integrity: a future trainer must verify required
image bytes/masks and apply its provenance policy for each objective. A
baseline-dependent, unreviewed deterministic current label is masked when its
baseline is marked invalid, without rewriting that label; independently
reviewing it restores its current-target eligibility.

## 10. Dataset Storage and Consent

### 10.1 Opt-in collection

Training-data collection shall be disabled by default. The user shall
explicitly enable it and shall be able to:

- Select the storage directory.
- Set a disk quota or retention limit.
- Pause collection immediately.
- See current storage use and retained-example counts.
- Delete a named collection session.
- Export a manifest without enabling model training.

The normalized diagnostic preview remains available when collection is off.

### 10.2 Privacy and minimization

The collector shall store normalized crops and masks, not unrestricted camera
frames, unless a separate diagnostic action explicitly requests a full frame.
The hand dataset shall mask the central board and discard content beyond the
two-square halo before writing the sample.

No network upload is part of this specification. Paths, device serial numbers,
account identifiers, player names, Lichess tokens, and other secrets shall not
enter manifests. Stable equipment identifiers shall be locally generated
aliases or hashes that do not expose raw serial numbers.

### 10.3 Record integrity

Each example shall have:

- An immutable sample identifier.
- A schema version.
- Image and mask dimensions.
- Content hashes for stored files.
- A manifest record written atomically with its assets.
- A label-revision history rather than destructive relabeling.

Interrupted writes shall be detected, excluded, and cleaned up with an explicit
diagnostic rather than placed in an image quarantine. Missing assets, hash
mismatches, unsupported schema versions, and incomplete metadata shall be
reported explicitly and excluded from training.

### 10.4 Storage planning

The following are planning estimates, not measured compression guarantees.
An example game is defined here as 80 individual moves, or 40 moves by each
player, with approximately 80 physical move visits.

| Asset | Assumption | Estimated size per example game |
| --- | --- | --- |
| Occupancy crops | Six post-move 200x200 colour crops per position; about 480 crops at 40-120 KB each as PNG | 20-60 MB |
| Hand observations | Four selected observations per visit; about 320 baseline/current pairs at 0.4-1.6 MB per pair using high-quality JPEG | 130-510 MB |
| Combined planning allowance | Selective collection with small masks and manifests, before substantial extras | Approximately 150-600 MB |

These estimates use decimal MB and GB. One uncompressed 200x200 three-channel
crop is 120 KB. One uncompressed 1200x1200 three-channel image is 4.32 MB, so
320 independently stored uncompressed hand pairs would consume approximately
2.8 GB. Lossless hand-image storage may be substantially larger than the JPEG
allowance and shall be measured separately.

At approximately 150-600 MB per game:

| Games | Approximate storage |
| --- | --- |
| 10 | 1.5-6 GB |
| 100 | 15-60 GB |
| 1,000 | 150-600 GB |

A 100 GB allowance would hold roughly 150-650 games after allowing some room,
subject to actual compression, retention, and sampling. Continuous video,
frequent complete snapshots, or saving before and after crops independently
would increase these figures. The six-crop estimate counts post-move examples
only; linked before frames may reuse already retained assets. Delayed commitment
discards the final pending position, and correction-triggered deletion reduces
retained counts further.

The collector shall store each clean baseline once and reference it from
multiple current observations. Masks and repeated assets may likewise be
deduplicated without losing record provenance. Excluded hand-image regions
shall be masked before encoding. Dataset reports shall distinguish logical
sample counts from actual bytes and unique image assets.

The initial hand policy should select up to four representative observations
per physical visit: entry, crossing, stationary crossing when it occurs, and
departure. It shall not label a frame as stationary or crossing merely to fill
a quota. Deliberate stationary-hand sequences may supply missing examples.

The first two or three pilot games shall measure actual bytes per asset,
compression artifacts, mask overhead, baseline reuse, and examples per visit.
Use those measurements to revise the storage forecast and select the codec.
Higher-quality or lossless storage shall be retained if JPEG artifacts harm
boundary or shadow evidence. A smaller source hand resolution is an experiment,
not an automatic storage shortcut.

### 10.5 Curated retention

Successful training shall not automatically delete the examples used to train
the model. Model weights are not a recoverable copy of the dataset. Retained
images support label correction, failure investigation, architecture changes,
retraining from scratch, and reproducible comparisons.

The dataset shall distinguish:

| Category | Retention policy |
| --- | --- |
| Frozen validation and test sets | Protect while their results are relied upon; never prune automatically |
| Reviewed representative training core | Retain coverage of phases, setups, classes, square locations, and obstruction conditions |
| Rare cases, hard negatives, and reviewed corrections | Preferentially retain |
| New unreviewed examples | Keep temporarily pending review and selection |
| Redundant bursts, duplicates, corrupt or unusable examples | Eligible for removal after integrity and reference checks |

Pruning shall target redundancy rather than simply deleting the oldest phase.
Near-duplicate candidates may differ in meaningful boundary or shadow evidence;
similarity alone shall not authorize deleting rare or protected examples.
Reports shall show retained coverage and bytes reclaimed.

Deleting one record shall not remove a shared image, mask, or baseline still
referenced by another retained record. Explicit session deletion shall disclose
affected protected records and preserve shared assets needed by other sessions.
Retention shall respect the user's explicit deletion choice, but shall not
silently invalidate a retained baseline/current pair.

Record the exact manifest, label revisions, and split assignment for each
training run. If pruning removes assets from an older run, keep its provenance
and explicitly mark that run as no longer fully reproducible; do not imply
that a manifest hash alone preserves the deleted training data. Preserve full
dataset snapshots for important reproducible model releases where storage
permits.

Quota enforcement shall not evict protected evaluation data or the retained
training core. If eligible pruning cannot free sufficient space, pause
collection and report the condition rather than silently discarding protected
data. Early collection should favor retaining diverse source evidence over
aggressive compression or pruning.

Routine pruning is distinct from discarding examples known or suspected to
have incorrect gameplay labels. Section 7.5 governs that deletion and overrides
routine protection, including for previously retained snapshots. A bounded
list of sample identifiers and deletion reasons is sufficient; a separate
quarantine dataset is not required.

### 10.6 External-drive storage

An external drive may be the primary dataset store. A nominal 1 TB drive
provides approximately 1,000 GB in the decimal units used above, or 931 GiB
before filesystem overhead. As an illustrative budget, allocating 800 GB to
the dataset and leaving 200 GB for headroom and other uses would accommodate
approximately 1,300-5,300 example games at 150-600 MB per game. This assumes
that capacity is actually free; the pilot measurements shall replace this
estimate, especially if lossless hand images are retained.

This capacity makes a substantial curated history practical and removes any
need to discard each training batch immediately. It does not remove the need
for sampling, review, provenance, or independent backups.

External-drive behavior shall satisfy the following:

- Confirm that the selected dataset directory is available and writable before
  collection. A changed drive letter shall require locating and verifying the
  intended dataset rather than creating a different dataset silently.
- Write and encode asynchronously through a bounded queue so slow storage
  cannot block the camera, game UI, or move-verification loop.
- On disconnection, full disk, permission failure, or sustained queue overflow,
  pause collection, report the reason and unsaved sample count, and leave game
  behavior unchanged. Never silently redirect captures to an unapproved path.
- After reconnection, verify the dataset and incomplete writes before resuming.
  Atomic record commits do not replace integrity checks after unsafe removal.
- Report available space, dataset quota, and pending writes independently.

The initial intended dataset store is a 1 TB external HDD. Its actual throughput
and small-file performance shall be measured;
USB attachment alone does not establish whether it can feed GPU training fast
enough. Keep active runtime models on reliable local storage so unplugging the
dataset drive does not disable an already loaded inference path. Training may
use a verified local SSD cache of the selected data when external-drive loading
is a bottleneck; caches shall remain traceable to immutable sample hashes.

One external drive is not a backup. Keep an independent copy of irreplaceable
reviewed examples, frozen evaluation data, manifests, label revisions, and
important model releases. A second copy on the same physical drive does not
protect against loss or failure of that drive.

## 11. Diversity and Collection Strategy

Changing the camera, projector, board, pieces, lighting, and room geometry is
valuable, but those changes shall be recorded as setup groups so their value
can be measured.

Collection should deliberately vary:

- Camera height, lateral position, rotation, and zoom.
- Board material, print, border, wear, and reflectivity.
- Piece design, height, colour, gloss, and base diameter.
- Projector position, intensity, colour balance, focus, and active content.
- Ambient daylight and artificial-light direction, brightness, and colour.
- Clothing, skin tones, sleeves, watches, and hand-entry edges.
- Cast shadows, glare, partial framing, and camera auto-exposure transitions.
- All four board orientations and all square locations.

One long game can yield hundreds of files but not hundreds of independent
setups. Dataset reports shall show unique games, positions, capture bursts,
physical setups, and collection days in addition to raw image count.

### 11.1 Phased collection and training

Collection shall introduce variation in four phases. These phases control the
data distribution; they are separate from implementation milestones and runtime
rollout stages. Training may begin during Phase 1 and shall continue as later
phases add data.

| Phase | Conditions varied | Conditions held roughly constant |
| --- | --- | --- |
| 1: Fixed-setup baseline | Natural positions, legal moves, hands, sleeves, and entry edges | Camera and projector placement, board, pieces, and lighting |
| 2: Placement variation | Camera position, height, angle, and then projector placement | Board, pieces, and lighting |
| 3: Lighting variation | Ambient brightness, direction, colour, daylight, and shadows | Original board and pieces; selected placements from Phases 1 and 2 |
| 4: Board and piece variation | Boards and piece sets, initially one at a time | A reference placement and lighting condition for initial comparisons |

Phase 1 establishes that normalization, labels, review, and basic classification
work on a controlled setup. It does not demonstrate generalization to new
equipment or conditions. Include deliberate hand and hard-negative recordings
even in this phase rather than relying only on normal games.

Within each later phase, initially change one factor at a time so regressions
can be attributed to geometry, lighting, board appearance, or piece design.
In Phase 2, test camera changes separately from projector changes before
combining them. Establish or verify calibration after each placement change;
never collect training examples under an invalid grid.

After isolated variations work, collect combinations of earlier variations.
For example, Phase 3 shall eventually combine different lighting with multiple
placements, and Phase 4 shall test new boards and pieces under more than one
previously validated placement and lighting condition. This avoids learning
that a particular board always implies one particular light or camera angle.

Each phase shall:

- Record its phase identifier and changed factors in audit metadata.
- Retain earlier-phase training examples and include them in subsequent
  training, rather than fine-tuning only on the latest condition.
- Re-evaluate the same frozen earlier-phase regression sets to detect forgetting.
- Reserve new-condition groups for validation and independent testing.
- Review systematic errors and collect targeted examples before adding further
  complexity.

Progression is evidence-driven, not a fixed number of games. The next phase
may start once collection integrity is confirmed, a baseline model is trained,
and the current phase's errors are characterized. Passing a phase does not
authorize runtime deployment or imply support for unseen later-phase setups.

## 12. Dataset Splitting and Leakage Prevention

Random per-image splitting is prohibited because adjacent frames, before/after
pairs, and repeated positions are highly correlated.

At minimum:

- All frames from one capture burst remain in one split.
- Linked baseline/current hand pairs remain in one split.
- All samples from one game remain in one split.
- From Phase 2 onward, validation and test sets contain held-out physical
  setup groups. Phase 1 instead holds out whole games and, where possible,
  collection sessions or days; reports shall describe this as fixed-setup
  evaluation, not unseen-setup generalization.
- A final generalization test should hold out at least one board/piece setup
  and, when enough data exists, one camera placement or collection day.

Duplicate and near-duplicate image checks shall run before training. Dataset
manifests shall record the grouping key and deterministic split assignment so
experiments are reproducible.

## 13. Model Contracts

### 13.1 Joint occupancy and piece-colour model

Input:

- One 2x2-square BGR or RGB crop.
- Its validity mask.
- Known square colour and encoded physical location as defined in Section 8.1.

Output:

- A single normalized three-class distribution: `empty`, `white-piece`,
  `black-piece`.
- Derived occupancy: `P(empty)` and
  `P(occupied) = P(white-piece) + P(black-piece)`.
- Derived piece colour: `none` for empty, otherwise white or black.
- Model and input-schema versions.

The runtime shall support an inconclusive policy based on calibrated
probability or out-of-distribution evidence. Occupancy can be confident while
piece colour remains inconclusive; do not force a colour when white/black
probabilities are ambiguous. `unknown` is a review/inconclusive state, not a
fourth physical class.

The three-class head prevents contradictory definite predictions such as an
empty square with a white piece. Training shall map consistent complete labels
to these classes. Occupied examples with unknown colour shall not be assigned
a fabricated joint class; exclude them from the initial three-class objective
unless a later version explicitly defines a partial-label objective.

### 13.2 Hand model

Input:

- Current ring and its validity mask.
- Optional linked baseline ring/mask, with explicit availability and pair
  compatibility information.
- A versioned, explicitly selected metadata vector.

Output:

- Independent current-image probabilities of `clear` and `crossing`, with a
  calibrated inconclusive policy.
- Baseline-content probabilities of `valid` and `invalid`, with inconclusive
  or unavailable status when appropriate.
- Optional paired comparison evidence, separately identified with its baseline
  dependency and compatibility status.
- Model and input-schema versions.

Baseline validity shall depend on baseline content, not which current image
accompanies it. The current-only output shall not depend on baseline features
or validity. Record confidence and availability separately for the outputs;
never zero out a current result merely because the reference is invalid.

The operating threshold shall prioritize crossing recall over convenience.
Use information according to the rollout stage:

| Current-image result | Baseline/comparison state | Interpretation |
| --- | --- | --- |
| Confident crossing | Any, including missing/invalid | Retain obstruction evidence; after approval it may support blocking |
| Confident clear | Missing/invalid/incompatible | Retain the result; clearing requires separately validated current-only authority |
| Inconclusive | Valid and compatible | Evaluated paired comparison may provide additional evidence |
| Inconclusive | Missing/invalid/incompatible | Remain inconclusive and use the existing detector/fallback |

A missing baseline need not disable independent current inference. Missing or
invalid required current input does. No uncertain output shall silently become
`clear`. High scores are not proof of reliability: calibrate and evaluate the
current-only path, including stationary hands and bad/missing-reference cases.
In shadow mode all outputs remain diagnostic. Guarded assistance still cannot
clear a deterministic block; current-only clearance requires explicit later
promotion under Section 19.3.

Flagging a baseline for replacement requests a new suitable reference for
future comparison; it does not veto current evidence. During shadow mode this
request is advisory and shall not change the gameplay detector's baseline
ownership, force calibration, or alter its acquisition/timing policy.

### 13.3 Architecture

The initial architecture should be a small convolutional network or compact
vision backbone suitable for local inference. For square recognition, fuse a
small square-colour/location branch with image features before the shared
three-class head. Occupancy and piece colour use one model, not separate
networks. The hand model remains independent.

Architecture choice remains an experiment, not a storage-format commitment.
The dataset shall remain usable by alternative PyTorch, ONNX, or other local
training pipelines.

The initial proposed experiments are:

| Model | Initial candidate | Inference image resolution |
| --- | --- | --- |
| Occupancy and piece colour | Pretrained MobileNetV3-Small, square-colour/location features, three-class head | 160x160; compare 128x128 |
| Hand gate | Shared MobileNetV3-Small encoder, independent current-obstruction and baseline-validity heads; optional comparison branch | 384x384 per available ring; compare 512x512 if boundary detail is lost |

For occupancy, fine-tune pretrained image features before considering a larger
network or training from scratch. Compare against a simple small CNN where
useful. Image-only here means no semantic metadata branch; validity masks and
the declared masking policy remain part of the input contract.

For hand detection, use shared encoder weights for current and available
baseline images. Attach the obstruction head to current features and the
validity head to baseline features. Optionally fuse features/differences in a
separately evaluated comparison branch; pair fusion shall not be the only route
to a current-image decision. Support absent references during training and
inference through explicit availability and suitable training coverage, not
unmarked placeholder images. Adapt mask handling without treating invalid
pixels as observed background. Test stationary crossings explicitly.

Use image/validity information plus square colour and location for the agreed
square model. Add further geometry or capture metadata only when an ablation
study on appropriate held-out groups demonstrates an improvement. Piece
colour is an output target, never a supplied recognition answer.

PyTorch is the proposed training framework. ONNX export is a candidate for
application inference after model selection, not a requirement to add runtime
dependencies now. GPU support shall be verified against the actual installed
hardware and supported backend rather than assuming CUDA availability.

## 14. Training and Evaluation

### 14.1 Training environment

Training shall run outside the real-time camera loop and may use the local GPU.
The training command shall:

- Report the selected device and fail explicitly when a requested GPU is
  unavailable.
- Record code revision, dataset manifest hash, split definition, random seed,
  hyperparameters, and dependency versions.
- Save checkpoints and metrics without modifying collected source examples.
- Support a CPU mode for reproducibility and environments without a compatible
  GPU.

CPU-only gameplay is a deployment requirement, not merely a training fallback.
Inference shall have a measured CPU path even if GPU inference is available.
Loss of GPU availability during application startup shall produce an explicit
fallback or model-unavailable state, not a false successful load.

The initial user-reported training GPU is an NVIDIA RTX A500 with 4 GB GDDR6.
The proposed compact models are candidates for this memory budget, not a
promise of a particular batch size, speed, or supported runtime. Verify driver,
PyTorch/CUDA compatibility, available GPU memory, and actual peak training
memory before choosing settings. The two independent models may be trained
sequentially rather than resident together.

Benchmark a small batch first, especially for the optional two-pass hand path
at 384x384 or 512x512. Adjust batch size and consider supported mixed precision,
a frozen encoder, or gradient accumulation as measured options. An
out-of-memory failure shall be explicit; do not silently reduce resolution or
change the experiment configuration. Training shall not compete with active
gameplay by default.

Before large-scale collection, benchmark CPU inference including normalization,
masking, batching, and scheduling overhead. Model inference time alone is not
move-registration latency. The selected model must fit the observation budget
without worsening gameplay responsiveness; GPU acceleration does not waive
this CPU requirement.

### 14.2 Augmentation

Permitted augmentation may include modest photometric changes, sensor noise,
blur, and geometry perturbations consistent with calibration uncertainty.
Augmentation shall not:

- Move content across the occupancy label boundary.
- Invent valid pixels where the source mask is invalid.
- Flip or rotate samples without updating square-edge metadata.
- transform baseline and current hand images inconsistently.
- Simulate unrealistic projector or shadow behavior without being identified
  as synthetic.

### 14.3 Metrics

Occupancy reporting shall include:

- Empty and occupied precision, recall, and F1.
- Confusion matrices by square parity, edge/corner class, piece colour, setup,
  and validity fraction.
- Calibration metrics and coverage when low-confidence examples are rejected.
- Per-position exact 64-square accuracy where complete snapshots exist.
- Errors per accepted game and errors on changed versus unchanged squares.

Hand reporting shall include:

- Crossing recall and false-clear count.
- Clear specificity and false-block duration.
- Metrics by entry edge, setup, clothing/hand tag, edge-piece tag, projector
  state, and baseline age.
- Time from physical entry to blocking and from departure to clear.
- Results for stationary hands and hard negatives such as tall edge pieces and
  shadows.
- Separate calibration, coverage, and errors for current obstruction and
  baseline validity; current-only versus paired comparisons; and valid,
  invalid, unknown, missing, and incompatible reference conditions.
- Baseline-validity results by distinct baseline identity, not merely by the
  number of current observations reusing that reference.

Raw frame accuracy alone is insufficient for either model.

### 14.4 When to start training

The following are experimental checkpoints, not minimum dataset guarantees or
deployment thresholds:

| Checkpoint | Approximate collection | Purpose |
| --- | --- | --- |
| Pipeline pilot | 2-3 games | Verify crops, labels, timing, storage, and baseline linkage |
| First training experiment | 5-10 games plus targeted hand sequences | Establish a baseline and discover missing or mislabeled cases |
| Broader evaluation | 20-30 games spanning available phase variations | Compare models and quantify fixed-setup or unseen-setup performance as applicable |

At approximately 480 occupancy crops per game, 21 games yield about 10,000
crops, nominally 5,000 empty and 5,000 occupied. The actual usable counts depend
on rejected frames, special moves, sampling, and review. Held-out examples are
not part of the training allocation, so 10,000 collected crops do not imply
10,000 training crops.

Train early instead of waiting for hundreds of games. Plot learning curves on
increasing, group-preserving training subsets while keeping validation data
fixed. If errors concentrate on shadows, edges, or a particular setup, collect
those cases specifically. Do not repeatedly tune against the final test set.

Each model shall have its own readiness assessment. Thousands of automatically
generated hand labels do not substitute for reviewed stationary crossings,
all-side entry sequences, tall edge pieces, captured pieces outside the board,
and shadow-only negatives. Targeted short sessions may provide these more
efficiently than full games.

### 14.5 Periodic retraining

After the initial experiment, approximately five additional games may be used
as an early retraining checkpoint. This is a review cadence, not a mandatory
automatic trigger. New camera placements, lighting conditions, boards, pieces,
or clusters of reviewed errors may justify retraining sooner; redundant new
frames alone need not justify a run.

Each retraining cycle shall:

1. Validate and review the new examples, keeping their label provenance.
2. Select a versioned training set from the retained representative core plus
   new eligible examples. Preserve earlier-phase representation.
3. Train outside the real-time camera loop. Fine-tuning or training from
   pretrained initialization may be chosen experimentally; neither permits
   dropping earlier conditions from evaluation.
4. Compare the candidate with the current model on frozen earlier-phase
   regression sets and appropriate held-out new-condition validation groups.
5. Apply the existing independent occupancy/hand rollout and promotion gates.
   A completed training run shall never automatically replace the operational
   model.
6. Retain the current model for rollback and preserve the candidate's dataset
   manifest, configuration, metrics, and decision.
7. Prune only eligible redundant or unusable assets under the retention policy,
   not the entire batch just used for training.

Final test sets shall not become a repeatedly consulted tuning signal.
Earlier-phase regression sets used routinely for candidate selection shall be
identified as validation data, with separate independent tests reserved for
promotion. Collecting additional phase data shall not quietly reassign frozen
evaluation samples to training.

Retraining on only the latest data risks catastrophic forgetting. Training
reports shall therefore show performance by earlier phase and setup as well as
on the new distribution. Dataset curation, retraining, and operational model
promotion are separate explicit operations.

## 15. Review Workflow

The review queue shall prioritize:

1. Disagreement between authoritative state, deterministic detector, and
   neural model.
2. Low-confidence or out-of-distribution examples.
3. Samples from new setup groups.
4. Rare class/metadata combinations.
5. A random audit sample from high-confidence agreements.

A reviewer shall be able to see the normalized input, masks, linked
before/after or baseline/current image, proposed label, provenance, and
relevant non-sensitive metadata. Review actions shall append reviewer label,
reason, and timestamp while preserving the original label.

The same simple image-and-label correction tool shall support both datasets:

- Occupancy: show the target-square crop, identify the target square, and offer
  `empty`, `occupied`, `unknown`, and discard actions.
- Hand gate: show baseline/current rings together and offer `clear`, `crossing`,
  `unknown` current labels plus independent `valid`, `invalid`, `unknown`
  baseline labels and discard actions.
- Provide quick accept/correct/next controls and keyboard navigation so review
  does not require opening individual image files.
- Persist accepted label revisions and exclude discarded records from training
  manifests with reference-safe asset deletion.

### 15.1 At-a-glance review and label editing

The main reviewer shall show the image(s), editable labels, and a compact
semantic summary simultaneously without scrolling. Support a 960x640 dialog
client area; previews shall scale to available space without distortion or
clipping. Raw numeric metadata shall move to a separately opened, read-only
**Technical details...** window, where scrolling is permitted.

Separate **Model inputs (read-only)** from **Model outputs (training labels)**
in visibly titled groups. Display known square colour and physical location
as input metadata, without a redundant **Images** text row. The previews
already display the image inputs; hide an empty input-metadata group for hand
examples. Outputs provide named **Occupancy** and
**Piece colour** dropdowns (`white`, `black`, `none`, `unknown` for colour).
Square colour and location remain derived facts, not editable output labels.
For hand examples, identify paired baseline/current rings and masks as inputs,
show **Current obstruction** and **Baseline validity** as separate outputs,
and hide square-specific fields. These output dropdowns are implemented.

Move projector state and piece-colour provenance out of the main panel and
retain them in **Technical details...**. They are capture/audit metadata, not
editable training targets or declared model inputs. Keep any coverage warning,
original primary label, review status, and demo-only exclusion visible.

Missing required input facts, such as square colour, shall be explicitly
identified as not recorded. Do not infer a light/dark colour from physical
parity alone when orientation is unknown. Optional audit fields may remain
absent from technical metadata when they were never captured. Do not invent a
projector angle. Metric projector pose remains optional under Section 8.2.

Dropdown changes are drafts until **Save and next** or its keyboard equivalent.
Next/Previous navigation leaves stored labels unchanged. Save the primary label and
piece-colour correction atomically, recording previous colour/source, new
colour, reviewer provenance, and timestamp without a dataset migration.
Accepting an unchanged colour shall preserve its original source. Reject
contradictory explicit labels (`empty` with white/black, or `occupied` with
`none`) visibly and keep the current example/draft available for correction.
Hand examples shall not acquire piece-colour labels.

Empty or unreadable examples shall disable editing/saving and report the
reason; a failed write shall retain the current draft and not advance.
Existing gameplay, collection policy, demo exclusion, and training-input
boundaries remain unchanged.

### 15.2 Remembered dataset location

The reviewer and collection tool shall share the last successfully selected
dataset directory through the existing application's local QSettings store.
When the user opens the reviewer without an explicitly supplied dataset,
attempt to reopen that directory. An explicitly supplied/current dataset
takes precedence. Both dataset directory choosers shall start at their current
dataset, or the remembered directory if none is selected.

Remember selections only after an existing dataset opens successfully or an
explicit collection-tool selection creates/opens a dataset. Cancellation and
invalid selections shall not replace the remembered directory. Report settings
write failures visibly rather than claiming the location was remembered.

A missing/disconnected drive, deleted dataset, or unsupported database shall
produce a non-modal error in the reviewer, preserve the remembered directory,
and leave **Open dataset...** available. Restoration shall never create a
replacement database, start collection/calibration, or run during application
startup; it occurs only when the user opens the reviewer. Collection still
requires explicit dataset selection and start.

Review shall be a separate user-invoked activity, never a compulsory
interruption during play. Routine occupancy examples may use the delayed
implicit-acceptance policy without per-image approval. Prioritized review and
random audits remain available, particularly for hand labels and independent
evaluation sets. Gameplay corrections discard the affected interval directly
under Section 7.5 instead of populating this review tool with quarantined
images.

### 15.3 Hand observation navigation and baseline review

Browse current observations, not an independent list of baselines. Show the
linked reference on the left, captioned **Linked baseline**, and the selected
image on the right, captioned **Current observation**, under one observation
identity/counter. The left image updates to match each current observation;
it can legitimately stay the same when consecutive observations share a
baseline. Missing references shall be explicitly indicated on the left.

The redundant **Images** explanatory row is removed for both datasets;
captions identify the actual image previews. Use **Previous**,
**Save and next**, **Next**, and **Discard observation**. Next/Previous change
the current observation without saving drafts; Save and next saves both
applicable output labels atomically before advancing. There is no independent
baseline-navigation control.

For both datasets, navigation is bounded by the current filtered list, not
circular: Previous is disabled at the first example, Next at the last, and
both for an empty or single-example list as applicable. Left/Right shortcuts
obey the same limits. Attempting to move past a boundary keeps the example
and its unsaved draft unchanged. At the last example the save action reads
**Save (Enter)**, commits labels, and stays there; empty lists cannot save.
Deletion/refresh shall clamp the selection to an existing remaining example,
never wrap to the start.

Review current obstruction and baseline validity separately. Changing a shared
baseline label shall disclose the number of linked observations affected and
require confirmation before applying the shared-label revision. An invalid
baseline shall not trigger automatic deletion, overwrite the current label,
or suppress its review. Unavailable baseline labels are not fabricated.
All labels/captions shall remain visible without scrolling at the declared
minimum review size. These captions, **Discard observation** wording, independent
output labels, shared-baseline editing, and bounded navigation are implemented.

### 15.4 Approved reviewer/data completion contract

This implementation phase completes hand observation storage and review only.
Neural models, training, inference, promotion, and change-based CI selection
remain deferred. Gameplay decisions and detector baseline ownership shall not
change.

- Preserve schema-1 SQLite datasets without a migration. Store new baseline
  labels, original labels, source/strength, reviewed status, and timestamped
  revisions in versioned metadata. Keep current-label provenance independent;
  reviewer revisions are marked `reviewed`, not automatically infallible.
  Legacy absent baseline labels are `unknown`.
- Identify a shared baseline by its image and validity-mask asset hashes.
  Identical reference content shares its content-validity label; acquisition
  time and geometry remain observation context, not intrinsic validity.
  Keep that stable identity in records and exports without rewriting old
  records merely by opening a dataset.
- Shared baseline review updates every linked observation atomically with the
  selected current label, never other current labels. Before a shared revision,
  show the affected count and request confirmation. Verify the confirmed
  member identifiers again inside the write transaction; concurrent link
  changes reject the save visibly without advancing or losing drafts.
- New observations referencing a reviewed baseline inherit its existing
  validity/history instead of overwriting it with an automatic proposal.
  Explicitly conflicting initial labels for the same baseline are rejected.
- Accept either current image/mask alone or a complete baseline/current pair.
  Reject half-present image/mask roles and malformed assets. The opt-in hand
  collector may retain transitions before a usable reference exists or after
  geometry discontinuity; validate current coverage independently. When a
  baseline acquisition is proposed by the existing gate, its `valid` label is
  weak, with separate provenance, not an independently reviewed fact.
- Expose per-output training eligibility for current obstruction, baseline
  validity, and clean-compatible comparison in records and manifest exports.
  Unknown targets and demo examples are masked out independently. A known
  independently labelled current target survives invalid/missing/unknown
  references; a reviewed invalid reference can train validity even if current
  obstruction is unknown.
  Comparison additionally requires explicitly recorded compatibility.
  Old records without compatibility evidence are not silently comparison-ready.
  Baseline-dependent weak current labels with a disputed invalid reference are
  retained for review but masked until independently reviewed.
- Manifest format version 2 shall include baseline identity, label/review
  state, linked observation grouping, and output eligibility. Shared baseline
  identity is a required grouping key for future split generation; this phase
  does not add a trainer or automatic split generator.
- Show one observation with **Linked baseline** on the left and **Current
  observation** on the right, with a shared identity/counter and baseline reuse
  count. Occupancy shows one **Square image**, not an empty second panel.
  Provide independent current-obstruction/baseline-validity dropdowns for
  hand observations, and keep occupancy's two output fields unchanged.
  Browse creation order with a stable insertion-order tie-break, not random
  identifiers when capture/write timestamps coincide.
- Missing/unreadable baseline assets are reported on the left and disable
  baseline editing, not review of a readable current image. Do not silently
  relabel baseline corruption as obstruction. Current-input failures still
  disable saving. All labels, captions, and controls fit at 960x640.
- Use **Discard observation** for explicit single-record deletion. Keep
  non-wrapping navigation, remembered locations, demo exclusion, and
  reference-safe deletion. A missing baseline never creates placeholder image
  assets, triggers automatic deletion, or starts calibration.

## 16. Runtime Rollout

### Stage 1: Collection only

- Implement normalized view and opt-in storage.
- Existing detectors remain unchanged.
- Validate labels, masks, manifests, and deletion/retention behavior.

### Stage 2: Offline experiments

- Train the agreed joint square model and an image-only ablation.
- Evaluate additional metadata only as a measured addition.
- Freeze held-out test groups before tuning.

### Stage 3: Shadow inference

- Run neural models without changing game decisions.
- Log outputs, latency, disagreements, and model versions.
- Add selected disagreements to the review queue.

#### Operating modes and authority

The first neural integration shall offer **Current detectors** (default,
neural inference off) and **Compare / shadow**. In shadow mode the existing
detectors, legal-move verification, and gameplay timing remain authoritative.
Neural results shall not change move acceptance, hand blocking, projection,
baseline ownership, or logical game state. Data collection remains a separate
explicit opt-in; enabling comparison does not authorize image persistence.

Occupancy and hand comparison shall be independently selectable. A future
**Neural detection** setting shall remain unavailable until the respective
model's rollout and acceptance criteria are met. Completing training or
selecting shadow mode shall never promote a model automatically.

#### Scheduling, comparable observations, and failure handling

Continuous comparison means repeated observations within an explicit measured
resource budget, not mandatory inference on every camera frame. Inference and
comparison persistence shall run off the UI/gameplay thread with bounded
pending work and latest-observation preference. Obsolete work shall be dropped;
late results shall retain their original observation identity and shall not
be compared against a newer detector result.

Compare matching image inputs, verified geometry, validity masks, projection
state, and relevant temporal context. Occupancy comparisons shall use settled,
unobstructed observations. Hand comparisons shall preserve the actual
baseline/current pairing and gate state; differing baselines or temporal
policies shall not be reported as directly equivalent predictions.

The comparison record shall identify the observation/frame and timestamps,
both outputs and available scores, model/checksum and preprocessing versions,
setup and collection phase, transform, projection state, and detector context.
Scores from different detectors shall not be assumed equally calibrated.
Selected persisted cases shall include the exact model inputs and masks,
subject to collection consent, quota, and retention rules.

Report inference latency, observation age, skipped/dropped work, unavailable
results, and comparison/storage failures non-modally. A failure shall pause
the affected shadow operation visibly while normal gameplay continues on
the existing path. Do not silently substitute a prediction or recreate a
missing dataset. CPU and optional GPU contention shall be measured against
existing-only gameplay; background execution alone does not establish that
gameplay performance is preserved. Training remains off during gameplay by
default.

#### Disagreement review and model improvement

A disagreement shall be a review candidate, not an automatic neural-error
label: the existing detector and implicitly accepted position may also be
wrong. Prioritize diverse disagreements, low-confidence cases, novel setups,
and random agreement audits. Deduplicate repeated observations of one
incident rather than storing every frame.

Use the shared occupancy/hand reviewer to assign corrected labels and retain
provenance. Subsequent-physical-move confirmation may supply provisional
occupancy evidence under Section 7; it is not independent ground truth.
Discard ambiguous or invalid cases instead of creating an image quarantine.
Retain useful, reviewed hard examples with correct labels.

Reviewed cases may feed targeted retraining or independent recorded replay.
Assign linked incidents to a single dataset split before use: examples used
to train a model shall not evaluate that model as held-out evidence.
Representative failures may become regression cases, but replay cases used
to tune models are not an untouched final test set. Preserve earlier-phase
coverage when adding new hard examples.

Compare candidate and released models on identical versioned replay inputs
and report class/setup errors and regressions, not only aggregate accuracy.
For hands, evaluate entry-to-block, stationary crossings, false blocking,
false clears, and departure-to-clear over sequences as well as still images.
Do not retrain after every disagreement or activate a retrained model without
separate evaluation and explicit promotion.

### Stage 4: Guarded assistance

- Permit neural occupancy only in existing occupancy workflows, with
  deterministic fallback on uncertainty or model failure.
- Permit the hand model to add a conservative block, but not to clear a block
  from the deterministic safety gate.

### Stage 5: Candidate replacement

A detector may become authoritative only after its own acceptance criteria are
met on held-out data, recorded replay, and live hardware. Occupancy and hand
models shall be promoted independently.

Rollback to the deterministic path shall remain available without deleting the
model or dataset.

## 17. Failure Handling

The pipeline shall report and reject, rather than silently repair:

- Missing or unverified camera grid.
- Incomplete required halo coverage.
- Invalid normalized transform.
- Baseline/current geometry mismatch.
- Missing, corrupt, or unsupported sample assets.
- Unknown metadata or model input schema.
- Model checksum mismatch.
- Unsupported inference runtime.
- Requested GPU unavailable for training.
- NaN, infinite, malformed, or wrong-shaped model output.

Collection failure shall not interrupt a game. It shall disable collection for
the affected sample or session, preserve game behavior, and report the reason
non-modally. A model failure shall follow the active rollout-stage fallback
policy and shall never produce a success-shaped default prediction.

For hand inference, reject the affected input or comparison branch rather than
unconditionally rejecting independent current-image evidence. A missing,
unreadable, or incompatible baseline shall be reported and disable its own
assessment/comparison as appropriate; a valid current image can still use the
supported current-only path. Required current-input failures remain
unavailable, never implicitly clear.

## 18. Tests

### 18.1 Normalized-view tests

- Exact 12x12 mapping with the board at `[0, 8] x [0, 8]`.
- Correct two-square extrapolation for perspective camera grids.
- Stable pixels-per-square dimensions.
- Validity-mask behavior at every camera-frame edge.
- Physical and UI-orientation preview mapping.
- Occupancy crop bounds for center, edge, and corner squares.
- Hand-ring masking of the central 6x6 board and beyond-halo content.

### 18.2 Collector tests

- Collection disabled by default.
- No files written merely by opening the preview.
- Atomic image/mask/manifest creation.
- Strong and weak label provenance.
- Before/after and baseline/current linkage.
- Pending occupancy samples stay out of the stored dataset until the next
  accepted physical move; activity or server-only updates do not commit them.
- Game end, shutdown, collection stop, and history discontinuities discard the
  unconfirmed pending occupancy record.
- Correction-triggered interval deletion, whole-game fallback without a
  checkpoint, and exclusion of affected snapshots and caches.
- Deferred physical deletion on an unavailable drive never permits affected
  identifiers into a training run.
- Quota, pause, retention, deletion, and interrupted-write behavior.
- Protected-core and frozen-evaluation retention; reference-safe pruning of
  shared baselines, masks, and images.
- External-drive disconnection, reconnection, changed path, full disk,
  permission failure, and bounded-queue overflow without affecting gameplay.
- Rejection of projected-pattern, moving, obstructed, or insufficiently framed
  occupancy samples.
- No raw credentials, device serials, or unrestricted full frames in normal
  records.

### 18.3 Dataset tests

- Group-preserving deterministic splits.
- No game, burst, or linked pair crossing splits.
- Duplicate detection.
- Class and setup distribution reports.
- Exclusion of `unknown` and disallowed weak labels from supervised training.
- Reproducible manifest hashes and experiment configuration.
- Accurate reproducibility status after pruning assets from an earlier run.
- Retraining preserves earlier-phase coverage and cannot move frozen
  evaluation samples into training.
- Label correction and discard actions for both occupancy and paired hand
  images; revisions preserve provenance without retaining discarded images.
- Atomic primary/piece-colour corrections, preserved original provenance,
  unchanged-colour acceptance, and rollback on invalid combinations.
- At-a-glance dropdowns and summary at 960x640, draft discard on navigation,
  separate technical details, and visible failed-save handling.
- Correct input/output grouping, prefilled stored labels, hand-specific field
  visibility, and provenance/projector metadata retained only in details.
- Last-directory persistence across tool instances, chooser initialization,
  cancellation/invalid-selection preservation, explicit-dataset precedence,
  and unavailable-dataset restoration without creation or collection.
- Training-run impact tracking and invalidation of evaluation results that
  depended on discarded examples.
- Independent hand-output labels, shared-baseline revision confirmation and
  provenance, retained useful current targets, per-output loss eligibility,
  and reference-grouped splits without automatic baseline-wide deletion.

### 18.4 Inference tests

- Input and model schema validation.
- Explicit unavailable state for load or runtime failure.
- Confidence-based inconclusive behavior.
- CPU inference.
- Optional GPU inference and controlled fallback.
- Shadow mode cannot alter game state.
- Separate enablement and rollback for occupancy and hand models.
- Completing retraining does not automatically promote its model.
- Default current-detector mode performs no neural inference; shadow selection
  does not implicitly start image collection.
- Both normal and failed/slow shadow inference preserve existing move,
  projection, and hand-gate decisions.
- Observation identity and baseline/geometry checks prevent mismatched or
  stale comparisons; bounded work drops obsolete observations visibly.
- Reviewed disagreement cases preserve label provenance and split isolation.
- Current-only obstruction output is unchanged when only baseline content,
  availability, or validity changes; baseline-content output is unchanged when
  only the linked current image changes.
- Missing/invalid/incompatible baselines disable only dependent comparison
  evidence, not supported independent current inference; missing current input
  remains explicitly unavailable.
- Per-output calibration, abstention, and rollout gating, including retention
  of confident current results without premature current-only clearance.
- Current-driven navigation keeps the correct linked baseline on the left and
  current image on the right, including baseline reuse and missing references.
- No redundant Images row for either dataset; no empty hand input-metadata
  group. Buttons and shortcuts cannot wrap at either filtered-list boundary,
  blocked navigation preserves drafts, and final/single-example saving stays
  on the saved example.

### 18.5 Change-based regression execution policy

This is the approved target policy, not an implemented test selector.
Vision-specific regressions shall run when this addition or its relevant
dependencies change, not merely because unrelated work is being pushed.
Validation already completed for unchanged task files shall not be repeated
solely during the push operation.

Trigger selection shall include vision modules, collector/reviewer integration,
accepted-move and correction notifications, camera input, shared geometry and
orientation, calibration, projection/hand-baseline contracts, dataset schemas,
preprocessing, model weights and thresholds, and relevant dependencies,
quality scripts, or workflows. A file-name filter limited to the new vision
modules is insufficient. Unknown dependency impact requires conservative
broader validation rather than silently skipping coverage.

Keep fast hardware-free integration checks separate from expensive recorded
sequence/model evaluation. Model or preprocessing changes require the relevant
versioned replay evaluation even when Python source is unchanged. Large
datasets, GPU training, and complete model benchmarks shall not become
unconditional work on ordinary pushes. Record the affected checks and their
results so conditional execution remains auditable.

The current `scripts/run_quality.py` compiles the application and runs all
tests; neither that runner nor CI is changed by this specification update.
Until an explicitly implemented and validated selector replaces this behavior,
the existing exact local quality gate remains in force before applicable
pushes. Conditional execution must not be simulated by bypassing that gate.

## 19. Acceptance Criteria

### 19.1 Data foundation

The first implementation milestone is accepted when:

1. The preview shows the complete normalized board plus requested halo and
   marks invalid pixels.
2. Collection is opt-in and obeys pause, quota, and deletion controls.
3. A center, edge, and corner occupancy sample has the specified fixed
   half-square margin.
4. A hand sample contains paired baseline/current rings with one square inside
   and two squares outside, while the central 6x6 board and farther
   surroundings are absent or masked.
5. Every record has versioned labels, provenance, grouping, transforms, masks,
   and content hashes.
6. Strong labels are distinguishable from inference-derived weak labels.
7. Split generation proves that no game, burst, setup group chosen for
   holdout, or linked image pair leaks across splits.
8. Pilot collection reports actual storage per game and explicitly compares it
   with the planning allowance.
9. Phase-specific reports distinguish fixed-setup evaluation from unseen-setup
   generalization and track performance on retained earlier-phase tests.
10. Routine retention and retraining do not delete protected data or leave
    dangling references; explicit deletion, correction-triggered overrides,
    and lost reproducibility are disclosed.
11. External storage failures pause collection visibly without blocking or
    changing game behavior.
12. Occupancy storage uses next-physical-move confirmation and discards the
    final unconfirmed position. Provenance distinguishes implicit acceptance
    from independent verification.
13. Both datasets have a usable image-and-label correction tool; gameplay
    corrections discard affected examples without an image quarantine.

### 19.2 Occupancy model

Before guarded runtime assistance:

1. Held-out evaluation reports class, setup, edge/corner, square-colour, and
   piece-colour slices, including three-class confusion, derived occupancy
   errors, and white/black errors on occupied squares.
2. The model supports an inconclusive result and demonstrates calibrated
   confidence on held-out setups.
3. Recorded complete-position replay measures exact-board errors, not only
   per-crop accuracy.
4. Shadow-mode disagreements have been reviewed for at least one representative
   collection from each supported physical setup.
5. CPU latency fits the application's occupancy schedule without degrading the
   UI or move tracker.

Numerical promotion thresholds shall be set only after a frozen representative
test set exists; inventing a percentage before data collection would not
establish safety.

### 19.3 Hand model

Before the model can influence blocking:

1. The held-out test set includes stationary hands, all four entry edges,
   sleeves, skin-tone and lighting variation, tall edge pieces, captured
   pieces, shadows, projection changes, and camera-grid movement.
2. There are no unexplained false clears in reviewed safety-critical test
   sequences.
3. Entry-to-block and departure-to-clear latency meet or improve the existing
   operational targets.
4. Shadow mode demonstrates that model failure or uncertainty never becomes a
   clear decision.
5. Current-obstruction and baseline-validity outputs are separately evaluated
   and calibrated. Tests prove that invalid/missing references do not veto
   independent current evidence or fabricate a clear result.
6. Training/review preserve useful per-output labels, include reviewed invalid
   baselines, and prevent shared-reference leakage or inflated evaluation.

Before replacing the deterministic gate, live hardware validation and an
explicitly approved false-clear threshold are required.
Before current-only predictions can clear observation, explicitly validate
their false-clear rate and stationary-hand performance across missing, invalid,
and incompatible reference conditions and approve that authority separately.
Baseline validity alone is never permission to clear the current observation.

### 19.4 Shadow-mode integration

Before shadow mode is accepted:

1. Existing-only mode is the default, with independent occupancy/hand
   comparison selection and no neural-authoritative option before promotion.
2. Recorded replay proves that enabling, slowing, or failing comparison cannot
   change existing gameplay decisions or start collection implicitly.
3. Comparisons refer to matching observations and declared temporal/baseline
   context; stale or incompatible results are explicitly excluded.
4. Pending inference/storage work stays bounded, with drops, failures, latency,
   and observation age reported without gameplay-thread waits.
5. Reviewed hard examples retain provenance, obey discard policy, and cannot
   leak between training and independent evaluation groups.
6. Live CPU and optional GPU measurements compare existing-only and shadow
   gameplay against an explicit resource/latency budget established before
   acceptance. Hardware-free tests alone cannot establish this criterion.

## 20. Implementation Milestones

### Milestone 1: Shared normalized view

- Add a reusable 12x12 warp and validity mask.
- Validate two-square camera framing.
- Add the optional diagnostic preview.
- Validate the preview before implementing image collection.

### Milestone 2: Versioned opt-in collector

- Define manifests, asset layout, identifiers, hashes, quota, and retention.
- Capture occupancy before/after records and sampled unchanged squares.
- Add the bounded pending buffer, next-move commit, and correction-triggered
  discard rules.
- Capture paired hand baseline/current rings.

### Milestone 3: Review and dataset tooling

- Add the image-and-label correction tool for occupancy and hand examples,
  including revision history and discard actions.
- Generate grouped splits and distribution reports.
- Detect corrupt, duplicate, and near-duplicate records.

### Milestone 4: Occupancy experiments

- Train the joint three-class square model with image/mask, square colour,
  and location inputs; compare an image-only ablation.
- Evaluate additional geometry and capture metadata independently.
- Export a versioned model with confidence calibration.

### Milestone 5: Hand experiments

- Train independent current-obstruction and baseline-validity heads with
  per-output labels, current-only coverage, and optional paired comparison.
- Emphasize false-clear analysis and difficult edge-piece negatives.
- Evaluate valid, contaminated, missing, and incompatible references without
  discarding independently useful current-image evidence.
- Export a separately versioned model and threshold policy.

### Milestone 6: Shadow mode and guarded rollout

- Add existing-only and independently selectable shadow comparison modes,
  with bounded measured inference and unchanged gameplay decisions.
- Review matched-input disagreements, retain useful hard examples, and expand
  held-out setup and sequence coverage without train/test leakage.
- Implement dependency-aware regression selection under Section 18.5 before
  claiming unrelated pushes skip vision regressions.
- Promote each model independently only when its criteria are met.

## 21. Open Decisions

- Pixels per square retained in the source dataset after storage and latency
  measurements.
- Exact valid-pixel fraction required for occupancy and hand examples.
- Final unchanged-square sampling rates after evaluating the initial two-empty,
  two-occupied policy, and full-board snapshot frequency.
- Storage quota and retention defaults.
- Manifest format and image codec.
- Detailed layout and shortcuts of the shared image-and-label correction tool.
- Final architecture and inference runtime after evaluating the proposed
  MobileNetV3-Small and paired-ring baselines.
- Calibration and abstention thresholds for derived occupancy versus piece
  colour when the joint probabilities support different confidence levels.
- Confidence calibration method and numerical promotion thresholds.
- Minimum number and diversity of physical setup groups before promotion.

