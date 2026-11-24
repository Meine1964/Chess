# Chess-Square Occupancy and UI Marking Specification

**Version:** 0.3
**Status:** Experimental implementation integrated with setup, reset,
reconciliation, and restoration workflows; further hardware tuning pending
**Platform:** Python on Windows  

**Related specifications:**

- [Projector/camera chessboard system](./projector-camera-chessboard-specification.md)
- [Piece-aware board localization](./piece-aware-board-localization-specification.md)
- [Single-shot coded-pattern calibration](./single-shot-coded-pattern-calibration-specification.md)
- [Chess-piece UI rendering and move synchronization](./chess-piece-ui-tracking-specification.md)
- [Reference-image move tracking](./reference-image-move-tracking-specification.md)
- [Neural vision data and recognition](./neural-vision-data-and-recognition-specification.md)

Post-initialization move inference is superseded by the reference-image
specification. The absolute occupancy classifier remains responsible only for
starting-position initialization, explicit reset verification, and
diagnostics. It shall not mutate an initialized logical piece position.

## 1. Purpose

Define how the application shall determine which physical chessboard squares
contain pieces and display that result on the digital chessboard in the user
interface.

The first visible representation shall be a large black dot centered on every
UI square classified as occupied.

This feature shall:

1. Use the calibrated camera grid to analyze all 64 physical squares.
2. Classify every square as occupied or empty.
3. Update only from stable, unobstructed camera observations.
4. Map physical-square observations through the confirmed board orientation.
5. Render occupancy independently from projector-square selection.
6. Avoid piece-type recognition; use aggregate piece brightness only to
   distinguish the White and Black starting sides.

## 2. Scope

The initial implementation covers:

- Occupancy detection from a stable camera frame.
- A standard starting-position initialization workflow.
- Light-square and dark-square empty-appearance models.
- Confidence-aware square classification.
- Temporal stabilization across multiple observations.
- Board-orientation conversion.
- Black occupancy dots on the existing UI chessboard.
- Diagnostics and explicit inconclusive states.

The initial implementation does not require:

- Recognition of kings, queens, rooks, bishops, knights, or pawns.
- Recognition of White versus Black pieces.
- Reconstruction of an arbitrary unknown position from one frame.
- Legal-move inference.
- Projecting occupancy markers onto the physical board.
- Updating occupancy while a hand or other moving object covers the board.

## 3. User-Visible Behavior

### 3.1 Occupied squares

Each confirmed occupied square shall contain a filled black circular marker in
the digital chessboard widget.

The marker shall:

- Be centered in the UI square.
- Have a diameter between 38% and 48% of the UI square width.
- Remain visible on both light and dark UI squares.
- Not change the square's coordinate labels.
- Not prevent mouse selection of the square.
- Coexist with the existing selected-square highlight.

The selected-square highlight shall be drawn before the occupancy dot so that
the occupancy state remains visible. The selected-square border shall remain
visible around the dot.

### 3.2 Empty squares

Empty squares shall have no occupancy marker.

The operational board state shall never contain unknown squares. If the whole
frame is obstructed or moving, the application shall reject that observation
and retain the previous complete binary state.

### 3.3 Status

The application shall report a concise occupancy status containing:

- Number of occupied squares.
- Number of empty squares.
- Whether the result is current, stabilizing, frozen by motion, or unavailable.

Examples:

- `Occupancy stable: 32 occupied, 32 empty.`
- `Occupancy stabilizing: 3/5 observations.`
- `Occupancy frozen while motion or occlusion is present.`
- `Occupancy unavailable: projector/camera calibration is not verified.`

## 4. Inputs

The occupancy subsystem shall consume:

- The latest BGR camera frame.
- A confirmed camera-space 9x9 board grid.
- The current board-tracker state.
- The current board orientation.
- Calibration validity and confidence.
- A timestamp or monotonically increasing frame sequence.

The subsystem shall not infer board geometry independently. Geometry remains
owned by the board-localization and calibration subsystems.

Occupancy requires a calibration that matches both the active camera and the
selected projector display. Camera-only operation is not supported: when the
calibrated display is unavailable or no saved calibration matches both
devices, occupancy shall pause and the condition shall be reported
non-modally. A loaded saved calibration shall keep occupancy frozen until the
alignment monitor has checked and, when necessary, remapped the saved camera
grid against the live board. The application shall not substitute another
display's projector mapping or automatically open an unrelated screen.

## 5. Coordinate Systems

### 5.1 Physical coordinates

Physical board coordinates use:

- Rows `0` through `7`.
- Columns `0` through `7`.
- The orientation of the camera-space calibrated grid.

### 5.2 UI coordinates

UI coordinates use:

- Row `0` at the top of the displayed chessboard.
- Column `0` at the left of the displayed chessboard.
- Existing board-orientation settings and rotation helpers.

Before rendering, every physical occupied square shall be transformed to its
UI square using the inverse of the existing UI-to-physical rotation. No new
independent orientation state shall be introduced.

When both starting sides retain at least eight confirmed pieces, the
application shall compare the central luminance of the confirmed pieces on
those two sides. Square-color parity supplies two rotations that differ by 180
degrees; the brighter piece side shall be assigned to White and mapped to the
bottom two UI ranks. This permits orientation recovery when a small number of
starting pieces are missed or have moved. If either side has insufficient
evidence or side contrast is insufficient, the saved complete orientation
shall remain unchanged and projection shall remain unavailable.

## 6. Canonical Board Image

Each accepted camera frame shall be rectified into a canonical top-down board
image.

The initial canonical image shall:

- Be square.
- Use at least 80 pixels per physical square.
- Preserve color information.
- Map the calibrated 9x9 grid to exact equal-size canonical cells.

A recommended initial size is 800x800 pixels, producing 100x100-pixel square
regions.

Rectification shall use the confirmed camera grid. It shall not use the
projector grid.

## 7. Square Analysis Regions

Each canonical square shall define:

- A full square region.
- A central analysis region.
- A boundary exclusion margin.

The boundary exclusion margin shall initially be 12% to 18% of square width.
It prevents printed grid lines, neighboring pieces, and small geometric errors
from dominating the classification.

The classifier may additionally use:

- A smaller center region.
- Four quadrant regions.
- Edge and texture maps.
- A shadow-sensitive region.

No sample outside the square polygon shall contribute to that square's
occupancy score.

## 8. Empty-Square Appearance Model

### 8.1 Separate parity models

Light and dark squares shall have separate empty-appearance models.

Each model may include robust estimates of:

- Median color in CIELAB or another illumination-tolerant color space.
- Luminance distribution.
- Color-distance distribution.
- Gradient and edge density.
- Local texture energy.
- Fraction of pixels inconsistent with an empty square.

Mean-only RGB comparison is insufficient because dark pieces on dark squares
and light pieces on light squares may have similar average brightness.

### 8.2 Starting-position initialization

For the first milestone, the application may assume that occupancy detection
is initialized while the physical pieces are in the standard starting
position.

The middle four logical ranks provide 32 known-empty reference squares. The
detected quarter-turn parity shall map those logical ranks into physical
camera-grid coordinates before model construction:

- At zero or two quarter turns, physical rows `2` through `5` provide the
  references.
- At one or three quarter turns, physical columns `2` through `5` provide the
  references.

These samples shall initialize both light-square and dark-square empty models.
The detector shall discard and rebuild its empty models whenever the supplied
quarter-turn parity changes. It shall never train the empty model from fixed
physical rows when the logical middle ranks run through physical columns.

The known-empty references establish appearance only. The occupied outer
ranks shall still be classified from image evidence; they shall not be marked
occupied merely because the standard starting position expects pieces there.

If the middle-rank samples are inconsistent, obstructed, or insufficient, the
result shall be inconclusive rather than silently producing the expected
starting-position occupancy.

### 8.3 Model updates

After initialization, an empty model may be updated only from squares that:

- Have been confidently empty for multiple stable observations.
- Are not close to current motion or occlusion.
- Do not contain strong shadows, glare, or clipping.
- Agree with the current square-parity model.

Occupied squares shall never update the empty model.

## 9. Occupancy Features

Each square shall produce an occupancy feature vector. The initial feature set
should include:

1. Robust color distance from the appropriate empty parity model.
2. Fraction of anomalous pixels in the central analysis region.
3. Edge density above the empty-square baseline.
4. Texture-energy difference from the empty-square baseline.
5. Difference between center and outer interior regions.
6. Saturated, clipped, or deep-shadow pixel fraction.

Additional evidence may later include:

- Piece-surface masks obtained from coded-pattern plane residuals.
- Background subtraction from a separately captured empty-board template.
- Learned lightweight image features.

The first implementation should remain deterministic and explainable. A neural
network is not required.

## 10. Classification

Each physical square shall be classified as:

- `empty`
- `occupied`

### 10.1 Adaptive thresholds

Thresholds shall be derived from the robust distribution of known-empty
samples rather than from fixed raw pixel values.

For example, each normalized feature may be expressed relative to the median
and median absolute deviation of matching-parity empty squares.

### 10.2 Decision policy

A square shall be `occupied` only when:

- Its combined occupancy score exceeds the occupied threshold.
- At least two independent feature families support occupancy.
- Visibility and image-quality checks pass.

A square shall be `empty` only when:

- Its score remains within the accepted empty distribution.
- Visibility and image-quality checks pass.

Intermediate per-square scores shall be resolved with a conservative binary
threshold. Weak evidence is empty; sufficiently strong evidence is occupied.
Motion, large obstruction, or invalid geometry rejects the complete frame
instead of producing an incomplete square map.

Tall pieces may project a silhouette into an orthogonally adjacent square. A
weak occupied candidate shall be downgraded to `empty` when an adjacent
occupied square has both strong evidence and at least twice its occupancy
score. Two adjacent squares with independently strong evidence shall remain
occupied.

## 11. Temporal Stabilization

Occupancy shall update only while the board tracker reports a stable board.

The initial temporal policy shall:

1. Display the first stable observation immediately as provisional.
2. Collect three stable observations for confirmation.
3. Require a square to receive the same binary classification in at least two
   of the three observations.
4. Freeze the last confirmed occupancy while motion or occlusion is present.
5. Resume accumulation only after stability returns.
6. Avoid changing a confirmed square because of one disagreeing frame.

A newly stabilized complete result shall replace the previously displayed
result atomically. Dots shall not appear or disappear one square at a time
while a hand is moving.

## 12. UI Component Contract

The chessboard widget shall expose an occupancy setter equivalent to:

```python
set_occupied_squares(squares: Collection[tuple[int, int]]) -> None
```

The widget shall:

- Validate every row and column.
- Store a copied immutable set.
- Repaint only when the set changes.
- Render dots without changing click handling.
- Preserve occupancy when square selection changes.
- Clear occupancy when explicitly requested.

The widget shall not:

- Analyze camera frames.
- Decide board orientation.
- Infer occupancy confidence.
- Modify projector state.

## 13. Application Integration

The camera application shall own an occupancy controller or detector separate
from the chessboard widget.

The integration flow shall be:

1. Receive a fresh camera frame.
2. Update board tracking.
3. If stable and calibrated, rectify and classify the frame.
4. Update the temporal occupancy accumulator.
5. On confirmation, rotate physical coordinates into UI coordinates.
6. Call the chessboard widget occupancy setter.
7. Update the occupancy status text.

Occupancy processing should be rate-limited if full camera frame rate is
unnecessary. A target of 2 to 5 analyzed stable frames per second is
sufficient for the first milestone.

## 14. Calibration and Failure Behavior

### 14.1 Calibration changes

When calibration is replaced or remapped:

- Suspend occupancy accumulation.
- Discard observations collected under the previous geometry.
- Preserve the last confirmed dots only while the new geometry is being
  validated.
- Recompute occupancy after the new calibration becomes verified.

If calibration is invalidated, occupancy dots shall be cleared and the status
shall report that occupancy is unavailable.

### 14.2 Motion and obstruction

A hand, backpack, captured piece outside the board, or other temporary
obstruction shall not create new occupied squares.

During motion or occlusion:

- Do not update the empty model.
- Do not accept new square classifications.
- Keep the last confirmed UI state.
- Report that occupancy is frozen.

### 14.3 Inconclusive initialization

Initialization shall fail explicitly when:

- Fewer than the required known-empty references are visible.
- Light and dark empty models cannot be separated reliably.
- Too many middle-rank squares are anomalous.
- Board orientation is unresolved.
- The camera grid is unavailable or unverified.

An inconclusive attempt shall not display the hard-coded starting position.

## 15. Diagnostics

For a requested diagnostic capture, save or expose:

- Original camera frame.
- Canonical rectified board image.
- Per-square analysis regions.
- Empty-model reference squares.
- Per-square feature values.
- Per-square occupancy scores.
- Per-square classification and confidence.
- Frame rejection and confidence reasons.
- Physical-to-UI coordinate mapping.
- Final occupied-square set.

The normal UI should provide enough status information to distinguish an
empty result from an unavailable or inconclusive result.

## 16. Tests

### 16.1 Unit tests

Tests shall cover:

- Canonical extraction of all 64 square regions.
- Boundary-margin exclusion.
- Separate light and dark empty models.
- Occupied and empty threshold behavior.
- Black piece on a dark square.
- Light piece on a light square.
- Shadow and glare rejection.
- Saturation and clipping rejection.
- Physical-to-UI orientation conversion.
- Occupancy setter validation.
- Dot rendering and coexistence with square selection.

### 16.2 Temporal tests

Tests shall cover:

- Immediate provisional occupancy from the first stable observation.
- Two agreeing observations among three.
- One disagreeing observation among three.
- Rejected whole-frame observations.
- Motion between stable observations.
- Freeze and resume behavior.
- Atomic replacement of the displayed set.

### 16.3 Synthetic and recorded-frame tests

Tests shall include:

- Standard starting position.
- Standard starting position at all four quarter-turn orientations.
- One piece removed from the starting position.
- One piece moved into a middle-rank square.
- Several midgame occupancy layouts.
- Uneven illumination.
- Piece shadows.
- Partial hand occlusion.
- Small board-localization changes within accepted tracking limits.

## 17. Acceptance Criteria

The first occupancy milestone is accepted when:

1. A standard starting position produces 32 occupied UI squares.
2. Occupancy is derived from camera evidence, not hard-coded placement.
3. Removing one piece removes exactly one dot after stabilization.
4. Moving one piece changes exactly the expected source and destination dots.
5. No dots change while a hand is moving over the board.
6. Rejected observations do not erase the last confirmed result.
7. All four supported board orientations map physical squares correctly.
   Each orientation shall also produce 32 occupied squares without training
   the empty model from an occupied starting rank.
8. Occupancy dots remain visible while square selection still works.
9. Calibration invalidation clears the dots.
10. The detector reports an inconclusive state instead of inventing a complete
    occupancy map when evidence is insufficient.

Rendering known piece identities and moving those identities between stable
occupied squares are specified separately in
[Chess-Piece UI Rendering and Move Synchronization](./chess-piece-ui-tracking-specification.md).

## 18. Implementation Milestones

The experimental implementation rectifies the verified camera grid to an
800x800 board, excludes a 15% boundary margin, builds separate robust light-
and dark-square models from the middle four ranks, and combines color anomaly,
edge density, luminance spread, and center-to-border contrast. The first
observation is displayed provisionally. Two-observation agreement within a
three-observation window confirms the result.

On recorded C930e frames, it classifies a standard setup as 32 occupied and 32
empty squares. In the tall-king regression frame, weak silhouette spillover
is resolved as empty while the square containing the base remains occupied.
Further live-frame and piece-movement validation remains required. The
starting-position frame also resolved the bright White side to the UI bottom
with strong side contrast.

### Milestone 1: UI rendering

- Add occupied-square state to the chessboard widget.
- Render large black center dots.
- Preserve clicking, coordinate labels, and selection highlighting.

Implemented.

### Milestone 2: Canonical square extraction

- Rectify the board from the confirmed camera grid.
- Extract 64 interior square regions.
- Add diagnostic visualization.

Rectification and extraction are implemented. Saved occupancy diagnostics
remain pending.

### Milestone 3: Starting-position occupancy

- Build light and dark empty models from the middle four ranks.
- Compute deterministic occupancy features.
- Classify every square as empty or occupied.

Implemented with orientation-aware logical middle-rank selection. Automated
tests cover all four quarter-turn orientations and model rebuilding after a
parity change. Replay of the recorded 90-degree hardware frame detects all 32
starting pieces.

### Milestone 4: Temporal stability

- Connect classification to the stable-board tracker.
- Accumulate several observations.
- Freeze through motion and obstruction.

Multi-observation accumulation and image-motion freezing are implemented.
Direct reuse of the board tracker's state remains pending.

### Milestone 5: Hardware validation

- Validate the starting position.
- Remove and move individual pieces.
- Tune only normalized feature thresholds.
- Record difficult same-color piece/square and shadow cases.

### Milestone 6: Move-detection foundation

- Emit stable observations before and after physical changes. Implemented.
- Feed stable occupancy into starting-position detection and reset,
  saved-position restoration, physical reconciliation, Lichess position
  adoption, and UI occupancy updates. Implemented.
- Keep detailed legal-move scoring in the reference-image tracker, where
  appearance evidence can distinguish transitions that binary occupancy alone
  cannot resolve. Implemented as an intentional subsystem boundary rather than
  pending occupancy work.

## 19. Open Decisions

- Exact canonical resolution.
- Exact square-boundary exclusion percentage.
- Occupancy feature weights.
- Empty and occupied thresholds.
- Whether occupancy analysis starts automatically or through an explicit UI
  control.
- How long a confirmed occupancy result may remain frozen before being marked
  stale.
