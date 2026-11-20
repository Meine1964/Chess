# Projector-to-Chessboard Calibration Specification

**Version:** 0.13
**Status:** Initial implementation validated on hardware  
**Platform:** Python on Windows  
**Camera:** Logitech Webcam C930e at 1920x1080  
**Projector:** MPR 600 over HDMI at 1280x720

## 1. Purpose

This document specifies how the application shall measure the relationship
between the HDMI projector, USB camera, and physical chessboard. The resulting
calibration shall allow the projector to illuminate each of the 64 physical
squares accurately.

This document refines the projector-calibration requirements in
`projector-camera-chessboard-specification.md`. The initial implementation
shall use Gray-code structured light for measurement and independently
projected targets for validation.

The proposed
[Single-Shot Coded-Pattern Chessboard Calibration](./single-shot-coded-pattern-calibration-specification.md)
defines an alternative that fits the chess lattice directly in projector
space. This document and the implemented Gray-code path remain the preserved
fallback until that alternative passes hardware acceptance.

### 1.1 Implementation status

The initial calibration path is implemented with:

- Explicit projector selection with display name, manufacturer/model,
  resolution, and desktop position.
- Exact frameless placement on the selected display without an operating
  system fullscreen transition.
- A safe black state on startup, cancellation, timeout, and failure.
- Projector intensity 255. Live empty-board evidence under daylight showed
  that intensity 112 produced a median response of only two grayscale levels
  despite zero saturation, so calibration uses the available optical
  headroom rather than weakening response thresholds.
- A 1000 ms settling interval and three averaged camera frames per state. The
  interval accommodates measured HDMI display latency that otherwise allowed
  captures to contain the preceding projector pattern.
- A three-second watchdog for every capture state.
- Gray-code capture with the optically unresolvable least-significant X and Y
  bits omitted.
- Two-projector-pixel coordinate bins for the decoded measurements.
- Dense RANSAC homography fitting and local 9x9 grid corrections.
- Nine independently projected validation targets on piece-clear planar
  patches for the standard starting position.
- Up to two bounded local-mesh refinement iterations.
- Calibrated 8x8 grid and 64 inset-square projection modes.
- Automatic storage of the latest successful calibration in
  `calibrations/latest.json`.
- Strict restoration of the saved calibration when camera identifier, camera
  resolution, projector name, and projector resolution all match.
- Manual quick realignment that fits a candidate from five projected targets
  and measures sufficiency with four independent validation targets.
- Separate validation-only action that reuses the active mapping unchanged and
  runs the nine distributed validation targets without Gray-code capture,
  fitting, or refinement.
- Up-to-five-Hz non-disruptive continuous board remapping and active-square
  projector verification.
- An interactive 8x8 board using the classic Lichess brown palette.
- Projection of one inset physical-square polygon when the corresponding
  interface square is clicked.
- Static cyan projection on dark physical squares and related static azure
  projection on light physical squares for strong contrast without a
  distracting hue change or animation.

The least-significant bits were removed after hardware measurement showed that
one-projector-pixel stripes produced a median camera difference of only 4
levels, compared with 22-47 levels for the retained bits. This is an optical
resolution limit rather than a software decoding failure.

The first successful hardware run measured:

- 100.0% projector coverage over the board.
- 54.2% valid dense decoded board pixels.
- 100.0% RANSAC inliers.
- 0.821 projector pixels RMS mapping error.
- 3.614 camera pixels maximum independent validation error.
- 8.0% maximum error as a fraction of local square width.

The current implementation restores a strictly matching saved calibration and
provides **Quick realign / assess**. Automatic drift-triggered recalibration
remains intentionally disabled.

## 2. Scope

The calibration feature shall:

1. Verify that the projector illuminates the complete board.
2. Measure which projector pixel illuminates each visible point on the board.
3. Derive projector coordinates for all 81 board grid intersections.
4. Construct a four-sided projector polygon for each of the 64 squares.
5. Project full-board and individual-square test patterns.
6. Measure alignment independently after calibration.
7. Reject incomplete, unstable, or inaccurate calibrations.
8. Save successful calibrations, restore only exact hardware/resolution
   matches, and provide measured quick realignment with held-out validation.

The first implementation assumes:

- The board is empty.
- The board detector has produced a valid 9x9 grid.
- The camera and projector are mounted on stands and remain fixed for the
  current experiments.
- The board remains stationary during each calibration capture.
- A saved projector-to-camera mapping remains reusable while the camera,
  projector, and calibrated board plane remain unchanged.
- The projector is configured as an extended Windows display.
- The selected projector display operates at 1280x720.

Chess rules, move recognition, and calibration with pieces present are outside
the scope of this document.

## 3. Coordinate Systems

### 3.1 Camera coordinates

Camera image pixels:

```text
(camera_x, camera_y)
```

For the current camera mode, the expected range is `(0, 0)` through
`(1919, 1079)`.

### 3.2 Projector coordinates

Pixels in the projector framebuffer:

```text
(projector_x, projector_y)
```

For the MPR 600, the expected range is `(0, 0)` through `(1279, 719)`.

### 3.3 Board coordinates

The board grid contains 9x9 intersections:

```text
(column, row), where column and row are 0 through 8
```

The square at `(column, row)` is bounded by grid intersections:

```text
(column, row)
(column + 1, row)
(column + 1, row + 1)
(column, row + 1)
```

Chess notation is not required for geometric calibration.

## 4. Calibration Outputs

A successful calibration shall produce:

- The 9x9 board grid in camera coordinates.
- A robust global camera-to-projector homography.
- Dense or sampled camera-to-projector measurements on the board plane.
- A 9x9 projector-space grid.
- A projector-space quadrilateral for each of the 64 squares.
- Projector coverage, decoding, fitting, and validation metrics.
- Hardware, resolution, and timestamp metadata.

The 64 squares shall not be represented as axis-aligned projector rectangles.
Perspective and optical distortion normally make each square a general
quadrilateral.

## 5. Calibration State Machine

Calibration shall use explicit states:

1. `IDLE`
2. `PREPARING`
3. `BOARD_CAPTURE`
4. `COVERAGE_CAPTURE`
5. `STRUCTURED_LIGHT_CAPTURE`
6. `DECODING`
7. `MAPPING`
8. `VALIDATING`
9. `PREVIEW`
10. `SUCCEEDED`
11. `FAILED`
12. `CANCELLED`

While calibration is active:

- The user shall be able to cancel at any time.
- Cancellation or failure shall immediately make the projector black.
- Unexpected exceptions shall not leave a bright pattern displayed.
- The interface shall show the current phase and progress.
- A calibration shall not be exposed as usable before validation succeeds.

## 6. Hardware Preparation

Before measurement, the application shall:

1. Confirm that the selected camera is available.
2. Confirm that the selected projector display is available.
3. Confirm the active camera and projector resolutions.
4. Open a borderless projector window.
5. Set the complete projector frame to black.
6. Allow camera exposure and projector output to stabilize.
7. Lock camera exposure, focus, white balance, and gain when supported.
8. Warn when settings cannot be locked.

The application shall detect display removal or resolution changes during
calibration and fail safely.

## 7. Board Measurement

With the projector black, the application shall:

1. Average multiple camera frames to form a black-reference image.
2. Detect the empty chessboard.
3. Require a complete 9x9 grid in camera coordinates.
4. Store board confidence and reprojection error.
5. Construct a board mask from the outer four grid intersections.
6. Calculate a representative local square width in camera pixels.

The board mask should be slightly eroded for structured-light fitting so that
the wooden border and unrelated background surfaces are excluded.

Calibration shall fail if the board cannot be detected reliably.

## 8. Projector Coverage Measurement

### 8.1 Captures

The application shall capture:

- A projector-black image.
- A uniform illuminated image at a configurable intensity.

Several stabilized camera frames shall be averaged for each state. Initial
testing should use a medium intensity before full white to avoid camera
saturation.

### 8.2 Response image

The projector response shall be calculated from the difference between the
illuminated image and the black-reference image. Measurements shall be limited
to the board mask.

The coverage check shall determine:

- The percentage of board pixels with measurable projector response.
- Response at every square center.
- Response near the four board corners.
- Saturated-pixel percentage.
- Response strength on both light and dark board squares.

### 8.3 Coverage acceptance

Calibration shall stop before Gray-code capture when:

- A board corner is outside the illuminated projector region.
- Any physical square has insufficient measurable response.
- A significant area is saturated.
- The signal-to-noise ratio is insufficient for reliable decoding.

The failure message shall distinguish physical coverage, low brightness,
excessive brightness, and unstable capture.

## 9. Gray-Code Structured-Light Capture

### 9.1 Pattern set

The projector shall encode its horizontal and vertical pixel coordinates with
binary-reflected Gray code.

At 1280x720:

- Horizontal coordinates require 11 bits.
- Vertical coordinates require 10 bits.
- Every bit shall be projected as a normal pattern and its inverse.

The optical system cannot resolve the one-projector-pixel stripes represented
by the least-significant bit of either axis. The initial implementation omits
those two bits and decodes the center of each two-pixel bin. It captures:

- 20 horizontal pattern states.
- 18 vertical pattern states.
- One black state.
- One uniform illuminated state.
- 40 states in total.

Pattern generation shall use the actual projector framebuffer size rather than
assuming 1280x720 internally.

### 9.2 Pattern display

For each state, the application shall:

1. Render the complete pattern.
2. Wait for the projector and camera pipeline to settle.
3. Discard frames that can still contain the previous pattern.
4. Capture and average a configurable number of stable frames.
5. Record capture timing and sequence number.

The implementation shall not assume that showing a Qt frame makes it
immediately visible to the camera.

### 9.3 Complementary decoding

For each bit, the normal and inverse images shall be compared. A camera pixel
is valid for that bit only when the absolute difference exceeds a threshold
derived from measured camera noise and projector response.

The sign of the difference determines the Gray-code bit. The completed Gray
code shall be converted to a binary projector coordinate.

Complementary pairs are required because they reduce sensitivity to:

- Alternating board colors.
- Wood grain and printed texture.
- Ambient illumination.
- Unequal surface reflectivity.
- Fixed camera color response.

Pixels shall be rejected when they are saturated, underexposed, inconsistent,
outside the board mask, or insufficiently modulated.

### 9.4 Decode result

Each accepted camera pixel yields:

```text
(camera_x, camera_y) -> (projector_x, projector_y)
```

The decoder shall report:

- Valid decoded-pixel count and percentage.
- Valid coverage per physical square.
- Modulation strength distribution.
- Rejection counts by reason.
- Spatial holes in the decoded board area.

## 10. Geometric Mapping

### 10.1 Global model

The application shall fit a camera-to-projector homography with RANSAC using
well-distributed decoded correspondences.

The fit shall:

- Sample measurements across the complete board.
- Avoid allowing one image region to dominate.
- Reject outliers.
- Report inlier count, inlier percentage, mean error, RMS error, and maximum
  error.

The global homography provides an initial mapping and fallback estimate.

### 10.2 Local grid estimates

A single homography cannot represent camera and projector lens distortion.
The application shall therefore estimate projector coordinates for all 81
board intersections using nearby decoded measurements.

For each camera-space grid intersection:

1. Select valid decoded samples in a local neighbourhood.
2. Fit a robust local affine mapping when enough samples exist.
3. Evaluate the local mapping at the exact intersection.
4. Compare the result with the global homography estimate.
5. Fall back to the global estimate only when local support is insufficient.

Outer intersections may use controlled local extrapolation. Extrapolation
distance and uncertainty shall be recorded.

The implementation may later replace local affine estimates with a smooth
mesh or distortion model, but it shall retain a global model for diagnostics
and failure recovery.

### 10.3 Grid validity

The resulting 9x9 projector grid shall be rejected when:

- A point falls outside the projector framebuffer.
- Rows or columns cross or reverse direction.
- A square has negligible or negative area.
- Adjacent cell sizes change implausibly.
- Local estimates disagree excessively with measured structured-light data.
- The grid requires unsupported extrapolation.

## 11. Projected Square Geometry

Each square shall be rendered from four adjacent points in the projector grid.
The renderer shall support:

- All 64 square polygons.
- One selected square.
- A thin full-board grid.
- An alternating checkerboard.
- A full-board fill.
- Configurable color and brightness.
- Configurable inset from physical gridlines.

The initial filled-square test should inset each polygon by approximately 5%
in board coordinates. The inset shall be adjustable. It shall be applied
before conversion to projector coordinates so perspective remains correct.

The default projector background shall remain black.

## 12. Independent Validation

### 12.1 Validation targets

Calibration measurements shall be validated using projected targets that are
not merely a rendering of the Gray-code sequence.

The implemented validation set is a 3x3 arrangement of square centers at board
coordinates `0.5`, `3.5`, and `7.5` on each axis. Using square centers avoids
reflectance-biased centroids from targets that straddle a light/dark boundary.

The application shall also support validation at all 64 square centers.

Targets should be compact filled circles, squares, or crosses on a black
background. Their size and color shall maximize localization accuracy without
covering adjacent squares.

### 12.2 Target measurement

For every validation position:

1. Capture or refresh a black reference.
2. Project the target at the calculated projector position.
3. Wait for stable camera frames.
4. Average multiple frames.
5. Difference the illuminated image from the black reference.
6. Locate the target in camera coordinates.
7. Compare its measured position with the expected board position.

The expected camera position shall be derived from board coordinates, not
from the observed target.

### 12.3 Error metrics

Validation shall report:

- Error for every target in camera pixels.
- Mean, RMS, 95th-percentile, and maximum error.
- Error as a fraction of local square width.
- Error grouped by board center, edges, and corners.
- Any target that was not detected or was ambiguous.

### 12.4 Initial acceptance criteria

A calibration may succeed only when:

- All nine initial validation targets are detected unambiguously.
- Projector illumination covers at least 70% of the detected playing surface.
- Every board quadrant has at least 35% average illumination response;
  individual piece-obstructed squares may have lower response.
- Decoded board-plane support passes the distributed inlier requirements.
- Every projected target lands in its intended physical square.
- Maximum validation error is less than 10% of local square width.
- No projected square substantially overlaps an adjacent square.
- Projector-grid validity checks pass.

The desired later target is a maximum error below 3-5% of local square width.

A missing target is a failure, not a zero-error measurement.

## 13. Iterative Refinement

When validation is reliable but exceeds the target error:

1. Add expected and measured validation positions as correction evidence.
2. Update the local projector grid or correction mesh.
3. Repeat the independent validation sequence.
4. Stop when the configured threshold is reached, improvement becomes
   negligible, or the maximum iteration count is reached.

The implementation permits at most two refinement iterations.

Calibration shall fail rather than save a result when refinement does not
converge.

## 14. Drift Detection and Recalibration

### 14.1 Continuous non-disruptive monitoring

After calibration is completed or restored, the application:

1. Runs empty-board detection in a background worker up to five times per
   second. A new task is skipped while the previous detection is still running.
2. Compares all 81 current camera-grid points with the calibrated grid.
3. Remaps the 9x9 projector grid after approximately 1.5 camera pixels of
   movement and immediately redraws any active square.
4. Transfers the calibrated local correction field to the moved board using a
   smooth quadratic camera-space model.
5. Geometrically warps the stored black reference with the board motion.
6. While a selected square is projected, locates its illuminated region and
   compares its centroid with the expected physical-square center.
7. Accepts in-plane board movement when the updated square remains aligned.
8. Requires two consecutive projected-square or remapping failures before
   invalidation.
9. Silently retains the last confirmed mapping while a hand, piece, or other
   temporary obstruction prevents board detection. A missed detection never
   constitutes evidence of board movement or calibration loss.
10. After a positive remap, follows the board immediately but waits for five
   consecutive visible stable-board checks before resuming optical failure
   counting.
11. Classifies failure after a camera-grid remap as camera calibration loss and
    failure with a stable camera grid as projector calibration loss.
12. Makes the projector black and changes the corresponding persistent UI
    indicator to red after confirmed calibration loss. The last optically
    verified mesh remains available and recalibration remains a manual action.

Monitoring pauses for full calibration, focus patterns, full-board grids, and
all-square fills because those modes intentionally
change or obscure the evidence used by the normal monitor.

A projector emits no measurable signal while completely black. Therefore the
non-disruptive monitor can detect projector movement only while a square is
illuminated or when the next square is selected. Periodic visible probes and
always-on anchor marks are intentionally not used.

### 14.2 Manual quick realignment and sufficiency measurement

A quick realignment sequence shall project:

- Five fitting targets at four internal corner-region grid intersections and
  the board center.
- Four independent validation targets at internal edge-region grid
  intersections.

The targets shall remain on the top playing plane while avoiding both the
physical board edge and the centers occupied by pieces in the standard
starting position. Their projected radius shall be limited to 14% of local
square width. If a target is not detected, the validator shall try one nearby
piece-clear location in the same board region before failing. After available
mesh refinement, a detected target with excessive residual error shall also
be retried at its nearby location to reject shifted reflections from pieces.
The outer validation rows shall use board coordinates 2 and 6, bordering the
unobstructed middle half rather than the narrow gaps between the pawn and back
ranks.

The fitting targets shall calculate a candidate camera-to-projector homography
without changing the active verified calibration. The saved local correction
field shall be transferred to the candidate mapping.

The implemented manual quick realignment:

1. Captures a current projector-black frame.
2. Redetects the complete empty board.
3. Remaps the saved projector mesh to the current camera-grid coordinates.
4. Projects five fitting targets and measures their camera centroids.
5. Fits a candidate homography while retaining the saved local mesh
   corrections.
6. Refreshes the black reference and projects four targets that were not used
   for fitting.
7. Accepts and saves the candidate only when every independent validation
   target remains below 10% of local square width.
8. Otherwise reports that full calibration is required and restores the
   previous verified calibration without replacing or deleting it.

A complete calibration is required after:

- Camera movement.
- Projector movement.
- Board movement that changes its depth or tilt relative to the calibrated
  plane.
- Camera or projector resolution changes.
- Display reconnection that changes geometry.
- Quick realignment insufficiency.

Moving the board within the same calibrated plane does not necessarily require
a complete Gray-code calibration. The application may redetect the 9x9 board
grid and apply the existing camera-to-projector mapping, followed by quick
realignment assessment. Any candidate that fails independent validation shall
trigger a complete calibration recommendation.

Move indicators shall be suspended while verification or calibration patterns
are displayed.

## 15. Persistence

A successful calibration shall be stored with:

- Schema version.
- Timestamp.
- Camera stable identifier and display name.
- Camera resolution and relevant camera settings.
- Projector display identifier, name, geometry, and resolution.
- Board detector confidence, method, and camera-space 9x9 grid.
- Black/white coverage metrics.
- Gray-code capture parameters.
- Decode metrics.
- Global homography and fit metrics.
- Projector-space 9x9 grid.
- Local-estimation support and uncertainty per grid point.
- Validation target results and aggregate metrics.
- Refinement iteration count.

Saved calibration data is treated as invalid if camera identifier, camera
resolution, projector name, or projector resolution does not match. A matching
record is restored with an explicit warning to recalibrate after hardware or
board-plane movement. Manual measured quick realignment is available;
automatic recalibration is intentionally not performed.

## 16. Diagnostics

Camera snapshots currently have bounded timestamped diagnostics containing the
original image, optional board overlay, and detection JSON. Per-attempt
calibration diagnostics remain planned and should contain:

- Black-reference image.
- Uniform illuminated image.
- Projector-response image.
- Board detection overlay.
- Captured Gray-code images or a configurable retained subset.
- Valid-decoding mask.
- Decoded projector X and Y visualizations.
- Homography inlier and residual visualizations.
- Camera-space and projector-space grid overlays.
- Every validation target image and difference image.
- A machine-readable JSON report.

Diagnostic retention shall be bounded. Write failures shall be reported
explicitly and shall not be presented as successful diagnostic storage.

## 17. User Interface

The implemented calibration interface provides:

- Camera and projector selectors.
- A **Full projector calibration** action.
- A **Cancel calibration** action.
- An always-available **Projector black** action.
- A calibration-independent **Focus pattern** action using a native-resolution
  black/white checkerboard, border, and central cross.
- Current phase and progress.
- Current pattern number and total pattern count.
- Clear success and failure messages.
- A full-board grid preview.
- An all-squares fill preview.
- Automatic save after successful validation.
- Strict matching-calibration restore.
- **Quick realign / assess** action.
- Automatic standard-orientation detection from the median light/dark parity
  of all 64 inset square interiors while the projector is black. Reversed
  parity applies a remembered 90-degree coordinate rotation consistently to
  selected-square projection and optical verification.
- Selectable continuous-alignment monitor status.
- The interactive square-selection board described below.

Coverage/decode detail views, alternating checkerboard projection,
previous/next stepping, manual save/discard actions, and automatic periodic
verification remain planned.

### 17.1 Interactive square selection

After a matching calibration is completed or restored, the control window
shall show an 8x8 board with:

- Lichess classic light squares in `#f0d9b5`.
- Lichess classic dark squares in `#b58863`.
- Files `a` through `h` and ranks `8` through `1`.
- Top-left interface square mapped to projector grid cell `(0, 0)`.
- Bottom-right interface square mapped to projector grid cell `(7, 7)`.
- A visible selected-square highlight.

Clicking a square shall:

1. Open the selected projector display black when necessary.
2. Render only the matching calibrated inset quadrilateral using:
   - Bright cyan `#00ffff` on dark physical squares.
   - Azure `#007fff` on light physical squares.
3. Show the selected chess coordinate and internal row/column.

Clicking the selected square again or selecting **Black** shall clear the
projection. Square selection shall remain disabled until a matching
calibration is available.

The application shall prevent changing the camera or projector selection while
calibration is capturing patterns.

## 18. Failure Handling

Calibration shall fail safely and explain the cause when:

- The board cannot be detected.
- The projector does not cover the complete board.
- Projected light is too weak or saturated.
- Camera frames are unstable or stale.
- Too few Gray-code pixels can be decoded.
- Decoded coverage is uneven or incomplete.
- The geometric model is invalid.
- Validation targets are missing or ambiguous.
- Alignment exceeds the configured tolerance.
- Hardware disappears or changes resolution.
- The board, camera, or projector appears to move.

Every failure path shall make the projector black before returning control to
the user.

## 19. Implementation Stages

### Stage 1: Pattern rendering and synchronized capture - implemented

- Render black, uniform, Gray-code, and inverse Gray-code patterns.
- Establish frame-settling and averaging behavior.
- Raw per-attempt calibration diagnostics remain planned.

### Stage 2: Coverage and Gray-code decoding - implemented

- Measure full-board projector response.
- Decode camera-to-projector correspondences.
- Visualization of valid and invalid decoded regions remains planned.

### Stage 3: Mapping and 64-square rendering - implemented

- Fit the global homography.
- Calculate the local 9x9 projector grid.
- Render arbitrary quadrilateral square polygons.
- Grid, all-square fill, and individual-square previews are implemented.
  Checkerboard preview remains planned.

### Stage 4: Automated validation - implemented

- Project and detect independent targets.
- Calculate alignment metrics.
- Reject inaccurate calibration.
- Add bounded iterative refinement.

### Stage 5: Persistence and manual quick realignment - implemented

- Save successful calibration records.
- Reload matching records.
- Fit five projected targets and independently validate four additional
  targets on request.

## 20. Test Requirements

Automated tests shall cover:

- Gray-code generation for multiple resolutions.
- Gray-to-binary decoding.
- Normal/inverse threshold handling.
- Saturated and low-response rejection.
- Synthetic planar camera/projector transformations.
- RANSAC outlier rejection.
- Local grid estimation.
- Invalid and folded projector grids.
- Construction of all 64 polygons from a 9x9 grid.
- Polygon inset behavior.
- Validation metric calculation.
- Calibration acceptance and rejection boundaries.
- Persistence schema round trips.
- Cancellation and failure transitions to projector black.

Synthetic tests shall include perspective distortion, noise, unequal square
reflectivity, missing decoded regions, and modest optical distortion.

Hardware acceptance testing shall verify:

1. Complete board coverage.
2. Successful Gray-code decoding on both square colors.
3. Correct rendering of all 64 square polygons.
4. Nine-point automated validation.
5. Sequential illumination of all 64 squares.
6. Correct rejection after deliberate hardware movement.

## 21. Initial Completion Criteria

The projector-calibration milestone is complete when:

1. The application calibrates the stationary empty-board setup without manual
   point selection.
2. All 64 physical squares have corresponding projector polygons.
3. A full-board test pattern visibly aligns with the board.
4. Every square can be illuminated individually.
5. Automated validation satisfies the 10% maximum-square-width error limit.
6. Calibration failure always returns the projector to black.
7. Successful calibration can be saved and restored for a strict matching
   hardware/resolution configuration.
8. A restored calibration can be quickly realigned from five projected targets
   and accepted only after four independent targets pass, without rerunning the
   40-state Gray-code sequence.

Automatic periodic verification remains beyond the initial implemented
milestone.
