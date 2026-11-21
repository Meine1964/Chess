# Projector/Camera Chessboard Calibration Specification

**Version:** 2.5
**Status:** Core camera/projector workflow implemented and hardware validated
**Initial platform:** Python on Windows  
**Camera connection:** USB  
**Projector connection:** HDMI

## 1. Purpose

Build a system that:

1. Captures images of a physical chessboard using a USB camera.
2. Determines the location and orientation of the 8x8 board.
3. Determines how projector pixels correspond to physical board locations.
4. Projects an overlay aligned with each of the 64 squares.
5. Detects alignment drift and supports recalibration.
6. Later uses the calibrated overlay to indicate chess moves.

The implemented system now includes markerless empty-board detection,
Gray-code projector calibration, a validated 9x9 projector mesh, saved
calibration restore, full-board projection tests, and clickable
individual-square indication. Piece-aware board localization and calibration
without an empty-board prerequisite are specified in
[Piece-Aware Board Localization and Calibration](./piece-aware-board-localization-specification.md).
An alternative direct projector-space approach is specified in
[Single-Shot Coded-Pattern Chessboard Calibration](./single-shot-coded-pattern-calibration-specification.md).
Camera-derived occupied-square marking on the digital chessboard is specified
in
[Chess-Square Occupancy and UI Marking](./chess-square-occupancy-ui-specification.md).
The initial occupied-square classifier and UI dots are implemented
experimentally. Hardware tuning, chess move recognition, and chess rules
remain future work.
Official SVG piece rendering and initial non-capturing move synchronization
are specified in
[Chess-Piece UI Rendering and Move Synchronization](./chess-piece-ui-tracking-specification.md).
The pinned cburnett assets, standard UI position, and ordinary one-piece move
synchronization are implemented experimentally.

## 2. Hardware

### 2.1 Required hardware

- A physical 8x8 chessboard.
- A Logitech Webcam C930e 1080p USB camera with a view of the entire board.
- An MPR 600 mini projector connected through HDMI, with a 1280x720 output
  resolution and coverage of the entire board.
- A Windows computer running the application.
- Python as the initial implementation language.

### 2.2 Preferred physical arrangement

- The camera and projector are mounted above the board.
- Initial experiments may use a relatively high mounting position to simplify
  coverage and reduce perspective distortion.
- Later experiments will progressively lower, offset, or separate the devices
  to test more difficult viewing and projection angles.
- For the current experiments, the camera and projector are fixed on stands.
  Moving either device invalidates the current geometric calibration and
  requires recalibration.
- The camera and projector may initially be placed close to each other, but
  the design must not depend on them sharing the same position or viewing
  angle.
- The complete outer boundary of the board is visible to the camera.
- The projector image covers the entire board with some margin.
- The physical board has alternating square colors. The background may be
  visually complicated because calibration measurements are restricted to the
  detected board polygon.
- Camera settings such as autofocus and automatic exposure can preferably be
  locked after initialization.

The camera and projector do not need to be mounted perpendicular to the board.
Perspective distortion will be handled in software.

## 3. Coordinate Systems

The system must distinguish four coordinate systems.

### 3.1 Camera coordinates

Pixel positions in captured camera images:

```text
(camera_x, camera_y)
```

### 3.2 Projector coordinates

Pixel positions in the HDMI projector framebuffer:

```text
(projector_x, projector_y)
```

The MPR 600 projector has coordinates from `(0, 0)` to `(1279, 719)` at its
1280x720 output resolution.

### 3.3 Board coordinates

Normalized positions on the physical chessboard:

```text
(board_x, board_y)
```

The complete board occupies:

```text
0 <= board_x <= 8
0 <= board_y <= 8
```

Each unit is one chess square.

### 3.4 Chess coordinates

Named squares such as `a1`, `e2`, and `h8`.

The board orientation must be known before board coordinates can be converted
into chess coordinates. The system shall determine orientation automatically
from the alternating square colors and the rule that each player has a light
square in the right-hand corner.

During empty-board calibration, the initial physical arrangement shall define
which detected board edge is White's side. Given that reference edge, the
light corner identifies White's right side and therefore assigns chess
coordinates to all 64 squares.

The light-corner rule alone leaves a 180-degree ambiguity if the system does
not know which edge belongs to White. In the later piece-aware phase, the
system shall resolve this automatically by detecting the colors and starting
or current placement of the pieces. Existing board details such as printed
coordinates or a logo may provide additional evidence but are not required.
No markers may be added for calibration.

## 4. Calibration Principle

Moving a projected square and observing it with the camera can determine the
relationship between projector pixels and camera pixels.

It does not, by itself, identify where the chessboard is. The system must also
detect the unmarked board directly in the camera image. Added fiducial,
temporary, or permanent calibration markers are not permitted.

Calibration therefore consists of two related measurements:

1. **Board-to-camera mapping:** Locate the chessboard corners and square grid
   in the camera image.
2. **Projector-to-camera mapping:** Project known patterns and locate those
   patterns in the camera image.

Combining these mappings produces the required board-to-projector mapping.

Because the board is approximately planar, each mapping can normally be
represented by a 3x3 perspective transformation called a homography. This
supports translation, rotation, scaling, skew, and perspective distortion,
not merely resizing and translation.

## 5. Functional Requirements

### FR-1: Camera discovery and capture

The application shall:

- Enumerate connected video cameras.
- Exclude integrated, built-in, front-facing, and rear-facing cameras.
- Allow selection only among eligible external USB cameras.
- Keep the application open in a waiting state when no external camera is
  connected, and activate an eligible camera when it is hot-plugged.
- Store the selected camera by stable device identifier and automatically reuse
  it on the next launch when it is still connected.
- Allow the camera to be changed from the live-preview interface without
  restarting the application.
- Display a live preview.
- Allow the user to freeze the exact visible frame with a **Take snapshot**
  action.
- Run board detection on the frozen frame and display the raster over that
  frame.
- Allow the user to return to the live preview with a **Resume live camera**
  action.
- Report the selected resolution and frame rate.
- Detect camera disconnection and report it clearly.
- Prefer the highest stable resolution that provides enough detail to identify
  board intersections.
- Allow camera exposure, focus, white balance, and gain to be configured when
  the camera driver supports them.

The Logitech camera's 1920x1080 mode is the preferred initial capture mode.
Lower modes may be used if required for stable frame rate or camera latency.

### FR-2: Projector selection

The application shall treat the projector as a display device and shall:

- Enumerate available displays.
- Allow selection of the HDMI projector.
- Show an exact-size frameless window at that display's global desktop
  geometry. Operating-system fullscreen transitions are not used because they
  were observed to move the window to a different monitor.
- Use the MPR 600 projector's 1280x720 output resolution.
- Prevent operating-system scaling from changing projection coordinates.
- Keep user-interface controls on the primary monitor.
- Support a completely black projector frame.

Black is the default background because it minimizes unwanted projected light.

### FR-3: Board detection

The implemented calibration uses an empty, alternating-color board. The
surrounding background may be complicated because projector measurements are
restricted to the detected board polygon. Later milestones shall detect and
track the board when some grid intersections, square edges, and colors are
occluded by pieces.

The application shall detect:

- The outer quadrilateral of the chessboard.
- The nine horizontal grid lines.
- The nine vertical grid lines.
- Their 81 intersections, where sufficiently visible.
- The resulting 64 square polygons.
- The board's chess orientation from its corner colors and the known White-side
  reference edge.
- A confidence score for the detected board.
- When more than one complete chessboard is visible, the board whose geometric
  center is closest to the center of the camera image.

The detector must not assume that board squares appear rectangular in the
camera image. Perspective may make them trapezoidal.

The empty-board detector shall use a staged strategy:

1. Attempt strict 7x7 internal-corner detection on the grayscale frame.
2. If strict detection fails, apply CLAHE local contrast normalization to
   compensate for illumination gradients and low-contrast wooden square
   boundaries.
3. Retry internal-corner detection with less restrictive search flags.
4. Fit a homography to the 49 internal corners and extrapolate the complete
   9x9 grid.
5. If internal-corner detection still fails, find quadrilateral playing-surface
   candidates, normalize each candidate to a square view, and verify that its
   64 cell centers form two alternating color populations.
6. Accept the best quadrilateral only when its alternating-color separation
   and consistency exceed explicit thresholds.
7. Retain distinct valid boards, merge repeated detections of the same physical
   board, and select the board center nearest the image center. Detection
   confidence shall not override this spatial selection rule.
8. For quadrilateral detections, measure image-gradient peaks near the seven
   expected internal grid positions in each direction, fit their spacing, and
   extrapolate the true playing-surface boundary. Apply refinement only when
   the fitted spacing, residual error, and edge-alignment score pass explicit
   guards.

This fallback is required because the physical wooden board has visible wood
grain, reflections, and non-uniform illumination that can defeat a detector
intended for high-contrast printed calibration boards. Detection of the
lowest-contrast physical board remains experimental and is explicitly deferred
from the current milestone; failure to detect that board shall not block
projector work with the validated boards.

#### Detection validation

A candidate board is valid only if:

- It contains an 8x8 arrangement.
- Grid intersections are geometrically consistent.
- Adjacent cells have plausible relative sizes.
- Horizontal and vertical lines converge consistently under perspective.
- The detected quadrilateral lies completely within the camera image.
- The board occupies enough image area for reliable calibration.

The initial version may require the user to confirm or correct the four
detected outer corners, but not to place markers. The initial configuration
may identify White's side through the physical setup. Later operation shall
infer White's side from piece colors and positions.

#### Detection with pieces

Detailed requirements are defined in
[Piece-Aware Board Localization and Calibration](./piece-aware-board-localization-specification.md).
The later detector shall:

- Infer the complete board from the visible outer boundary, unoccluded grid
  segments, alternating colors, and temporal information.
- Support continuous board analysis at a configurable rate, initially
  targeting 10 captured frames per second. The live preview frame rate may be
  higher and shall remain independent from the analysis rate.
- Tolerate partial occlusion without treating piece edges as grid lines.
- Detect light and dark pieces sufficiently to determine which board edge
  belongs to White and resolve the board's 180-degree orientation.
- Reuse a recently validated board model only while drift checks confirm that
  it remains aligned.
- Request recalibration when the visible evidence is insufficient or the board
  has moved.

The quadrilateral detector may currently recover a board with pieces when its
outer boundary and enough internal transitions remain visible. This is useful
diagnostic behavior but is not yet a guarantee of piece-aware board detection.

### FR-4: Projector-to-camera calibration

The detailed implemented protocol is specified in
`projector-calibration-specification.md`. Gray-code structured light is the
primary measurement method. Independently projected targets validate and
locally refine the resulting projector mesh.

The implemented sequence:

1. Captures an averaged projector-black reference.
2. Measures full-board projector response at intensity 112.
3. Displays 40 normal/inverse Gray-code states. The optically unresolvable
   least-significant X and Y bits are omitted and decoded as two-pixel bins.
4. Waits 250 ms after every display change and averages three fresh frames.
5. Rejects any capture state that does not receive fresh frames within three
   seconds.
6. Restricts decoding to the detected board polygon.
7. Fits a robust global camera-to-projector homography.
8. Applies local corrections to produce a 9x9 projector mesh.
9. Projects nine independent validation targets on piece-clear planar patches
   for the standard starting position.
10. Applies at most two bounded local-mesh refinement iterations.
11. Accepts only a result below 10% of local square width.

#### Calibration targets

Calibration patterns and validation targets are:

- Displayed at measured intensity 112 to avoid saturation while retaining
  complete board response.
- Large enough to be detected reliably.
- Displayed against a black background.
- Positioned away from the projector image boundaries.
- Sampled at multiple locations across the full board region.

Gray-code stripes provide dense calibration correspondences. Compact filled
circles provide independent validation without sharing geometry with the
calibration patterns.

#### Image differencing

Detection should compare a projected frame with a recent black reference
frame. This helps isolate projected light from:

- The printed board pattern.
- Pieces or other stationary objects.
- Ambient room lighting.
- Camera color inaccuracies.

Multiple camera frames should be averaged after each projection change to
reduce noise and account for camera latency.

### FR-5: Board-to-projector mapping

After both calibrations are known, the system shall calculate projector
coordinates for:

- The four board corners.
- The 81 grid intersections.
- The 64 individual square polygons.
- The center of every square.

Each physical square will normally map to a four-sided polygon in projector
coordinates. It should not be represented only as an axis-aligned rectangle.
This allows the projected overlay to follow perspective distortion accurately.

### FR-6: Projection tests and individual-square indicators

The implemented renderer provides a thin full-board grid, inset fills over all
64 squares, and one selected square. Alternating checkerboard and automatic
all-64 stepping remain planned test modes.

#### Individual-square mode

- Display a Lichess-colored 8x8 board in the control window.
- Map interface `a8` at top left through `h1` at bottom right to the physical
  8x8 mesh.
- Project one inset quadrilateral when its interface square is clicked.
- Use static bright cyan `#00ffff` on dark physical squares.
- Use related static azure `#007fff` on light physical squares.
- Clear the projection when the selected square is clicked again or **Black**
  is selected.
- Show chess notation and the internal row/column for the selection.

#### Full-board grid mode

- Project the outlines of all 64 squares.
- Use thin lines so the physical grid remains visible.
- Allow line width and brightness adjustment.

#### Full-board fill mode

- Fill all 64 square polygons using a high-contrast calibration color.
- Preserve small configurable gaps around physical grid lines.
- Primarily serve as a coverage and clipping test.

#### Alternating mode

- Illuminate alternating squares as a checkerboard.
- Swap illuminated and non-illuminated squares periodically.
- Make one-square alignment errors easier to see.

#### Board-edge activity indicator

- After camera and projector calibration, render a small white circular
  indicator beside h4 with its center half a square outside the h-file.
- Make the indicator available as an overlay in every projector display mode,
  including an otherwise black frame.
- Illuminate the indicator only when camera activity contains a connected
  component with material pixels inside the board, outside the board, and at
  the outer rim of the one-square observation halo. This full connected path
  distinguishes a hand or arm entering from outside from a changed edge piece
  whose silhouette extends only a short distance beyond the playing grid.
- Use an independent rolling camera baseline and illuminate on the first
  qualifying crossing observation so projector content, board-confined
  changes, and activity wholly outside the board do not illuminate the
  indicator.
- Clear the indicator on the first observation without a qualifying boundary
  crossing.
- Keep this diagnostic independent of Lichess manipulation lifecycle state;
  it shall not trigger blackout, move verification, or other game behavior.
- Record every activation and clearing transition with the current camera
  image, active state, current crossing decision, component area, inside-board
  pixels, outside-board pixels, outer-halo-rim pixels, and component bounding
  box. Expose the same latest measurements in runtime state so persistent
  false crossings can be traced to their actual image region.

The operational indicator colors are currently fixed. Brightness, inset, and
color controls remain possible future enhancements.

### FR-7: Alignment measurement

After projecting a test pattern, the camera shall capture the result and
estimate:

- Alignment error at board corners.
- Alignment error at internal grid intersections.
- Average pixel error in the camera image.
- Maximum pixel error.
- Approximate error as a fraction of one board-square width.
- Whether the projection spills significantly beyond the physical board.

#### Initial acceptance criteria

For a successful first prototype:

- Every square is mapped to the correct physical cell.
- Maximum alignment error should be less than 10% of a square width.
- The later target should be less than 3-5% of a square width.
- No projected square should substantially overlap an adjacent square.
- The system must reject calibration rather than claim success when confidence
  is inadequate.

Exact tolerances should become configurable after testing with the real
hardware.

### FR-8: Iterative refinement

The initial homography may be refined iteratively:

1. Calculate an initial board-to-projector transformation.
2. Project targets at selected board intersections or square centers.
3. Detect their actual locations with the camera.
4. Compare expected and observed locations.
5. Update the calibration.
6. Repeat until error is below the configured threshold or no longer improves.

The process shall have:

- A maximum iteration count.
- A convergence threshold.
- Detection-confidence requirements.
- A clear failure result if convergence is not achieved.

### FR-9: Recalibration and drift detection

The application shall support:

- Full calibration at application startup.
- Manual full recalibration.
- Manual quick recalibration.
- Periodic alignment checks.
- Automatic recalibration when measured drift exceeds a threshold.
- Immediate invalidation of calibration when movement is detected or suspected.

#### Quick recalibration

A quick check can briefly project small targets near several board corners and
compare their observed camera positions with expected positions.

If the devices and board have not moved, normal projection continues. If drift
is detected, the application pauses move indication and recalibrates.

The system shall not assume that a previous calibration remains valid merely
because the camera and projector identifiers and resolutions are unchanged.
Saved calibration data provides a starting estimate only and must be verified
against the current physical arrangement.

#### Implemented continuous monitor

The implemented non-disruptive monitor:

- Runs board detection in a background worker up to five times per second.
- Compares all 81 live grid intersections with the calibrated camera grid.
- Remaps all 81 projector vertices after approximately 1.5 camera pixels of
  board movement.
- Immediately redraws an active projected square at the new board position.
- Warps the stored black reference with the board motion so projector
  verification remains meaningful while the board moves.
- Uses the next optical square measurement to distinguish valid in-plane board
  movement from camera movement or board depth/tilt changes.
- Retains the most recent stable projector-black camera frame.
- While one square is illuminated, measures its projected centroid against the
  expected physical-square center to detect projector movement.
- Silently retains the last confirmed board position when detection is
  temporarily blocked by a hand or piece. Absence of a detection is not
  evidence that the board moved.
- After positive movement, immediately follows the board but postpones optical
  failure counting until five consecutive visible checks show a stable board.
- Classifies an optical failure following camera-grid remapping as a camera
  calibration error, and an optical failure with a stable camera grid as a
  projector calibration error.
- Requires two consecutive geometric failures before invalidation.
- Immediately makes the projector black and disables the calibration only
  after confirmed optical or geometric loss.
- Pauses during calibration, focus patterns, and full-board previews that
  intentionally obscure normal monitoring evidence.

Projector movement is not physically observable while the projector is
completely black. In the selected non-disruptive mode it is detected while a
square is active or immediately after the next square is selected. No periodic
visible probe is projected.

Recalibration during a game must avoid confusing calibration patterns with
move indicators. It should occur between moves or during an explicitly visible
calibration state.

### FR-10: Calibration persistence

A successful calibration shall be saved with:

- Camera identifier.
- Camera resolution.
- Projector or display identifier.
- Projector resolution.
- Detected board corners.
- Detected board orientation and the visual evidence supporting it.
- Camera-to-board homography.
- Projector-to-camera homography.
- Derived board-to-projector homography.
- Calibration timestamp.
- Calibration error metrics.
- Relevant camera settings.

A saved calibration may be reused only if hardware and resolution settings
match. It should still undergo a quick alignment check.

### FR-11: Diagnostic snapshot storage

During development, the camera interface shall save diagnostic data whenever
the user takes a snapshot, unless diagnostic saving has been disabled in the
interface.

Each diagnostic capture shall use its own timestamped directory and contain:

- The original lossless camera frame as `original.png`.
- The rendered raster as `overlay.png` when board detection succeeds.
- A `detection.json` file containing the timestamp, camera name, stable camera
  identifier, image resolution, OpenCV version, detection success, confidence,
  reprojection error, and all 81 detected grid points when available.

Failed detections shall be retained because they are needed to improve the
detector. Diagnostic write failures shall be displayed to the user rather than
ignored. The application shall retain at most the latest 100 captures and
remove older capture directories. The generated `diagnostics` directory shall
not be committed to version control.

## 6. Staged Workflow

### Phase 0: Camera-only board detection

1. Enumerate connected cameras by device name.
2. Restore the previously selected camera by stable device identifier when it
   is available.
3. Display the live camera stream at 1920x1080 where supported.
4. Let the user position the complete empty board while viewing the stream.
5. Freeze the visible frame when the user selects **Take snapshot**.
6. Save the original frame and diagnostic metadata when diagnostic saving is
   enabled.
7. Detect the 7x7 internal corner pattern using strict detection followed by
   contrast-normalized fallback when necessary.
8. Reconstruct and draw the complete 9x9 raster over the frozen image.
9. Display selectable success or failure feedback, confidence, reprojection
   error, and diagnostic location.
10. Let the user select **Resume live camera** to reposition the board and try
    again.

### Phase A: Hardware initialization

1. Select the USB camera.
2. Select the HDMI projector display.
3. Set the camera resolution.
4. Open the projector window.
5. Show black on the projector.
6. Allow camera exposure to stabilize.
7. Lock camera settings when supported.

### Phase B: Board detection

1. Capture an image with the projector black.
2. Detect the outer board boundary.
3. Detect the 8x8 grid.
4. Display the result on the control monitor.
5. Let the user confirm or correct the corners.
6. Calculate the board-to-camera transformation.

### Phase C: Projector measurement

1. Project a calibration target.
2. Wait for projector and camera latency.
3. Capture several frames.
4. Detect the projected target using image differencing.
5. Repeat at multiple projector coordinates.
6. Estimate the projector-to-camera transformation.
7. Reject outliers.
8. Test the mapping against unused validation points.

### Phase D: Board overlay

1. Combine both transformations.
2. Calculate projector polygons for all 64 squares.
3. Project the full grid.
4. Measure alignment.
5. Refine the transformation if necessary.

### Phase E: Acceptance test

1. Illuminate every square sequentially.
2. Capture each result.
3. Verify that the illuminated region lies in the expected physical square.
4. Record error measurements.
5. Save the calibration if all checks pass.

## 7. User Interface

### 7.1 Current camera interface

The current interface contains:

- Named external-camera selector that activates a camera immediately when
  selected. Integrated cameras never appear in the selector.
- Persistent color-coded camera-calibration and projector-calibration
  indicators. Green means verified, red identifies the likely failed
  calibration, and gray means no calibration is active.
- Alignment errors do not discard the last verified calibration or start
  recalibration. Temporary board-follow mappings become verified only after
  optical confirmation; an error restores the previous verified mapping.
- The interactive board remembers its last selected square independently from
  current projector output. Black, previews, closure, calibration, and
  alignment errors retain the highlight. Temporary operations automatically
  restore that square; an explicit Black action keeps output black until the
  highlighted square is clicked again.
- While the projector is black, automatic board-orientation detection compares
  inset brightness over all 64 physical squares. Standard orientation requires
  the bottom-right square to be white; when the detected color parity is
  reversed, logical coordinates are rotated by 90 degrees. The result is
  remembered for starts where pieces obscure the board. When occupancy matches
  a complete starting position, aggregate piece luminance distinguishes the
  two sides and rotates the brighter White side to the bottom of the UI.
  Projection and optical
  monitoring use the same transformed coordinates, while adaptive indicator
  color follows the logical square color.
- Live camera preview.
- **Take snapshot** action.
- **Resume live camera** action.
- Frozen snapshot with detected 9x9 raster overlay.
- Default-enabled **Save diagnostic snapshots** option.
- Camera status and board-detection feedback.
- Selectable camera and detection feedback text so it can be copied.
- Detection confidence, reprojection error, and diagnostic capture location.

### 7.2 Current projector and calibration interface

The current interface contains:

- Projector/display selector showing name, manufacturer/model, resolution, and
  desktop position.
- Automatically opened black projector output on the remembered selected
  display at application startup.
- **Open projector (black)** action for reopening or moving the output after
  display changes.
- **Black** and **Close projector** actions.
- **Focus pattern** action showing a native-resolution black/white checkerboard
  with a central cross for manual optical focus adjustment.
- Remembered projector display and target settings.
- Live display-list refresh when a screen is connected or disconnected.
- Automatic reselection and calibration reactivation when the saved projector
  identity and resolution return. A display selected while that projector was
  absent shall not take precedence over the calibrated display.
- **Full projector calibration** and cancel actions.
- Calibration phase, progress, success, and failure status.
- **Project 8x8 grid** and **Project all 64 squares** actions.
- Emergency "projector black" control.
- Strict restore of the latest saved calibration when camera identifier,
  camera resolution, projector name, and projector resolution match.
- **Quick realign / assess** action using five fitting targets and four
  independent validation targets.
- **Validate calibration** action that skips Gray-code calibration and tests
  the existing mapping with nine distributed targets.
- Selectable continuous-alignment status showing board drift and active-square
  projection error.
- Camera Snapshot falls back to the current verified and camera-remapped grid
  when a fresh piece-obstructed board detection is inconclusive.
- Lichess-colored clickable 8x8 board.
- Static cyan/azure selected-square projection.

Automatic periodic verification, manual correction, alternating checkerboard
projection, previous/next stepping, and brightness/inset controls remain
future enhancements.

The projector window itself should contain no operating-system borders, cursor,
status text, or controls.

## 8. Failure Handling

Calibration shall stop and explain the problem when:

- The full board is not visible.
- An 8x8 grid cannot be established.
- Reflections or ambient light prevent target detection.
- The projector does not cover the complete board.
- Too few projected points are detected.
- Detected correspondences are inconsistent.
- Calibration error exceeds the threshold.
- Camera or projector resolution changes.
- The camera disconnects.
- The selected display disappears.

The system must not silently reuse an invalid transformation.

## 9. Environmental Considerations

The design must account for:

- Changing ambient light.
- Projector brightness and color response.
- Glossy-board reflections.
- Shadows cast by pieces or hands.
- Camera auto-exposure changes.
- Camera frame latency.
- Projector or display latency.
- Partial occlusion by chess pieces.
- Accidental movement of the board, camera, or projector.
- Dark pieces absorbing projected light.
- Light pieces reflecting and spreading projected light.

The implemented calibration currently uses an empty, alternating-color board
under stable room lighting. The target piece-aware workflow shall calibrate
with a normal populated board by fitting only distributed structured-light
correspondences belonging to the planar playing surface. Pieces, hands, board
side faces, and unreliable reflections shall be excluded.

## 10. Suggested Implementation Architecture

The implemented modules are:

```text
camera_app.py
    PySide6 control window, Qt Multimedia camera capture, and UI coordination.

board_detector.py
    Markerless empty-board detection and 9x9 camera grid reconstruction.

projector.py
    Exact-display frameless output and polygon rendering.

projector_calibration.py
    Gray-code generation/decoding, coverage measurement, mapping, mesh
    refinement, and validation calculations.

calibration_controller.py
    Non-blocking pattern/capture state machine and safety handling.

calibration_store.py
    Atomic save and strict hardware/resolution-matched restore.

chessboard_widget.py
    Interactive Lichess-colored 8x8 selection board.

diagnostics.py
    Bounded camera snapshot diagnostics.
```

OpenCV performs image processing and geometric fitting. PySide6 provides Qt
Multimedia camera capture, the control window, and projector rendering.

## 11. Milestone Deliverables

### 11.1 Milestone 1: Camera-only board detection

This milestone is complete when the application can:

1. Enumerate connected cameras by name.
2. Select a camera and remember it by stable device identifier.
3. Read and display live 1920x1080 video from the Logitech Webcam C930e.
4. Freeze the visible frame on request.
5. Detect the empty 8x8 wooden chessboard despite perspective and uneven
   illumination.
6. Reconstruct and draw all nine horizontal and nine vertical raster lines.
7. Display selectable detection status, confidence, and reprojection error.
8. Save bounded diagnostic captures for successful and failed detections.
9. Resume the live stream for another attempt.

The implementation has been validated against synthetic perspective boards
and two real Logitech camera snapshots. Both real snapshots are detected after
local contrast normalization, with measured confidence above 93%.

### 11.2 Milestone 2: Projector calibration

This milestone is implemented for the stationary empty-board setup:

1. The selected HDMI display is opened black at exact global geometry.
2. Gray-code patterns are projected and decoded.
3. A global homography and locally corrected 9x9 projector mesh are calculated.
4. All 64 projector polygons can be rendered.
5. Grid and all-square test patterns are available.
6. Nine projected square-center targets measure alignment.
7. Calibration is accepted below 10% of local square width.
8. A successful result is saved atomically and restored only for strictly
   matching camera/projector identities and resolutions.
9. A clickable Lichess board projects one physical square using adaptive
   cyan/azure colors.
10. A measured quick-realignment mode redetects and remaps the current board,
    uses five projected targets to fit a candidate camera-to-projector
    homography while retaining local mesh corrections, and accepts it only
    when four independent targets pass the same 10% error threshold.
11. A non-disruptive background monitor checks the live board grid up to five
    times per second, continuously remaps board movement, and verifies any
    currently projected square.

The first successful hardware run achieved 100% board coverage, 54.2% valid
dense decoding, 0.821 projector-pixel mapping RMS, and 8.0% maximum validation
error as a fraction of square width.

Automatic periodic verification, automatic orientation confirmation, all-64
automated validation, and operation with pieces remain future work.

## 12. Confirmed Constraints and Open Decisions

The following constraints are confirmed:

- The camera and projector may move, requiring recalibration.
- No added calibration markers are permitted.
- The current implementation uses an empty board, but the target workflow shall
  calibrate and operate with pieces present.
- The board has alternating colors. A complicated background is acceptable
  when the complete board remains detectable.
- The camera and projector are above the board.
- The camera and projector may initially be close together but may later be
  separated.
- Board orientation is initially detected from the light corner and a known
  White-side edge, then later resolved automatically from piece colors.
- Calibration uses intensity 112 Gray-code patterns and filled validation
  circles. Operational indicators use cyan on dark squares and azure on light
  squares.
- The initial camera is a Logitech Webcam C930e 1080p USB camera.
- The initial projector is an MPR 600 mini projector at 1280x720.

The following choices remain open:

- The initial mounting height and the sequence of harder physical
  configurations.
- How often to perform drift checks during a game.
- The acceptable interruption time for automatic recalibration.
- Whether operational indicator brightness, inset, and color should become
  user-configurable.
