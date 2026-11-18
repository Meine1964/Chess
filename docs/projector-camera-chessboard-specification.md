# Projector/Camera Chessboard Calibration Specification

**Version:** 0.4  
**Status:** Draft  
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

The first milestone stops after reliable markerless board detection from a
camera snapshot and display of an aligned raster over the captured image. The
projector is deliberately excluded from this milestone. Later milestones add
projector calibration, operation with pieces present, individual-square move
indicators, and chess rules.

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
- The camera and projector may move between uses or during operation. Movement
  invalidates the current geometric calibration and requires recalibration.
- The camera and projector may initially be placed close to each other, but
  the design must not depend on them sharing the same position or viewing
  angle.
- The complete outer boundary of the board is visible to the camera.
- The projector image covers the entire board with some margin.
- The physical board has alternating square colors and is placed on a plain
  table.
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
- Allow selection of the USB camera.
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
- Open a borderless full-screen window on that display.
- Use the MPR 600 projector's 1280x720 output resolution.
- Prevent operating-system scaling from changing projection coordinates.
- Keep user-interface controls on the primary monitor.
- Support a completely black projector frame.

Black is the default background because it minimizes unwanted projected light.

### FR-3: Board detection

For the first milestone, calibration shall use an empty, alternating-color
board on a plain table. Later milestones shall detect and track the board when
some grid intersections, square edges, and colors are occluded by pieces.

The application shall detect:

- The outer quadrilateral of the chessboard.
- The nine horizontal grid lines.
- The nine vertical grid lines.
- Their 81 intersections, where sufficiently visible.
- The resulting 64 square polygons.
- The board's chess orientation from its corner colors and the known White-side
  reference edge.
- A confidence score for the detected board.

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

This fallback is required because the physical wooden board has visible wood
grain, reflections, and non-uniform illumination that can defeat a detector
intended for high-contrast printed calibration boards.

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

Support for pieces is deferred until the empty-board detector is reliable.
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

### FR-4: Projector-to-camera calibration

The system shall display a sequence of structured-light patterns:

1. Capture an image with the projector black.
2. Project one or more bright calibration targets.
3. Capture several images while each target is visible.
4. Compare illuminated images with the black reference image.
5. Locate each projected target in camera coordinates.
6. Use the known projector coordinates and measured camera coordinates to
   calculate a projector-to-camera homography.
7. Validate the result using additional target positions not used to fit the
   transformation.

#### Calibration targets

Targets should initially be:

- Automatically selected for high contrast according to camera response,
  ambient light, and the colors of the physical board.
- Large enough to be detected reliably.
- Displayed against a black background.
- Positioned away from the projector image boundaries.
- Sampled at multiple locations across the full board region.

Candidate targets may include a dot, filled circle, filled square, outlined
shape, cross, or coded pattern. Candidate colors and shapes shall be evaluated
experimentally. Red is not required. The selected target should maximize
detection reliability and localization accuracy rather than resemble a future
move indicator.

At least four non-collinear point correspondences are mathematically required.
The implementation should use substantially more points, such as 9, 16, or
25, and estimate the mapping robustly.

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

### FR-6: Projection test and later square indicators

The first calibration milestone shall provide full-board grid, fill, and
alternating test modes. Individual-square indication belongs to the following
milestone after geometric calibration is reliable.

#### Individual-square mode

- Illuminate or mark one physical square using a high-contrast color and shape.
- Step automatically through all 64 squares.
- Support manual previous and next controls.
- Show the current internal square index.
- Show chess notation after orientation is detected.
- Allow experimental indicators such as a square fill, outline, dot, circle,
  cross, or other localized symbol.

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

Color and geometry are configurable throughout these tests. Calibration shall
automatically recommend a detectable color, while allowing manual selection
for experiments.

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

### 7.1 Current camera-only interface

The current interface contains:

- Named camera selector and **Use selected camera** action.
- Live camera preview.
- **Take snapshot** action.
- **Resume live camera** action.
- Frozen snapshot with detected 9x9 raster overlay.
- Default-enabled **Save diagnostic snapshots** option.
- Camera status and board-detection feedback.
- Selectable camera and detection feedback text so it can be copied.
- Detection confidence, reprojection error, and diagnostic capture location.

### 7.2 Future projector-calibration interface

The projector-calibration interface should add:

- Projector or display selector.
- Start and cancel calibration actions.
- Calibration status and current step.
- Manual four-corner correction.
- Projection test selector.
- Previous and next square controls.
- Brightness and inset controls.
- Save and load calibration controls.
- Emergency "projector black" control.

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

Initial calibration uses an empty, alternating-color board on a plain table
under stable room lighting. Later milestones must tolerate pieces and partial
occlusion.

## 10. Suggested Implementation Architecture

```text
CameraManager
    Discovers and captures USB camera frames.

ProjectorWindow
    Controls the borderless HDMI display output.

BoardDetector
    Finds the board boundary, grid, and orientation.

ProjectionTargetDetector
    Finds projected calibration patterns in camera images.

CalibrationEngine
    Estimates, validates, refines, saves, and loads transformations.

BoardGeometry
    Converts between board cells, camera points, and projector polygons.

ProjectionRenderer
    Renders square fills, grids, checkerboards, and later move indicators.

CalibrationController
    Coordinates the multi-step calibration state machine.

ControlUI
    Presents preview, diagnostics, corrections, and test controls.
```

OpenCV is a likely foundation for camera capture, image processing, homography
estimation, and robust fitting. A GUI framework such as PySide6 can manage the
control window and full-screen projector window.

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

The next milestone will:

1. Open a full-screen black window on the HDMI projector.
2. Project and detect calibration targets.
3. Calculate the board-to-projector transformation.
4. Render all 64 projected square polygons.
5. Project a high-contrast full-board test pattern aligned to all 64 squares.
6. Display measured alignment error.
7. Detect orientation from the light corner and the configured White-side
   reference edge.
8. Save, reload, and revalidate a calibration.

The following milestone will add experimental individual-square indicators
and validate dot, circle, outline, fill, and other candidate shapes and colors.

## 12. Confirmed Constraints and Open Decisions

The following constraints are confirmed:

- The camera and projector may move, requiring recalibration.
- No added calibration markers are permitted.
- Initial calibration uses an empty board.
- Later operation must work with pieces on the board.
- The board has alternating colors and sits on a plain table.
- The camera and projector are above the board.
- The camera and projector may initially be close together but may later be
  separated.
- Board orientation is initially detected from the light corner and a known
  White-side edge, then later resolved automatically from piece colors.
- Calibration color and target shape are chosen for detection reliability;
  red and square targets are not required.
- The initial camera is a Logitech Webcam C930e 1080p USB camera.
- The initial projector is an MPR 600 mini projector at 1280x720.

The following choices remain open:

- The initial mounting height and the sequence of harder physical
  configurations.
- The best calibration target color, geometry, size, and display duration.
- How often to perform drift checks during a game.
- The acceptable interruption time for automatic recalibration.
