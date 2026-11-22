# Adaptive Board-Cropped Calibration Specification

## Purpose

Provide a third projector calibration method that minimizes light outside the
physical chessboard while preserving the existing empty-board Gray-code and
populated complementary-coded calibration methods unchanged.

## Capture Sequence

1. Capture a stable projector-black populated-board reference.
2. Detect the physical board once while the projector remains black.
3. Briefly project one full-field uniquely coded discovery pattern.
4. Allow at least 250 ms for the camera pipeline to begin reaching the coded
   pattern, reject frames that still match the black reference, and require
   three mutually stable changed frames. Return the projector to black before
   performing coded-marker detection and rough-fit processing.
5. Fit a permissive rough camera-to-projector homography from at least four
   coded markers observed on the board plane. This rough fit is used only to
   localize later projection and is not accepted as the final calibration.
6. Transform the board into projector coordinates and create a convex mask
   expanded by half the median projected square size.
7. Project every normal and inverse Gray-code image only inside that expanded
   mask; all pixels outside it remain black.
8. Decode the cropped Gray-code captures inside the camera-space board mask,
   fit the final dense projector mapping, and validate it with surplus
   targets on usable squares (see "Validation with pieces").

## Exposure and Failure Handling

- The full-field discovery pattern shall use a black background, settle for at
  least 250 ms, and remain visible until three changed, mutually stable camera
  frames have been averaged. Frames that still match the black reference and
  unstable transition frames shall not count toward the average. The existing
  capture watchdog bounds the visible interval.
- Marker detection and rough-homography fitting shall occur while the
  projector is black. A failed rough fit shall report the underlying
  calibration error and end the attempt rather than repeatedly flashing the
  discovery pattern.
- Projector-black reference capture may continue to average the existing
  number of frames because it does not disturb nearby players.
- Cropped Gray-code capture shall be transition-synchronized rather than use a
  fixed settling delay. After displaying each pattern, reject camera frames
  that still match the previously accepted pattern. Accept the new pattern
  only after two consecutive frames are mutually stable, then average those
  two frames. The required change threshold shall scale down for patterns
  whose projector-space difference inside the board mask is small.
- If two successive cropped projector patterns are identical inside the board
  mask, reuse the previous camera capture without projecting or waiting for a
  redundant image.
- If transition synchronization has not accepted a pattern after 600 ms,
  fall back to three-frame averaging for that pattern. The delay must exceed
  the projector and camera pipeline latency, so a forced capture never shows
  the previous pattern. Field evidence: with a 250 ms fallback, a startup run
  decoded only 14,000 of about 140,000 board pixels, and empty squares failed
  as well as occupied ones.

### Gray-code pair verification

After each inverse capture, check the normal/inverse pair before using it:

- **Contrast:** the fraction of camera board pixels whose normal and inverse
  images differ by at least eight grey levels.
- **Pattern agreement:**
  - Predict each board pixel's projector position with the rough discovery
    homography.
  - Sample the projected normal pattern there, but only where the pattern is
    uniform for 8 projector pixels in every direction. This tolerates rough
    error and piece parallax.
  - Agreement is the fraction of those pixels whose normal and inverse
    images differ by at least eight grey levels with the brighter image
    matching the prediction. Pixels without that difference count against
    agreement: a stale normal frame (still showing the previous projector
    state) differs from the inverse on only about half of them, while every
    pixel that does differ still has the correct sign.
  - The check applies only when at least 5% of the board pixels are usable,
    which covers the coarse bits.
- **Accept** the pair when contrast is at least 30% and, where checked,
  agreement is at least 70%. A stale or mis-synchronized frame gives roughly
  50% agreement or a collapsed contrast. The 70% threshold leaves room for
  dark or occupied squares, whose contrast and parallax lower agreement.
- **Recapture** a failing pair (normal and inverse) at most twice. This
  re-takes a single image pair and does not repeat the calibration.
- **Fail** if the pair still fails: end the calibration with an error that
  names the axis, the bit, and the measured contrast and agreement.

### Capture diagnostics

- For every Gray-code capture, record the step key, the synchronization mode
  (`transition`, `forced`, or `reused`), the changed-pixel ratio, and the
  elapsed time since the pattern was shown.
- For every pair, record the contrast, the agreement, and the number of
  recaptures.
- Store these in the `gray_code_captures` and `gray_code_pairs` details of
  the calibration event.
- On failure, also save the weakest pair as
  `runtime-calibration-gray-weakest-normal.png` and
  `runtime-calibration-gray-weakest-inverse.png`.
- Cancellation, timeout, camera-resolution change, insufficient markers,
  invalid rough geometry, Gray-code decode failure, or validation failure
  shall immediately return the projector to black.
- Failure shall preserve the previous verified calibration and shall not
  automatically invoke either existing calibration method.
- Game clocks, game results, and the board-edge occlusion/activity indicator
  shall remain suppressed throughout discovery, Gray-code capture, and
  validation.
- After successful validation, return to black immediately, without a
  calibrated-grid preview, and restore any remembered square.

## Calibration With Pieces On The Board

Board-cropped calibration shall succeed with pieces in any legal or partial
position, so players never have to clear the board. Pieces are off the board
plane. Their decoded projector coordinates are valid Gray codes but do not
belong to the camera-to-board-plane mapping, and they also cast projector
shadows and occlude neighbouring squares. The calibration therefore searches
for informative board-plane places instead of using every decoded pixel.

Field evidence: with 22 pieces on the board, discovery succeeded, but the
unfiltered dense fit was rejected with "33% inliers, 1.30 px RMS error".

### Empty-square candidates

- After the black reference and board detection, classify every square from
  the colour black-reference frame with the camera-only occupancy model.
- Because the board orientation is unknown, build the empty-square model for
  both rank and file parities. A square is an empty candidate only when both
  models label it empty.
- Occupancy is only a prior. Plane consistency, described below, is the
  deciding test. If occupancy fails or the retained candidates do not satisfy
  the spread rules, repeat the selection using every square.

### Board lattice with pieces

Discovery first refines the rough camera board grid from the black reference.
The existing method needs an unbroken 7×5 block of interior corners, which
only a nearly empty middle board provides. Field evidence: with about 35
pieces spread over the board, discovery failed with "The visible 7x5
middle-board intersections were not detected", before plane support or any
board map existed, so the player received only a generic instruction.

When the 7×5 block is not found, or the block found disagrees with the rough
board location (pieces can form a false lattice), and the rough detection is
not a verified internal-corner detection, the lattice is fitted from
individually visible corners:

- **Candidates:** each of the 49 interior corners is refined with sub-pixel
  corner search, starting from its rough-grid position. A candidate moving
  more than a quarter of a square is discarded.
- **Corner test:** four samples, offset from the candidate by 0.2 of a square
  along both grid axes towards each adjacent square, must form a
  checkerboard corner: both diagonally opposite samples are brighter than
  both of the other pair by at least 16 grey levels. A piece covering a
  sample breaks this pattern, so the corner is discarded.
- **Parity:** the brighter diagonal alternates from corner to corner. Corners
  whose parity disagrees with the majority are discarded.
- **Fit:** a RANSAC homography from board coordinates to camera pixels, with
  a threshold of 5% of a square, selects the inliers. The fit needs at least
  12 inliers, an inlier strictly inside each 4×4 quadrant, and at most
  2 camera pixels RMS error after a least-squares refit. The 9×9 grid is
  then extrapolated from it.

### Clearing instruction when the board is hidden

If the lattice still cannot be fitted, no projector mapping exists yet, so
nothing can be projected. Instead:

- The calibration fails with "Too many pieces cover the board squares."
- The camera preview freezes on the black reference, with four middle rows
  of squares outlined in red. These are grid rows 2–5, or grid columns 2–5
  when more verified corners lie there. Clearing them leaves at least 21
  fully surrounded corners spread over all quadrants.
- The corrective action is "Clear the squares outlined in red in the camera
  view, then start Board-cropped calibration again."
- The next calibration start resumes the live preview.

### Phase 1: discovery markers

- Discard coded markers whose camera-space centre lies in a square that is not
  an empty candidate. When fewer than four markers remain, use every
  on-board marker.
- RANSAC on the remaining markers rejects off-plane markers. The rough
  mapping only positions the cropped Gray-code mask, which has a half-square
  margin, so moderate rough error is harmless.

### Phase 2: Gray-code pixel selection

Apply these filters in order, and record how many pixels survive each one:

1. **Contrast:** every normal/inverse bit pair differs by at least eight
   grey levels, and the decoded coordinate lies inside the projector.
2. **Smoothness:** the decoded projector x and y lie within 4 projector
   pixels of their 5×5 camera-neighbourhood median. This removes bit errors,
   piece silhouettes, and shadow edges.
3. **Square interior:** the pixel lies inside the central 80% of a
   candidate square. The square polygon comes from the detected camera grid,
   scaled 0.8 about its centre. This excludes grid lines and the bleed of
   neighbouring pieces.
4. **Per-square plane consistency:**
   - Fit a RANSAC homography, with a 2.5 projector-pixel threshold, to the
     surviving pixels.
   - Keep a square only when it has at least 40 surviving pixels and at least
     80% of them are plane inliers.
   - A square containing a piece fails this test, even if occupancy missed
     it.
5. **Spread:**
   - At least 12 squares are retained.
   - Every 4×4 board quadrant contains a retained square.
   - Retained squares span at least three ranks and three files.

The surviving pixels of retained squares feed the existing dense fit. Its
70% inlier, 2.0 px RMS, and distributed-support checks then apply to the
retained squares, not to the whole board.

Local corner corrections use only plane-consistent pixels from retained
squares. A grid point with fewer than 20 nearby samples, for example the
corner of an occupied square, keeps the global board-plane homography value.
### Validation with pieces

The fixed validation targets of the other methods avoid only the starting
position. In a mid-game position they can land on, or be hidden behind, a
piece. Field evidence: a startup run decoded every Gray-code pair at over 90%
contrast and agreement and passed plane support. Its centre target at
(4.00, 4.00), next to an occupied square, was then measured 37% of a square
off, and the run failed. Board-cropped validation therefore projects more
targets than it needs and tolerates misses:

- **Target placement:** 13 nominal squares, namely the ring of rows and
  columns {1, 3, 4, 6} without its inner four squares, plus the centre
  square (3, 3). Each nominal square is replaced by the nearest unused
  retained square within two squares, preferring squares with fewer
  unretained neighbours. A score of distance plus 0.5 per unretained
  neighbour decides. The target is the centre of that square, so the dot
  lies on the empty board plane. A nominal square with no candidate is
  skipped.
- **Misses:** a target that is not detected, or whose error is at least 10%
  of a square, counts as a miss: a piece hid it or shifted it. No alternative
  location is tried and no refinement is run from the measurements, so a
  piece-displaced dot can never distort the grid.
- **Acceptance:** at least 9 targets must hit, and every 4×4 board quadrant
  must contain a hit. The stored validation result contains only the hits.
  The diagnostics record every target with its square, error, and outcome.
- **Failure:** if too few targets hit, calibration fails with a message
  stating how many hit and how many were required.

The empty-board and populated methods and quick verification keep their
fixed targets, alternatives, and refinement.

### Board map on failure

A failed board-cropped calibration projects a board map whenever plane
support was computed, so the player can see what the calibration could use:

- **Green:** every retained square, meaning usable for calibration.
- **Red:** squares the player must clear. When too few plane-consistent
  squares remain, these are the unretained squares of the central 4×4 block
  (grid rows and columns 2–5). With all 16 retained, every spread rule is
  met. Otherwise no square is red.
- Each square is drawn at 80% of its size, so the board lines stay visible
  between squares. The final mapping is used when available, otherwise the
  rough discovery mapping.
- The map stays on until the next calibration starts or another projector
  output replaces it. It is never shown after a successful calibration.
- The calibration is not repeated automatically.

### Failure instructions

Every board-cropped failure ends the status text (or the dialog of a manual
calibration) with a corrective action:

- With red squares: "Green squares were usable. Remove the pieces from the
  red squares, then start Board-cropped calibration again."
- With a map but no red squares: "Green squares were usable for calibration.
  Keep hands and shadows off the board, then start Board-cropped calibration
  again."
- When pieces hide the board lattice: "Clear the squares outlined in red in
  the camera view, then start Board-cropped calibration again."
- Otherwise, without a map: "Keep hands and shadows off the board, then start
  Board-cropped calibration again."

### Diagnostics

- Every board-cropped calibration records the per-filter pixel counts, the
  retained, plane-rejected, sparse, and occupancy-excluded squares, and
  whether the occupancy prior was used. These are stored in the
  `calibration-completed` or `calibration-failed` runtime event.
- It also saves `runtime-calibration-plane-support.png`, which draws on the
  black reference:
  - accepted pixels in green;
  - smooth pixels rejected by the plane test in red;
  - retained squares outlined in green;
  - plane-rejected squares outlined in red;
  - occupancy-excluded squares outlined in blue.

## Startup Calibration

- Whenever the camera starts or the projector selection changes, and a
  projector display is selected, the application runs this calibration
  automatically. The saved calibration is not trusted first.
- **Focus and pointing step.** Before calibrating, the projector shows the
  focus pattern so the player can focus the camera and the projector and
  point the camera at the board:
  - When a saved calibration exists for the selected projector display and
    resolution, the pattern is limited to the saved board area, enlarged by
    one square on every side. Everything else stays black. The saved
    calibration only places this pattern; it is never used as the mapping.
  - Without such a saved calibration, the full-screen focus pattern is shown.
  - When the saved calibration also matches the camera and its resolution,
    the live camera preview outlines the saved board position in green, so
    the player can aim the camera until the board fills the outline.
  - A non-modal pop-up, "Focus and point the camera", explains this and
    offers **OK** and **Cancel**. The live preview keeps updating behind it.
  - **OK** turns the projector black, removes the outline, and schedules the
    calibration. **Cancel** does the same but skips the calibration, so the
    application stays uncalibrated until a calibration is requested.
  - Restarting the camera or changing the projector closes an open pop-up
    and begins a new focus step.
  - Manually requested calibrations skip this step.
- The calibration starts after the camera has delivered frames for two
  seconds after OK, so exposure can settle.
- On success, the result is activated and saved like a requested
  calibration.
- On failure or cancellation, the application remains uncalibrated. A saved
  calibration is never loaded, because the board, camera, or projector may
  have moved since it was made.
- The status bar reports the failure and the corrective action, and the
  board map is projected as described in "Board map on failure". No modal dialog
  interrupts startup, and the calibration is not repeated automatically.
- Explicit calibration requests use the same piece-tolerant procedure.

## Isolation

- **Full calibration - empty board** remains unchanged.
- **Populated calibration - experimental** remains unchanged.
- The new action is **Board-cropped calibration - experimental**.
- Saved calibration data uses the existing `CalibrationResult` and
  `CalibrationStore` format.

## Acceptance Criteria

1. The discovery phase detects the board while black, waits at least 250 ms,
   rejects stale and unstable camera frames, averages three stable coded
   frames, and returns to black before fitting the rough mapping.
2. Every Gray-code frame is black outside the expanded projector-board mask.
3. The final mapping comes from dense Gray-code decoding, not the rough coded
   homography.
4. Existing calibration methods retain their current steps and timing.
5. The third action participates in the existing active/cancel UI lifecycle.
6. Focused controller and pattern tests cover masking, capture timing, final
   dense fitting, and isolation of existing modes.
7. Calibration overlays remain absent throughout capture, and success returns
   the projector to black without a grid preview.
8. With synthetic off-plane piece regions covering the starting position, the
   unfiltered dense fit is rejected, and the piece-tolerant selection produces
   a grid within one projector pixel of the true grid.
9. Too few plane-consistent squares produce a calibration error that names
   the retained-square count.
10. Startup runs the board-cropped calibration automatically. On failure it
    stays uncalibrated without a modal dialog and never loads the saved
    calibration.
11. A Gray-code pair whose predicted-pattern agreement or contrast is too low
    is recaptured at most twice, and then fails with a message naming the
    bit.
12. A failure after plane support projects retained squares in green, and
    after a plane-support failure also the unretained squares of the central
    4×4 block in red. Every failure states a corrective action.
13. Validation targets lie on retained square centres. Calibration succeeds
    when at least 9 targets hit and every quadrant has a hit, even if a
    piece hides or displaces the others. Misses never feed refinement.
14. With pieces breaking every 7×5 block, discovery fits the board lattice
    from individually visible corners to within one camera pixel. When too
    few corners are visible, the failure outlines the squares to clear in the
    camera view and names that action.
15. At startup, the focus pattern is shown cropped to the saved board area
    plus one square, or full-screen without a saved calibration. The camera
    preview outlines the saved board when the camera matches. Calibration
    starts only after OK; Cancel leaves the application uncalibrated.

## Implementation Status

Implemented as the independent **Board-cropped calibration - experimental**
action. The existing empty-board and populated methods retain their original
capture sequences. Hardware testing confirmed successful dense calibration
and the faster transition-synchronized cropped Gray-code cycle. The
transition-synchronized, watchdog-bounded three-frame discovery policy and
five-second completion-preview lifecycle are automated-test validated and
await a final physical retest.

Piece-tolerant plane-support selection, occupancy-prior marker filtering,
the plane-support diagnostics, and automatic startup calibration are
implemented and validated by automated tests. Startup never loads the saved
calibration: on failure the app stays uncalibrated and, for insufficient
plane support, lights the central squares to clear. The synthetic
starting-position test confirms that the unfiltered fit is rejected while
the plane-support fit stays within one projector pixel.

The 600 ms forced-capture delay, Gray-code pair verification with at most
two recaptures, and the per-capture and per-pair diagnostics are implemented
and validated by automated tests (matching, stale, and collapsed pairs;
recapture then failure). A hardware startup run with pieces confirmed that
every pair passed with over 90% contrast and agreement and that plane support
succeeded. The fixed starting-position validation targets then failed on a
piece, which led to the surplus-target validation.

Surplus-target validation on retained squares, the green/red board map on
failure, and the corrective action for every failure are implemented and
validated by automated tests: target selection avoids unretained squares,
validation succeeds with 10 of 13 hits without refinement and fails with 8,
the map colours, and RGB projector output. They await a hardware retest. The
contrast and agreement thresholds, the forced-capture delay, and the
smoothness and per-square inlier thresholds also await further hardware
validation with pieces on the board.

The piece-tolerant board lattice and the camera-view clearing instruction are
implemented and validated by automated tests: the starting position and a
scattered position with an unverified rough grid fit within one camera pixel,
fully covering pieces raise the clearing error with red outlines on the middle
band, and a startup failure freezes the camera view with the clearing action.
On the field image of the failed 35-piece startup, the fit verified 41 of the
49 interior corners at 0.65 camera-pixel RMS. A hardware rerun of the
complete calibration with that position confirmed it.

The startup focus and pointing step is implemented and validated by
automated tests: the pattern is cropped to the enlarged saved board area or
shown full-screen without a saved calibration, the camera guide outline is
set only when the saved calibration matches the camera, OK schedules the
calibration, Cancel leaves the app uncalibrated, and a stale dialog is
ignored. The dialog interaction and the preview outline await hardware
confirmation.

Validation:

- Projector calibration and projector tests: 120 passed.
- Full repository quality gate: 495 passed.
