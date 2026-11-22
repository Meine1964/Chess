# Adaptive Board-Cropped Calibration Specification

## Purpose

Provide a third projector calibration method that minimizes light outside the
physical chessboard while preserving the existing empty-board Gray-code and
populated complementary-coded calibration methods unchanged.

## Capture Sequence

1. Capture a stable projector-black populated-board reference.
2. Detect the physical board once while the projector remains black.
3. Briefly project one full-field uniquely coded discovery pattern.
4. Allow 250 ms for the camera pipeline to reach the coded pattern, then
   average three fresh frames. Return the projector to black before performing
   coded-marker detection and rough-fit processing.
5. Fit a permissive rough camera-to-projector homography from at least four
   coded markers observed on the board plane. This rough fit is used only to
   localize later projection and is not accepted as the final calibration.
6. Transform the board into projector coordinates and create a convex mask
   expanded by half the median projected square size.
7. Project every normal and inverse Gray-code image only inside that expanded
   mask; all pixels outside it remain black.
8. Decode the cropped Gray-code captures inside the camera-space board mask,
   fit the final dense projector mapping, and run the existing localized
   validation and refinement workflow.

## Exposure and Failure Handling

- The full-field discovery pattern shall use a black background, settle for
  250 ms, and remain visible only while three fresh frames are averaged.
  Marker detection and rough-homography fitting shall occur while the
  projector is black. This avoids both a long continuous exposure while the
  roughly 0.8-second fit runs and an unreliable response-only capture that can
  repeatedly flash partially updated candidates.
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
- If transition synchronization has not accepted a pattern after 250 ms,
  fall back to the proven three-frame averaging path for that pattern.
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

## Isolation

- **Full calibration - empty board** remains unchanged.
- **Populated calibration - experimental** remains unchanged.
- The new action is **Board-cropped calibration - experimental**.
- Saved calibration data uses the existing `CalibrationResult` and
  `CalibrationStore` format.

## Acceptance Criteria

1. The discovery phase detects the board while black, averages three coded
   frames after a 250 ms settling interval, and returns to black before fitting
   the rough mapping.
2. Every Gray-code frame is black outside the expanded projector-board mask.
3. The final mapping comes from dense Gray-code decoding, not the rough coded
   homography.
4. Existing calibration methods retain their current steps and timing.
5. The third action participates in the existing active/cancel UI lifecycle.
6. Focused controller and pattern tests cover masking, capture timing, final
   dense fitting, and isolation of existing modes.
7. Calibration overlays remain absent throughout capture, and success returns
   the projector to black without a grid preview.

## Implementation Status

Implemented as the independent **Board-cropped calibration - experimental**
action. The existing empty-board and populated methods retain their original
capture sequences. Hardware testing confirmed successful dense calibration
and the faster transition-synchronized cropped Gray-code cycle. The final
250 ms/three-frame discovery policy and five-second completion-preview
lifecycle are automated-test validated and await a final physical retest.

Validation:

- Focused projector, calibration, and menu tests: 120 passed.
- Full repository quality gate: 426 passed.
