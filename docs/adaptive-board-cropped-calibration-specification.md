# Adaptive Board-Cropped Calibration Specification

## Purpose

Provide a third projector calibration method that minimizes light outside the
physical chessboard while preserving the existing empty-board Gray-code and
populated complementary-coded calibration methods unchanged.

## Capture Sequence

1. Capture a stable projector-black populated-board reference.
2. Detect the physical board in camera coordinates.
3. Briefly project one full-field uniquely coded discovery pattern.
4. After a 100 ms minimum display interval, inspect fresh camera frames until
   one contains enough on-board coded markers for a valid rough homography,
   then stop broad discovery projection immediately. Do not accept the first
   frame merely because it arrived.
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

- The full-field discovery pattern shall use a black background and remain
  visible only until a fresh frame passes coded-marker and rough-homography
  validation after the 100 ms minimum interval.
- Projector-black reference capture may continue to average the existing
  number of frames because it does not disturb nearby players.
- Cropped Gray-code captures shall wait 250 ms and average the existing
  configured frame count. The short single-frame rule applies only to the
  evidence-validated discovery image; using it for successive Gray-code bits
  can shift camera frames between patterns and invalidate the complete code.
- Cancellation, timeout, camera-resolution change, insufficient markers,
  invalid rough geometry, Gray-code decode failure, or validation failure
  shall immediately return the projector to black.
- Failure shall preserve the previous verified calibration and shall not
  automatically invoke either existing calibration method.

## Isolation

- **Full calibration - empty board** remains unchanged.
- **Populated calibration - experimental** remains unchanged.
- The new action is **Board-cropped calibration - experimental**.
- Saved calibration data uses the existing `CalibrationResult` and
  `CalibrationStore` format.

## Acceptance Criteria

1. The discovery phase accepts exactly one validated full-field coded frame
   after the black reference.
2. Every Gray-code frame is black outside the expanded projector-board mask.
3. The final mapping comes from dense Gray-code decoding, not the rough coded
   homography.
4. Existing calibration methods retain their current steps and timing.
5. The third action participates in the existing active/cancel UI lifecycle.
6. Focused controller and pattern tests cover masking, capture timing, final
   dense fitting, and isolation of existing modes.

## Implementation Status

Implemented as the independent **Board-cropped calibration - experimental**
action. The existing empty-board and populated methods retain their original
capture sequences.

Validation:

- Focused calibration and menu tests: 89 passed.
- Full repository quality gate: 421 passed.
