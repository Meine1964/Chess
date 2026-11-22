# Move Detection Performance Specification

## 1. Purpose

Physical moves shall be recognized and acted on as quickly as possible without
weakening the existing protection against false acceptances. This document
defines a staged improvement. Each stage is specified, implemented, covered by
focused tests, passed through the quality gate, and checked on the real
camera/projector before the next stage starts.

## 2. Baseline Analysis

An offline benchmark on a recorded 1920x1080 camera frame with the stored
calibration produced these costs on the GUI thread:

| Work | Cost |
| --- | --- |
| Board-edge activity detector, full frame | about 107 ms per observation |
| Reference move tracker, quiet board | about 41 ms per observation |
| Reference move tracker, changed board | about 58 ms per observation |
| Live preview conversion and smooth scaling | about 10 ms per frame, about 300 ms per second at 30 fps |

The occupancy timer requests a tick every 100 ms, but one tick costs about
150-165 ms, so the effective observation period is about 200-250 ms.
Acceptance needs four tracker observations after the hand leaves: the first
quiet frame still differs from the last hand frame, two settled observations
follow, and the candidate must be seen twice. Offline play then marks the
accepted move for one second and waits a further fixed second for projector
black before observation resumes.

Lichess opponent moves arrive through the account and board event streams.
On a network that blocks streaming response headers, the worker falls back to
polling `/api/account/playing` every five seconds.

## 3. Stages

1. Measurement and optional live preview.
2. Cheaper image processing: cropped board-edge detection with geometry
   cached per camera grid, and cheaper tracker brightness normalization.
3. Shorter Offline blind period: accepted-move marking for 0.5 seconds and
   frame-confirmed projector black instead of a fixed one-second wait.
4. Background detection with frame-to-frame motion comparison, a 50 ms
   observation period, and time-based settling and candidate stability.
5. Reliable Lichess event streaming, with polling retained as a diagnosed
   fallback.

Stages 3-5 are planned and are specified in detail when they start.

## 4. Stage 1: Measurement and Optional Live Preview

### 4.1 Detection timing diagnostics

The application shall measure every occupancy tick:

- Total tick duration.
- Interval between consecutive tick starts (the effective observation period).
- Board-edge detector observation duration when the detector ran in that tick.
- Reference move tracker observation duration when the tracker ran in that
  tick.

Timing statistics cover the most recent 50 measurements per quantity and
report the mean and maximum in milliseconds.

The application shall measure the move detection latency: the time from the
last tick that still showed motion to the tick that accepted the move. A tick
shows motion when the board-edge indicator is active or the tracker reports
`activity` or `waiting-for-clear`. Latency is recorded for `accepted` and
`compound-accepted` tracker results only when motion was observed since the
previous acceptance. Camera capture latency before a frame reaches the
application is not included.

Each acceptance with a measured latency records a `move-detection-timing`
runtime event containing the latency and the timing statistics. Every runtime
state snapshot includes the current timing statistics and the most recent
latencies under `tracker.detection_timing`.

Timing measurement shall not change any detection decision.

### 4.2 Optional live camera preview

The Camera menu shall contain a checkable **Show live preview** action. The
choice persists and defaults to off.

- While the action is off, incoming camera frames are still decoded and passed
  to calibration, alignment monitoring, occupancy, and move tracking, but they
  are not converted, scaled, or drawn in the preview. The preview area shows a
  short notice explaining how to enable the preview.
- While the **Focus and point camera** dialog is open, or while the projector
  shows the pattern from **Projector > Show focus pattern**, the live preview
  is shown regardless of the setting; the focus dialog also draws the
  saved-board guide outline. When the dialog closes or the projector output
  changes, the preview returns to the notice if the setting is off.
- Snapshots, calibration clearance views, and other frozen still images are
  shown regardless of the setting. **Resume live camera** returns to the live
  preview or to the notice according to the setting.

### 4.3 Acceptance criteria

1. With the preview off, a full Offline game is detected as before.
2. Runtime events include `move-detection-timing` after accepted moves, and
   the runtime state reports tick and detector timings.
3. The focus dialog shows the live preview with the preview setting off.
4. The preview setting persists across restarts.

### 4.4 Implementation status

Implemented. `src/chess_camera/detection_timing.py` provides the rolling
statistics and latency measurement. The board-edge detector and the reference
tracker expose their observation count, last observation duration, and, for
the tracker, last state; `CameraWindow` measures each occupancy tick around
the unchanged occupancy logic. Focused tests cover the statistics, the latency
rules, the detector timing properties, and the preview gating.

Hardware baseline with the preview off (50 ticks, three Offline moves):
occupancy tick mean 99 ms (max 391 ms), tick period mean 190 ms (max
409 ms), board-edge detector mean 66 ms (max 209 ms), reference tracker mean
60 ms (max 228 ms), and move latency after the last observed motion of 0.38,
0.92, and 1.34 seconds.

## 5. Stage 2: Cheaper Image Processing

Stage 2 reduces processing cost without changing detection decisions.

### 5.1 Cropped board-edge detection

- The board-edge detector processes only a crop of the camera frame: the
  bounding rectangle of the guidance region (board plus one-square halo)
  expanded by 24 pixels and clipped to the frame.
- The crop, the guidance, board, outside, and outer-halo masks, the
  illumination smoothing kernel, the median square area, and the normalized
  smoothing weights are computed once and reused while the camera grid and
  frame size are unchanged.
- A changed camera grid rebuilds the cached geometry. The crop itself is kept
  while the new guidance bounding rectangle, expanded by 4 pixels, still fits
  inside it, so small alignment remaps keep the detector baseline. When the
  guidance region no longer fits, the detector builds a new crop and resets
  its baseline and active state, as when it is first armed.
- Reported component bounds remain in full-frame camera coordinates.
- Region-weighted illumination smoothing treats pixels outside the crop as
  outside the region (zero weight). The previous implementation reflected the
  frame at its border; this only matters where the guidance region touches the
  frame edge, and decisions were verified to be unchanged there.

### 5.2 Exact cheaper statistics

- The median brightness difference between two 8-bit images is computed from
  a histogram of the differences and equals the floating-point median,
  including the half-value result for an even pixel count.
- Applying a brightness offset to an 8-bit image uses a lookup table that
  reproduces the previous floating-point clip and truncation exactly.
- The board-edge detector and the reference tracker's candidate brightness
  normalization use these helpers.

### 5.3 Unchanged behavior

Registration of the canonical board keeps full-resolution phase correlation,
because half-resolution correlation produced shift errors up to 0.6 pixels.
Thresholds, morphology, component rules, and tracker decisions are unchanged.

### 5.4 Acceptance criteria

1. On recorded camera frames with simulated hand crossings, brightness
   changes, and noise, the edge detector reports the same active and crossing
   decisions and component statistics as before.
2. Candidate normalization output is pixel-identical to the previous
   implementation.
3. The board-edge indicator and Offline move acceptance behave as before on
   hardware, with lower `tracker.detection_timing` values.

### 5.5 Implementation status

Implemented in `src/chess_camera/image_statistics.py`,
`src/chess_camera/guidance_activity.py`, and
`src/chess_camera/reference_move_tracker.py`. Focused tests cover the exact
statistics helpers and the crop-invariance of edge detection.

A one-off comparison against the Stage 1 implementation on the recorded
1920x1080 camera frame (40 frames with simulated crossings, gradients,
brightness steps, and noise, and the same frames shifted so the guidance
region touches the top of the frame) produced identical decisions and
component statistics and pixel-identical normalization. Measured on the
development machine, the board-edge detector mean dropped from 133 ms to
32 ms and candidate normalization from 23 ms to 10 ms. Hardware timing is
pending.
