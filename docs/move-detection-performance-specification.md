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
2. Cheaper image processing: cropped, reduced-resolution board-edge detection
   with masks cached per camera grid, and cheaper tracker registration.
3. Shorter Offline blind period: accepted-move marking for 0.5 seconds and
   frame-confirmed projector black instead of a fixed one-second wait.
4. Background detection with frame-to-frame motion comparison, a 50 ms
   observation period, and time-based settling and candidate stability.
5. Reliable Lichess event streaming, with polling retained as a diagnosed
   fallback.

Stages 2-5 are planned and are specified in detail when they start.

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
