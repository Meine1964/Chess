# Single-Shot Coded-Pattern Chessboard Calibration Specification

**Version:** 0.2
**Status:** Experimental implementation; existing Gray-code calibration preserved
**Platform:** Python on Windows
**Camera:** Logitech Webcam C930e at 1920x1080
**Projector:** MPR 600 over HDMI at 1280x720

**Related specifications:**

- [Projector/camera chessboard system](./projector-camera-chessboard-specification.md)
- [Existing Gray-code projector calibration](./projector-calibration-specification.md)
- [Piece-aware board localization](./piece-aware-board-localization-specification.md)

## 1. Purpose

Define a less intrusive camera-to-projector calibration method based on one
uniquely coded projected pattern.

The method shall:

1. Require only a rough camera estimate of the chessboard region.
2. Measure projector coordinates directly on visible parts of the playing
   plane.
3. Use visible physical square boundaries as observations rather than treating
   an estimated outer quadrilateral as the final 9x9 raster.
4. Fit the complete equal-square chess lattice directly in projector space.
5. Extrapolate through piece-covered areas from distributed measured evidence.
6. Reject projected evidence measured on pieces or other non-planar objects.
7. Complete its visible capture in a fraction of a second where hardware
   timing permits.

The design is an alternative to, not a replacement or migration of, the
existing Gray-code specification. The current implementation and saved
calibration format shall remain available until this method passes automated
and physical-hardware acceptance tests.

## 2. Motivation

The existing workflow first reconstructs a complete camera-space 9x9 board
lattice and then maps that lattice into projector coordinates. A systematic
camera-lattice error is therefore inherited by the projector mapping.

This is particularly sensitive when:

- Pieces hide the outer ranks and files.
- A decorative border resembles a square-width extension of the board.
- An object partially hides the playing-surface boundary.
- Perspective and board thickness make the outer silhouette misleading.
- Piece shadows make individual square centers unreliable.

The projector focus pattern is highly visible on the physical board. This
demonstrates that a single projected pattern can provide strong optical
evidence. A uniquely coded pattern can additionally identify the projector
coordinate represented by each detected feature.

## 3. Core Principle

The proposed workflow reverses the current dependency:

1. Locate only a rough camera-space board region.
2. Observe visible physical chess-grid boundaries in a projector-black frame.
3. Project one uniquely coded calibration pattern.
4. Decode projector coordinates at visible board-plane locations.
5. Associate visible physical chess boundaries with measured projector
   coordinates.
6. Fit the complete equal-square 9x9 chess lattice directly in projector
   coordinates.

The outer camera-space quadrilateral is a search constraint, not the source of
the final square corners.

## 4. Preserved Legacy Path

The following shall remain unchanged while the coded-pattern method is
proposed or experimental:

- Gray-code pattern generation and capture.
- Existing dense decoder and RANSAC mapping.
- Existing validation-only action.
- Existing quick realignment.
- Existing saved-calibration loading.
- Existing projector mesh rendering.

The UI shall identify the two full-calibration methods separately while both
exist. Failure of the coded-pattern method shall not delete or overwrite the
last verified calibration and shall not automatically invoke the legacy path
without informing the user.

## 5. Definitions

### 5.1 Rough board region

A camera-space polygon or mask expected to contain the complete playing
surface, with sufficient margin for localization error. It may be derived from:

- Alternating square colors.
- Visible internal chess lines.
- A previous verified board position.
- Approximate outer edges.
- Optional user-assisted corner selection.

It need not identify final square corners accurately.

### 5.2 Physical chess boundary

A visible image segment belonging to one of the nine horizontal or nine
vertical boundaries of the 8x8 playing surface.

### 5.3 Coded projector feature

A detected marker, line segment, point, or local neighborhood whose identity
determines its projector-space coordinate without counting from an image edge.

### 5.4 Board-plane support

Decoded coded-pattern observations consistent with one dominant planar
camera-to-projector transformation.

## 6. Pattern Requirements

### 6.1 Unique local identity

The normal focus checkerboard is periodic and therefore ambiguous. The
calibration pattern shall instead have unique local neighborhoods.

Acceptable designs include:

- ArUco- or AprilTag-style marker tiles.
- A De Bruijn-coded line grid.
- Asymmetric dot clusters with encoded row and column identities.
- Another binary code with equivalent local uniqueness and error detection.

Seeing a sufficiently large local fragment shall determine its absolute
projector coordinates without requiring an uninterrupted path to the pattern
center.

### 6.2 Central anchor

The pattern should include a visually and algorithmically distinctive central
anchor. The anchor may accelerate detection but shall not be the only source
of absolute position because it may be hidden by a piece.

### 6.3 Spatial distribution

Features shall cover the projector image densely enough that the portion
falling on the board provides observations in multiple board regions.

The feature layout shall not depend on the projector center landing at the
physical board center. The rough board region may appear anywhere within the
projector image.

### 6.4 Optical properties

The initial implementation shall:

- Use a black background.
- Use measured safe projector intensity, initially 112.
- Avoid feature widths below the measured optical resolution.
- Include quiet zones needed by the selected decoder.
- Avoid large continuously bright regions that saturate light squares.
- Remain decodable after camera perspective distortion and moderate blur.

### 6.5 Error detection

Feature codes shall detect invalid or incomplete local patterns. An ambiguous
feature shall be discarded rather than assigned a guessed projector
coordinate.

## 7. Capture Sequence

The normal sequence shall be:

1. Wait for a stable camera frame and no hand motion.
2. Make the projector black.
3. Capture and average a short black-reference burst.
4. Estimate the rough board region and visible chess-boundary evidence.
5. Project the coded pattern.
6. Wait only for the measured projector/camera settling time.
7. Capture and average a short coded-pattern burst.
8. Make the projector black immediately.
9. Decode and fit offline from the captured frames.

The visible coded-pattern interval should target 250 milliseconds or less.
Actual timing shall be measured on hardware. Frame averaging may extend the
total operation while keeping the pattern visible only for the minimum
necessary interval.

Capture timeout, cancellation, projector loss, camera resolution changes, and
application closure shall return the projector to black.

## 8. Black-Reference Processing

The black-reference image shall be used to:

- Detect visible chess-square boundaries without projected contamination.
- Subtract ambient board texture from the coded-pattern image.
- Measure local modulation.
- Identify saturation and underexposure.
- Detect motion between the reference and pattern captures.

If board or camera motion between the two captures exceeds the configured
threshold, the measurement shall be rejected as inconclusive.

## 9. Rough Board Localization

The rough region shall:

- Include the full playing surface with margin.
- Exclude unrelated checkerboards where possible.
- Permit temporary uncertainty at occluded edges.
- Remain stable across the black and coded captures.

The rough region shall not:

- Be saved as the final 9x9 raster.
- Be assumed to have exact playing-surface corners.
- Be selected solely because it is the largest rectangular silhouette.
- Include background coded features in the calibration fit.

## 10. Visible Chess-Boundary Extraction

The black-reference image shall be analyzed for physical square boundaries.

Evidence may include:

- Light/dark color transitions.
- Collinear boundary fragments.
- Intersections of visible horizontal and vertical fragments.
- Equal-spacing evidence after provisional rectification.
- Continuity across empty middle ranks.

Pieces and shadows may interrupt boundaries. The detector shall preserve
separate line fragments and their uncertainty rather than requiring every
boundary to be continuously visible.

The starting position is expected to expose strong evidence in the four middle
ranks. Boundaries hidden by pieces shall be inferred only through the fitted
equal-square projective model.

## 11. Coded-Pattern Decoding

The difference between coded-pattern and black-reference captures shall be
used to detect coded features.

For each accepted observation, the decoder shall provide:

- Camera-space position.
- Absolute projector-space position.
- Feature identity.
- Decode confidence.
- Local response strength.
- Local geometric residual or marker-corner quality.
- Saturation and clipping flags.

Observations with ambiguous identity, insufficient modulation, clipping, or
invalid geometry shall be rejected explicitly.

## 12. Playing-Plane Classification

Decoded observations may occur on:

- The playing plane.
- Piece surfaces.
- Decorative borders.
- Vertical board sides.
- Background objects.

The algorithm shall:

1. Fit a dominant camera-to-projector homography with robust RANSAC.
2. Classify observations by reprojection residual.
3. Reject observations on elevated or non-planar surfaces.
4. Refit using only plane-consistent observations.
5. Require spatially distributed support.

Minimum support shall include:

- At least 500 plane-consistent feature observations or an empirically
  equivalent number for sparse marker corners.
- Evidence in all four rough board quadrants.
- At least half-board span in both board directions.
- A two-dimensional support hull rather than one narrow strip.

Exact thresholds remain subject to hardware measurement.

## 12A. Adaptive Measurement-Location Selection

### 12A.1 Principle

The projected coded pattern may cover the complete rough board region, but the
fit shall use only observations suitable for measurement. Suitability shall be
determined from captured evidence rather than from fixed square-center
locations.

Selection shall not require the final 9x9 lattice to be known already.

### 12A.2 Candidate quality

Each candidate observation or local patch shall receive a quality score based
on:

- Decode confidence and unique feature identity.
- Projector modulation relative to the black reference.
- Board-plane homography residual.
- Distance from probable piece surfaces.
- Distance from saturation, deep shadow, glare, and image clipping.
- Local sharpness and feature-corner precision.
- Temporal consistency between averaged frames.
- Camera-space and projector-space uncertainty.
- Proximity to a visible physical chess-boundary fragment.
- Usefulness for constraining an unmeasured board region.

Candidates shall be rejected when any mandatory confidence threshold fails.
A large number of weak observations shall not outweigh a smaller set of
reliable planar observations.

### 12A.3 Geometric distribution

The selected fitting set shall:

- Include evidence in all four rough board quadrants.
- Span both board directions.
- Include observations near the useful extremes of visible support.
- Avoid excessive clustering on one square, rank, file, or marker.
- Avoid using multiple highly correlated observations when they add little
  geometric information.
- Maintain a well-conditioned homography and lattice fit.

Selection may maximize a conditioning metric such as determinant,
minimum singular value, or expected parameter-information gain while
respecting confidence and distribution constraints.

### 12A.4 Chess-boundary utility

Candidates close to visible physical chess boundaries are especially useful
for the direct projector-space lattice. The selector shall distinguish:

- Plane observations used to establish camera-to-projector correspondence.
- Boundary observations used to constrain rank and file positions.
- Observations that can serve both purposes.

Boundary candidates shall be distributed among horizontal and vertical
families. The selector shall not allow one strongly visible family to dominate
while the perpendicular family remains underconstrained.

### 12A.5 Fitting and held-out sets

Selected reliable observations shall be partitioned into:

- A fitting set.
- A spatially independent held-out validation set.

Held-out observations shall:

- Not be used to select local correction values at the same location.
- Cover multiple board regions.
- Include both chess-boundary families where possible.
- Remain sufficient after any ambiguous or non-planar candidates are removed.

Selection shall be repeated if a candidate becomes invalid during fitting.
The replacement shall come from the same underrepresented region or
constraint family where possible.

### 12A.6 Adaptive projected validation

When a second validation projection is used, its targets shall be chosen from
the measured visibility and confidence masks. Targets shall prefer:

- Exposed planar patches.
- Strong predicted camera response.
- Regions not used to fit the candidate mapping.
- Locations that test extrapolated or weakly supported mesh areas.

Targets shall avoid piece masks, elevated surfaces, board sides, decorative
borders, saturation, and known weak-response areas.

### 12A.7 Insufficient support

If no selection satisfies the minimum confidence, distribution, conditioning,
and held-out requirements, calibration shall return `inconclusive`.

The application shall:

- Make the projector black.
- Preserve the previous verified calibration.
- Report which board regions or constraint families lack support.
- Suggest removing only the specific obstruction that prevents a safe fit,
  rather than requesting an empty board.

## 13. Direct Projector-Space Chess Lattice

Before fitting, the system shall select a geometrically useful subset of the
decoded board-plane observations as specified in Section 12A.

### 13.1 Model

The physical board shall be represented by canonical coordinates from
`(0, 0)` through `(8, 8)`.

The final lattice shall be a projective image of that canonical equal-square
plane in projector coordinates. It shall not be obtained by dividing camera
image edges into equal pixel lengths.

### 13.2 Observation conversion

For each reliable visible chess-boundary observation:

1. Determine its camera-space position in the black-reference image.
2. Interpolate the decoded camera-to-projector field from nearby
   board-plane observations.
3. Convert the boundary observation into projector coordinates.
4. Retain its line orientation, boundary-family hypothesis, and confidence.

### 13.3 Joint fit

The lattice fit shall jointly estimate:

- The projector-space 9x9 grid.
- Assignment of visible fragments to horizontal or vertical boundary indices.
- Perspective transformation from canonical board coordinates.
- Optional smooth local optical correction.

The objective shall combine:

- Distance from converted boundary points to their assigned projector-space
  chess boundaries.
- Equal-square projective consistency.
- Coded-feature reprojection residual.
- Confidence weights.
- A smoothness penalty for local corrections.
- A prior from the previous verified correction mesh when available.

### 13.4 Boundary indexing

Alternating square colors, line ordering, equal spacing, and the rough board
extent shall determine which canonical boundary each visible fragment
represents.

The fit shall test plausible index assignments and reject ambiguous solutions.
A decorative border shall not be accepted as a rank or file merely because it
has approximately one-square width.

## 14. Hidden-Region Extrapolation

The global projective lattice may be extrapolated through piece-covered
regions when measured support is distributed.

Local optical corrections shall be handled as follows:

- Reuse the previous verified correction field when camera and projector
  hardware remain fixed.
- Measure corrections where current plane support exists.
- Interpolate unsupported nodes smoothly.
- Mark every node as measured, inherited, or interpolated.
- Reject calibration when unsupported extrapolation exceeds configured limits.

Piece surfaces shall never provide local board-plane corrections.

## 15. Validation

Validation shall be independent of the observations used to fit the lattice.

Acceptable validation methods include:

- Reserved coded features not used for fitting.
- A second short pattern with different encoded locations.
- Existing adaptive projected targets on visible middle-board patches.

Validation shall report:

- Per-target or per-feature camera-pixel error.
- Error as a fraction of local square width.
- Mean, RMS, 95th-percentile, and maximum error.
- Support distribution.
- Measured, inherited, and interpolated mesh regions.

The initial acceptance limit remains a maximum error below 10% of local square
width. The desired later target remains 3-5%.

A missing, ambiguous, or non-planar validation observation shall not be
treated as zero error.

## 16. Runtime and Recalibration

### 16.1 Fixed hardware

When camera, projector, and board do not move, the saved calibration shall be
restored normally. Validation-only operation shall remain available.

### 16.2 Board movement

For small in-plane board movement:

- Camera tracking may continue remapping the verified lattice.
- A short coded-pattern capture may provide a measured quick realignment.
- The previous local correction field may be transferred to the new pose.

### 16.3 Camera or projector movement

Movement that changes the camera-to-projector relationship requires a new
coded-pattern measurement or the legacy full calibration.

## 17. User Interface

While experimental, the UI should provide:

- **Full calibration - coded pattern**
- **Full calibration - legacy Gray code**
- **Validate calibration**
- **Quick realign / assess**
- **Cancel calibration**

The status area shall distinguish:

- Capturing black reference.
- Projecting coded pattern.
- Decoding features.
- Fitting board plane.
- Fitting projector-space chess lattice.
- Validating.
- Inconclusive support.
- Failed calibration.
- Successful calibration.

No unsuccessful candidate shall replace the previous verified calibration.

## 18. Diagnostics

Each coded-pattern attempt shall optionally save:

- Black-reference image.
- Raw coded-pattern image.
- Difference image.
- Detected feature identities and camera positions.
- Projector coordinates for every accepted feature.
- Rejected ambiguous features.
- Plane-inlier and non-planar masks.
- Visible chess-boundary fragments.
- Boundary-index assignments.
- Projector-space fitted 9x9 lattice.
- Per-observation and validation residuals.
- A machine-readable JSON report.

Diagnostics shall make it possible to distinguish:

- Pattern decode failure.
- Board-line detection failure.
- Piece-surface contamination.
- Insufficient distribution.
- Incorrect boundary indexing.
- Projector-space lattice error.

## 19. Automated Tests

Automated tests shall include:

1. Unique decoding of every projected feature.
2. Partial pattern visibility with the central anchor hidden.
3. Perspective, blur, exposure, and moderate noise.
4. Starting-position piece occlusion.
5. Rejection of marker fragments projected onto elevated pieces.
6. Decorative border approximately one square wide.
7. Missing outer board boundaries.
8. Rough region translated or enlarged relative to the true board.
9. Recovery of the correct projector-space 9x9 lattice.
10. Extrapolation through hidden ranks.
11. Rejection of support confined to one strip or quadrant.
12. Preservation of the previous calibration on failure.

Synthetic acceptance shall compare every recovered projector-grid node with
ground truth, not only the global homography residual.

## 20. Hardware Acceptance

Hardware acceptance shall use:

- Logitech Webcam C930e at 1920x1080.
- MPR 600 projector at 1280x720.
- Normal starting position.
- Projector intensity 112 unless remeasured.

The method shall demonstrate:

1. The pattern is visible for no longer than the configured short interval.
2. The complete capture and fit succeed with all pieces present.
3. A hidden central anchor does not prevent absolute decoding.
4. At least one usable plane observation exists in every board quadrant.
5. The projected 8x8 grid follows the physical square boundaries.
6. Maximum independent validation error remains below 10% of a square.
7. Repeated calibrations are stable.
8. Failure leaves the previous verified calibration active.

Results shall be compared directly with the legacy Gray-code path on the same
unchanged hardware setup.

## 21. Implementation Milestones

The first experimental implementation uses a native-resolution field of
`DICT_4X4_100` ArUco markers. It captures a black reference and one coded
frame, rejects non-plane marker corners with RANSAC, and requires support in
all four board quadrants.

On a populated starting-position board, it detects a visible 7x5 block of
physical chess intersections. The rough raster is used only to select its
orientation and one of the possible five-rank spans. A projective equal-square
9x9 lattice is then fitted to those measured intersections and transformed
into projector coordinates. Hidden outer ranks are extrapolated from that
fit. Rough outer-boundary observations may weakly anchor the extrapolation
only when they remain within 20% of a square and the rough interior has no
significant systematic translation relative to the measured intersections.
This improves outer-corner stability without accepting a decorative
one-square border as part of the playing surface.

This path is exposed as **Full calibration - coded pattern**. It reuses the
independent validation sequence and leaves the previous verified calibration
active if capture, fitting, or validation fails. **Full projector calibration**
continues to invoke the legacy Gray-code path.

### Milestone 1: Pattern experiment

- Generate a native-resolution candidate coded pattern. Implemented.
- Project it at native resolution. Implemented.
- Capture black and patterned frames. Implemented.
- Measure modulation, blur, saturation, and decode rate.
- Compare alternative encodings on physical hardware.

### Milestone 2: Single-shot projector decoding

- Decode absolute projector coordinates. Implemented.
- Produce confidence and ambiguity masks.
- Fit and validate the dominant playing plane. Implemented.
- Require distributed quadrant support. Implemented.
- Add finer confidence scoring and conditioning-based selection.

### Milestone 3: Visible chess-boundary observations

- Detect a contiguous visible 7x5 intersection block in the black reference.
  Implemented.
- Select its rank span and orientation against the rough board region.
  Implemented.
- Generalize detection to disconnected boundary fragments.

### Milestone 4: Direct projector-space lattice

- Fit the canonical 9x9 lattice. Implemented.
- Extrapolate hidden boundaries. Implemented.
- Transfer or interpolate local corrections.

### Milestone 5: Independent validation and UI

- Reuse held-out validation. Implemented.
- Reuse adaptive nearby validation fallbacks. Implemented.
- Add diagnostics and failure reporting.
- Add explicit coded-pattern calibration control. Implemented.
- Preserve the legacy control. Implemented.

### Milestone 6: Hardware comparison

- Compare accuracy, repeatability, visibility time, and failure rate with
  legacy Gray code.
- Promote the coded method only after it is measurably superior.

## 22. Open Decisions

- Marker tiles versus De Bruijn lines versus asymmetric dot clusters.
- Minimum projected feature width.
- Whether one coded frame is sufficient or an inverse frame is justified.
- Minimum number of unique decoded features.
- Measurement-selection conditioning metric and regional quotas.
- Boundary-fragment index-assignment algorithm.
- Local correction interpolation method and unsupported-area limit.
- Whether the central anchor should be mandatory when visible.
- Exact capture duration and number of averaged camera frames.
- Whether quick realignment can use the same coded pattern at lower density.
