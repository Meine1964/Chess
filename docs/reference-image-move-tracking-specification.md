# Reference-Image Chess Move Tracking Specification

**Version:** 0.3
**Status:** Implemented for ordinary moves, captures, castling, en passant, and promotion signatures
**Platform:** Python on Windows  

**Related specifications:**

- [Projector/camera chessboard system](./projector-camera-chessboard-specification.md)
- [Chess-square occupancy and UI marking](./chess-square-occupancy-ui-specification.md)
- [Chess-piece UI rendering and move synchronization](./chess-piece-ui-tracking-specification.md)
- [Lichess physical-board integration roadmap](./lichess-integration-roadmap.md)

## 1. Purpose

This specification defines an event-driven method for tracking physical chess
piece relocations by comparing each settled camera image with the image stored
after the last accepted position.

After initialization, move tracking shall answer:

1. Has activity occurred above the board?
2. Has the hand or other obstruction left the board?
3. Which squares changed relative to the last accepted position?
4. Do the changed squares match one permitted physical signature, including
   multi-square and staged special-move evidence?

The system shall not continuously derive a complete logical position from
independent absolute classifications of all 64 squares.

## 2. Design Principle

The last accepted position is authoritative.

After the standard starting position has been confirmed, the application shall
store:

- The logical piece position.
- The accepted logical occupied-square set.
- A canonical rectified reference image of the board.
- Per-square reference descriptors derived from that image.
- The camera-grid generation and orientation used to create the reference.

The logical position and reference shall change only after an unambiguous
permitted physical signature is accepted, a verified starting-position reset
completes, or explicit reconciliation adopts an authoritative position.

Rejected, obstructed, unstable, or ambiguous observations shall never replace
the reference.

## 3. Scope

### 3.1 Included

The first implementation shall support:

- Initialization from the complete standard starting position.
- Detection of activity above the board.
- Detection that the board is clear and visually settled.
- Comparison with the last accepted reference.
- One known piece moving from one source square to one empty destination.
- Deliberately non-legal chess relocations.
- Persistent UI source/destination highlighting.
- Optional two-second physical source/destination projection.
- Camera-only operation when the calibrated projector is disconnected.
- Conservative rejection without loss of the accepted position.

### 3.2 Excluded from the first implementation

- Captures.
- Castling.
- En passant.
- Promotion identity.
- Simultaneous movement of multiple pieces.
- Arbitrary-position piece recognition.
- Recovery of piece identity from occupancy alone.
- Chess-rule or turn-order enforcement.

Chess legality shall not be used to accept or reject a physical relocation.

## 4. Coordinate Systems

### 4.1 Camera frame

The raw USB-camera image may contain perspective, board movement, surrounding
objects, and illumination changes.

### 4.2 Camera grid

The current camera-space `9x9` grid defines the 64 physical board squares.
Board localization and alignment monitoring own this grid.

### 4.3 Canonical board image

Every image used for comparison shall be rectified through the current camera
grid into a fixed-size top-down board image.

The initial implementation shall use an `800x800` canonical image, producing
`100x100` pixels per square.

### 4.4 Logical UI coordinates

Physical squares shall be transformed through the persisted complete
quarter-turn orientation before they are associated with logical source and
destination squares.

The logical position shall remain independent of camera and projector
coordinates.

## 5. Initialization

Initialization shall require:

- A valid camera-space board grid.
- A stable, unobstructed camera image.
- Complete binary occupancy with exactly the standard 32 starting squares.
- White-side orientation established from physical piece colors.

After confirmation, the application shall:

1. Create the standard logical position.
2. Rectify the accepted camera image.
3. Store it as the reference image.
4. Compute and store per-square reference descriptors.
5. Enter the `watching` state.

Absolute 64-square occupancy classification may be used for initialization.
It shall not remain the primary move-tracking mechanism afterward.

## 6. Reference Image

### 6.1 Contents

The stored reference shall contain:

- Canonical color image.
- Canonical luminance image.
- Per-square normalized color histograms.
- Per-square edge or texture descriptors.
- Logical position.
- Logical occupied-square set.
- Camera-grid generation.
- Orientation generation.
- Acceptance timestamp or frame sequence.

### 6.2 Persistence

The logical position shall persist across process restarts.

The canonical reference may also persist when it can be associated safely with
the same:

- Camera identity.
- Camera resolution.
- Board calibration.
- Orientation.

If the persisted image reference cannot be trusted after restart, the logical
position shall remain visible, but automatic move acceptance shall remain
paused until a new reference is established from a camera observation matching
the logical occupied-square set.

### 6.3 Replacement

The reference shall be replaced only:

- After initialization.
- After an accepted move.
- After an explicit user reset.
- After an explicit reference-recapture operation that verifies the physical
  occupancy matches the logical position.
- After the user starts an Offline game from the accepted current position.

Current-position Offline startup shall not compare the live image with the
previous reference. It shall black the projector, discard at least one second
of exposure transition, require a stable quiet observation, and install and
persist that observation as the new reference before entering `watching`.

A saved Offline game restored at process startup shall follow the same
recapture sequence even when the saved legal position and accepted piece map
already match. The accepted piece map remains authoritative, and incomplete
binary occupancy shall not block recapture. Until a stable quiet frame has
replaced the reference, the persisted appearance reference shall not be used
for move tracking.

When a confidently detected king relocation exactly matches the king leg of
an available castling signature, the tracker shall enter `pending-castling`
even if appearance classification cannot confidently prove that the rook
squares are unchanged. The authoritative legal signature supplies the rook
source and destination. The castling move remains unaccepted until the full
four-square signature is observed.

## 7. Runtime State Machine

The tracker shall expose the following states.

### 7.1 `uninitialized`

No accepted logical position or reference exists.

### 7.2 `watching`

The board matches the reference closely enough that no activity is underway.

The application performs only the lightweight activity check.

### 7.3 `activity`

A significant image difference indicates a hand, arm, piece, shadow, or other
object is moving above the board.

No move inference or occupancy update is allowed in this state.

### 7.4 `waiting-for-clear`

Motion has reduced, but a large connected foreground region or widespread
change still suggests that the board is obstructed.

### 7.5 `settling`

The board appears clear. The tracker waits for a small number of mutually
similar frames to ensure the candidate position is stationary.

### 7.6 `candidate`

A settled image differs from the reference on the squares required by one
ordinary relocation or supplied legal physical signature. Targeted source,
destination, captured-piece, rook, or en-passant-square tests are running as
required by that signature.

### 7.7 `accepted`

One unambiguous relocation or completed legal physical signature has been
accepted atomically. The UI and reference are updated before returning to
`watching`.

### 7.8 `ambiguous`

The settled image cannot be represented safely as one supported relocation or
permitted legal physical signature. The accepted position and reference remain
unchanged. The tracker continues comparing future clear frames with the same
reference.

## 8. Lightweight Activity Detection

Activity detection shall run continuously on a low-resolution canonical board
image.

The detector shall compare the current image with:

- The immediately preceding camera image, to measure motion.
- The accepted reference image, to measure persistent board change.

Useful signals include:

- Mean absolute luminance difference.
- Fraction of pixels exceeding a difference threshold.
- Size of the largest connected changed region.
- Number of board squares intersected by that region.
- Optical or translational board movement reported by alignment monitoring.

A hand or arm normally creates a large connected changed region spanning
multiple squares. A completed one-piece move normally leaves localized
persistent changes concentrated in two squares.

Before connected-component analysis, thin board-edge residuals caused by a
small camera-grid remap shall be removed. Those residuals shall not be joined
into a false board-sized obstruction. The shape filter shall retain broad,
filled regions produced by a hand, arm, or other real obstruction.

A component shall count as covering a square only when a meaningful fraction
of that square's pixels belongs to the component. Its rectangular bounding box
shall not determine the covered-square count, because a wide piece silhouette
may slightly cross a neighboring square boundary without occupying that square.

A sufficiently large filled region spanning two adjacent squares shall also
remain obstructed. Before accepting a candidate, source and destination change
footprints shall be compatible with a released physical piece. A hand-sized
single-square footprint or a destination footprint grossly larger than the
vacated source shall remain `waiting-for-clear`. These checks shall not add a
fixed post-motion delay.

Obstruction classification shall not reject a settled frame solely because the
connected source and destination silhouettes are large. Candidate squares
shall first be localized, then excluded from the general obstruction mask.
Obstruction evidence elsewhere on the board shall continue to block the move.
The released-piece footprint bound shall accommodate the largest normal chess
piece while remaining below the tested hand-sized footprint. The implemented
profile permits up to 70% changed-pixel coverage within either candidate square;
the hand-sized regression exceeds that bound.

Skin-color detection shall not be required.

## 9. Board-Clear Detection

The tracker shall evaluate move evidence only after the board is clear.

The board may be declared clear when:

- Frame-to-frame motion is below threshold.
- No large connected changed region spans multiple squares.
- No board boundary is obscured.
- The projector output is black or unavailable.
- Camera-grid alignment is valid.
- Two or three consecutive clear frames are mutually similar.

The clear decision shall be evidence-based, not a fixed one-second delay.

The target clear-detection latency is `100-300 ms` after the hand leaves the
board under stable camera conditions.

## 10. Photometric Normalization

Before comparison, the candidate and reference shall be normalized to reduce
false differences caused by camera exposure and illumination.

Normalization may include:

- Small global translation registration against the accepted reference.
- A robust luminance offset estimated from aligned per-pixel differences.
- Per-channel normalization in LAB color space.
- Mild blur before low-resolution activity comparison.
- Exclusion of clipped highlights and deep shadows from gain estimation.

Normalization shall preserve localized piece appearance changes.

The application shall reject a frame rather than normalize it when the global
illumination change is too large or spatially non-uniform.

## 11. Per-Square Difference Measurement

For every square, the tracker shall calculate a change score between the
reference and settled candidate.

The score should combine:

- Interior absolute color difference.
- Normalized color-histogram distance.
- Edge-map difference.
- Texture difference.
- Change in a lower/base region near the board plane.

Square-boundary margins shall reduce contamination from adjacent squares.

The lower/base region shall receive additional weight because tall piece
silhouettes may extend into neighboring squares while the physical piece base
remains on only one square.

## 12. Candidate-Square Selection

The 64 square-change scores shall be ranked.

A candidate one-piece relocation requires:

- Exactly two dominant changed squares, or two squares whose combined evidence
  is clearly separated from all remaining squares.
- One dominant square corresponding to a known logical source piece.
- One dominant square corresponding to a logically empty destination.
- All remaining square changes below the ambiguity threshold.

No chess movement rule shall constrain the source/destination pair.

The unrestricted two-square rule applies to Board setup / diagnostic movement.
When Offline legal play or Lichess synchronization supplies permitted physical
signatures, candidate selection shall instead match one exact signature.
Supported signatures may include an occupied capture destination, four
castling squares, a separate en passant captured-pawn square, or promotion
source/destination evidence. Recognized captured-piece-first, king-first, and
captured-pawn-first evidence shall remain pending against the original
reference until the signature completes.

For an ordinary permitted relocation, the destination may use the same reduced
evidence floor as the vacated source only when piece-color classification
independently confirms that the source is empty and the destination contains
the moving piece's color. An uncertain or wrong-color destination shall retain
the normal evidence floor. This exception supports low-contrast quiet moves
without globally lowering movement or ambiguity thresholds.

If more squares remain materially changed than the active signature permits
after normalization and boundary suppression, the observation shall normally
be ambiguous. A
high-scoring square adjacent to the provisional source may be treated as a
silhouette residual only when it is logically occupied by the opposite color
and its current lower/base color signature still matches the accepted
reference. This targeted check confirms that the competing piece remains in
place; logical color alone shall not suppress the ambiguity.

## 13. Targeted Vacancy Tests

Absolute vacancy classification shall be performed only on squares involved
in the candidate relocation or active legal signature.

### 13.1 Source test

The source test shall confirm that:

- A known logical piece occupied the source in the reference position.
- The candidate source's base-region evidence decreased substantially.
- The candidate source now resembles an empty square of matching parity.
- Residual upper-body silhouettes from adjacent pieces do not count as a
  remaining source piece.

### 13.2 Destination test

The destination test shall confirm that:

- The logical destination was empty, unless the active signature is a capture.
- Piece-base evidence increased substantially.
- The candidate destination no longer resembles its reference empty square.

### 13.3 Confirmation

Both tests shall pass on at least two mutually similar settled frames before
the move is accepted.

## 14. Move Acceptance

An accepted move shall be applied atomically:

1. Apply the complete accepted relocation or legal move signature to the
   authoritative physical position.
2. Preserve the selected promotion identity and every capture, rook, or en
   passant side effect represented by that signature.
3. Update the UI piece map.
4. Highlight source and destination on the UI until the next accepted move.
5. Store the settled candidate as the new reference.
6. Persist the logical position and reference metadata.
7. Return to `watching`.

Offline legal play shall project the accepted source and destination for
exactly one second after the move and its new reference have been committed.
The marking may include the optional next-turn dot. Ordinary move verification
shall remain paused while the marking is visible. After one second, the
projector shall return to black and verification shall remain paused for a
fixed one-second camera-exposure settling interval before resuming against the
committed reference.

This brief lifecycle shall use generation-guarded timers so stale callbacks
cannot clear a newer marking. It shall not use hand detection, adaptive
settling, automatic reprojection, or a configurable duration. Diagnostic
movement may retain temporary accepted-square projection, and illegal
correction may retain its red projection; neither shall expose the turn dot.

Illegal-relocation correction shall project both the authoritative source and
observed destination in red until a board-and-halo activity detector confirms
that the restoring hand has entered and subsequently left. Camera comparison
shall resume only after the red output is removed and projector light has
settled. If the rejected physical position remains, the same two-square red
correction shall be restored; a correction shall never degrade to showing only
one of the two relocation squares. When an empty source and same-color
destination are both confidently classified, the complete illegal relocation
shall take precedence over an illegal-removal hypothesis for that source.
Correction hand detection shall exclude the two illuminated square polygons
from image differencing so steady red output cannot trigger its own blackout;
the surrounding board and one-square halo remain active for hand detection.
This correction-specific detector shall require a hand-sized connected change
of at least half the median square area, preventing small calibration-edge
spill or local exposure variation around the red polygons from being treated
as a restoring hand. Before genuine manipulation begins, it shall compare
against a rolling previous-frame baseline rather than one fixed projected
frame, so gradual exposure drift cannot accumulate into false activity. One
hand-sized temporal change is sufficient to begin manipulation tracking;
afterward, consecutive quiet observations determine when the hand has left.
For illegal-correction tracking, the connected change must also extend from
the board into the one-square exterior halo. A restoring hand or arm enters
through this halo, while projector light confined to the calibrated 8x8 board
cannot satisfy the condition.

Resetting any guidance or correction activity detector shall return it to a
fully inactive lifecycle state: baseline, arming progress, activity
confirmation, active-manipulation state, and quiet progress shall all be
cleared. A detector reused after blackout shall never inherit a completed
manipulation from the preceding projection.

Runtime diagnostics shall record each illegal-correction lifecycle transition
once: projection or reprojection, confirmed hand entry, blackout after hand
departure, readiness for black-frame verification, and restoration of the
accepted reference. Each event shall include the correction generation and
relevant logical or physical squares, and the current diagnostic state shall
expose projection-active, blackout-pending, and detector-active flags.

When legal signatures are supplied and no signature matches, an otherwise
unambiguous settled source-to-empty-destination relocation shall be reported
as an illegal relocation rather than accepted or left ambiguous. This outcome
shall not advance the reference or authoritative position. Offline legal mode
shall use it to mark and project the observed source and destination in red,
including when a player moves the wrong side's piece. Illegal-relocation
inference shall use piece-color classification to require that the source now
looks empty and the destination now looks like the source piece's color;
small change-score residuals on still-occupied squares shall not compete as
possible sources.

No intermediate repaint shall show the piece missing from both squares or
present on both squares.

## 15. Ambiguity and Failure Behavior

The tracker shall reject, without changing the reference or logical position:

- A hand or object still above the board.
- Global or non-uniform illumination changes.
- Camera-grid alignment uncertainty.
- More materially changed squares than the candidate signature permits.
- No dominant source.
- No dominant destination.
- Source without a known logical piece.
- Destination occupancy that is inconsistent with the candidate signature.
- Weak or contradictory base-region vacancy evidence.
- Candidate frames that do not agree.

The normal user-facing status line shall present stable, actionable guidance
rather than every internal tracker transition. `activity`, `settling`, and
`candidate` are short-lived processing phases and shall not overwrite the last
useful instruction. The visible line shall change for durable or actionable
states such as:

- Move a hand or object away from the board.
- Restore an unclear board change.
- Complete a pending capture.
- Correct an illegal move or removal.
- Confirm an accepted move.
- Confirm that the board matches the accepted reference.

Tracker state, dominant changed squares and scores, source/destination evidence,
and rejection details shall remain available to diagnostics rather than
flashing through the normal UI.

It shall not display raw absolute occupancy dots as authoritative move state
after initialization. Diagnostic overlays may show difference scores.

## 16. Board Movement and Remapping

Board alignment shall remain independent of move inference.

When alignment monitoring updates the camera grid:

1. Treat the updated grid as pending while a physical move may still be under
   evaluation. Alignment monitoring shall not reset a candidate, supported
   multi-stage move, accepted move awaiting submission, or illegal-move
   correction before that outcome is handled.
2. Commit the pending grid after the current move evaluation reaches a safe
   boundary. When no move evaluation is active, a bounded delay may commit the
   grid so genuine whole-board movement continues to be followed.
3. Pause activity and move evaluation once for the committed grid.
4. Rectify the current frame with the new grid.
5. Register any small residual canonical translation against the accepted
   reference.
6. Suppress thin grid-edge residuals before obstruction analysis.
7. Resume `watching` with the existing accepted canonical reference.

A camera-grid change shall never clear or replace the logical piece position.

Geometry changes themselves shall not be treated as piece movement.

Repeated near-threshold grid fits shall not starve physical move recognition.
When alignment and move observations overlap, the localized chess-move
evaluation has priority over committing the pending global remap.

If a settled one-piece relocation has dominant source/destination evidence and
a valid released-piece footprint, broad residuals confined to the outer board
files or ranks may be classified as camera-grid remap residuals rather than a
hand only when they touch the canonical image boundary and do not overlap the
candidate squares. This exception exists so an illegal relocation can still
reach correction guidance after remapping. Interior obstruction components,
or any boundary component overlapping the candidate, shall continue to block
acceptance. The exception shall never commit an illegal move; it only permits
the existing legality rejection and red correction workflow on both source and
destination to run.

## 17. Camera-Only Operation

The tracker shall operate without an attached projector when:

- The camera identity and resolution match saved camera calibration.
- The live camera grid has been checked or remapped.
- The projector is absent, closed, or black.

Projector controls and physical illumination shall be unavailable, but move
detection, UI updates, reference storage, and persistence shall continue.

The application shall not open an unrelated display as a substitute projector.

## 18. Diagnostics

For every activity episode, diagnostics shall be able to record:

- Reference image.
- First motion frame.
- Last obstructed frame.
- Settled candidate image.
- Normalized difference image.
- Connected-component mask.
- Per-square change scores.
- Source and destination base-region crops.
- Camera grid and orientation.
- State transitions and timestamps.
- Acceptance or rejection reason.

Diagnostics shall make it possible to reproduce a missed or incorrect move
without relying on screenshots of the UI.

Routine `watching`, `settling`, `activity`, and first-candidate observations
shall not synchronously encode images or write runtime files. Significant
tracker decisions may capture immutable diagnostic snapshots on the UI
thread, but JSON serialization, PNG encoding, event truncation, and disk
writes shall run on one bounded background writer. The move decision shall
continue immediately after a diagnostic record is queued. Repeated ambiguous
or pending observations shall be coalesced until the tracker changes state.
Queue overflow and writer failures shall be surfaced explicitly rather than
silently discarding significant diagnostics.

## 19. Performance Requirements

On the current 1920x1080 C930e camera:

- Lightweight activity detection should run at camera rate or at least 10 Hz.
- Board-clear detection target: `100-300 ms` after the hand leaves.
- Settled candidate comparison target: under `100 ms`.
- Typical accepted-move latency target: under `700 ms` after the hand leaves.
- UI and projector updates shall not block camera capture.
- Runtime diagnostic compression and file output shall not block move
  acceptance.

Performance optimization shall not weaken reference integrity or ambiguity
rejection.

Per-square appearance features derived from the accepted reference shall be
computed once and cached with the logical position used to label reference
samples. Candidate observations shall reuse that immutable model. The cache
shall be invalidated whenever the reference is cleared, replaced, remapped,
or advanced after an accepted move, and whenever the supplied logical
position differs. Cached classification shall produce the same labels,
distances, and margins as classification from a freshly built model.

Before complete legal-signature residual analysis, the tracker shall apply a
conservative change-score prefilter. A signature may be excluded from that
expensive path only when its source or destination lacks even the lowest
change evidence that the complete matcher could accept. Partial capture,
castling, en-passant, promotion, removal, and illegal-relocation analysis
shall continue to receive the full authoritative signature set. Prefiltering
shall therefore change processing cost, never the accepted or rejected
result.

## 20. Tests

### 20.1 Initialization

- Standard starting position creates a reference.
- Incomplete setup does not initialize.
- White-side orientation is retained.

### 20.2 Activity and board-clear detection

- Moving hand enters `activity`.
- Stationary hand remains obstructed.
- Hand removal reaches `settling` quickly.
- Shadows spanning several squares remain obstructed.
- A stationary clear board returns to `watching`.

### 20.3 Difference localization

- `e2-e4` produces dominant changes on `e2` and `e4`.
- `Nb8-c6` produces dominant changes on `b8` and `c6`.
- Deliberately non-legal `Nb8-d8` is supported.
- Tall-piece silhouettes do not create a third authoritative square.
- Small grid translations do not become candidate moves.
- `Ng1-f3` remains detectable after a small perspective/grid remap.
- A wide king crossing a square boundary is not classified as a multi-square
  obstruction.
- `Ke1-f1` remains detectable when the released king covers slightly more than
  60% of its destination square.
- `Qd2-h6` remains detectable when uncovering the opposite-color queen on d1
  creates a strong adjacent residual but the d1 lower/base signature remains.
- An equivalent same-color adjacent residual remains ambiguous.
- Thin grid-edge residuals do not become a false multi-square obstruction.
- Moderate global exposure changes normalize away.

### 20.4 Vacancy tests

- Vacated source passes the empty-base test.
- Destination passes the occupied-base test.
- Residual source silhouette does not block acceptance.
- Adjacent real pieces remain occupied.

### 20.5 Rejection

- One changed square is rejected.
- Three or more dominant changed squares are rejected.
- Conflicting settled frames are rejected.
- Destination already logically occupied is rejected.
- Obstructed frames never replace the reference.

### 20.6 Persistence and remapping

- Logical position survives process restart.
- Trusted reference survives process restart.
- Untrusted reference pauses acceptance until recaptured.
- Camera-only startup verifies the grid before comparison.
- Board remapping preserves the logical position and reference semantics.
- Explicit reset verifies live starting occupancy before replacing either the
  logical position or reference.
- Failed reset verification preserves the previous logical position and
  reference.

## 21. Acceptance Criteria

The replacement tracker is accepted when:

1. Starting-position initialization produces a reference and logical position.
2. A hand above the board never changes the logical position.
3. A settled one-piece relocation is accepted from reference differences.
4. Typical move latency is below `700 ms` after hand removal.
5. Tall-piece spillover does not create duplicate occupancy.
6. Non-legal one-piece relocations are tracked.
7. Ambiguous changes preserve the previous reference and position.
8. Board remapping does not create false moves or clear pieces.
9. Camera-only operation works without opening a substitute projector.
10. UI highlighting and optional physical projection use only accepted moves.
11. Explicit reset succeeds from a verified physical starting setup and never
    resets from stale cached occupancy.
12. Candidate source and destination squares are excluded from general
    obstruction analysis only after localization; obstruction evidence
    elsewhere still blocks acceptance.
13. An adjacent opposite-color residual is suppressed only when its retained
    lower/base signature matches the accepted reference.
14. Captures, castling, en passant, and promotion advance the position and
    reference only after their exact legal physical signatures complete.
15. Captured-piece-first, king-first, and captured-pawn-first intermediate
    states preserve the original reference until the remaining action
    completes.

## 22. Migration Plan

### Phase 1: Reference capture and activity detection

- Capture a canonical reference after starting-position confirmation.
- Add lightweight frame/reference difference measurement.
- Add explicit tracker states and diagnostics.

### Phase 2: Board-clear and square ranking

- Detect large connected obstruction regions.
- Detect clear settled frames.
- Produce per-square change scores and diagnostic overlays.

### Phase 3: Targeted vacancy and move acceptance

- Add source/destination base-region tests.
- Update logical position and reference atomically.
- Drive UI and optional projector indications.

### Phase 4: Remove superseded move inference

After reference tracking passes recorded and live acceptance tests:

- Stop using continuous 64-square absolute occupancy as the move trigger.
- Remove hidden-source legal or pseudo-legal inference.
- Remove global spillover thresholds used only for post-initialization moves.
- Remove fixed post-motion quiet delays.
- Retain absolute occupancy only for initialization, reset verification, and
  diagnostics.

Superseded logic shall not remain active as a fallback because competing move
trackers can produce contradictory state transitions.

All four migration phases are implemented for ordinary relocations and the
supported legal special-move signatures. Absolute occupancy remains available
for initialization, explicit and automatic starting-position reset
verification, orientation, and reconciliation rather than as a competing
general post-initialization move trigger.
