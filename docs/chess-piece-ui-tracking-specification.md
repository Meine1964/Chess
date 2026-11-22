# Chess-Piece UI Rendering and Move Synchronization Specification

**Version:** 0.4
**Status:** Implemented offline legal play, special moves, recovery, and automatic starting-position reset
**Platform:** Python on Windows  
**Piece artwork:** Lichess `cburnett` SVG set  

**Related specifications:**

- [Projector/camera chessboard system](./projector-camera-chessboard-specification.md)
- [Piece-aware board localization](./piece-aware-board-localization-specification.md)
- [Chess-square occupancy and UI marking](./chess-square-occupancy-ui-specification.md)
- [Reference-image move tracking](./reference-image-move-tracking-specification.md)

The reference-image specification supersedes the post-initialization movement
inference described here. This document remains authoritative for artwork,
logical piece representation, UI rendering, and projector interaction.

## 1. Purpose

Define how the application shall:

1. Display recognizable chess pieces on the digital chessboard.
2. Initialize the UI to the standard chess starting position.
3. Keep the UI position synchronized with unambiguous physical piece moves.
4. Use the accepted reference image as post-initialization movement evidence.
5. Keep White at the bottom of the UI.
6. Refuse ambiguous updates rather than moving the wrong UI piece.

The implemented workflow supports ordinary moves, captures, castling, en
passant, auto-queen promotion, and projected physical underpromotion
selection.

## 2. Piece Artwork

### 2.1 Selected set

The application shall use the Lichess `cburnett` SVG set:

- `wK.svg`
- `wQ.svg`
- `wR.svg`
- `wB.svg`
- `wN.svg`
- `wP.svg`
- `bK.svg`
- `bQ.svg`
- `bR.svg`
- `bB.svg`
- `bN.svg`
- `bP.svg`

The first letter denotes piece color. The second letter denotes king, queen,
rook, bishop, knight, or pawn.

Official upstream directory:

`https://github.com/lichess-org/lila/tree/master/public/piece/cburnett`

### 2.2 Pinned source

Assets shall be downloaded during development from a pinned Lichess revision,
not from an unversioned runtime URL.

The initial pinned revision is:

`5b44d719ec5913dd62aa965f4b8d21e60a00c2c1`

Each raw asset URL shall have the form:

`https://raw.githubusercontent.com/lichess-org/lila/<revision>/public/piece/cburnett/<file>`

The twelve files shall be stored within the application repository so normal
operation does not require internet access.

### 2.3 Integrity

The download or import step shall:

- Require all twelve expected files.
- Reject empty or non-SVG responses.
- Record a SHA-256 checksum for every file.
- Avoid silently substituting a different piece set.
- Be repeatable from the pinned upstream revision.

### 2.4 License and attribution

Lichess identifies `public/piece/cburnett` as:

- Author: Colin M. L. Burnett.
- License: GPLv2 or later.

The repository shall include:

- A third-party notice naming the author and asset set.
- The applicable GPLv2-or-later license text or an accepted repository-standard
  license reference.
- The pinned source repository and revision.
- A statement identifying which files are covered by the third-party license.

No piece artwork shall be copied from the raster screenshot when the official
SVG source is available.

## 3. Asset Location and Loading

The recommended repository location is:

`src/chess_camera/assets/pieces/cburnett/`

The application shall locate assets relative to the installed package, not the
current working directory.

Assets shall be loaded once and cached. Missing or invalid assets shall produce
an explicit UI error; the application shall not silently render blank pieces.

Qt SVG support shall be used directly. A new Python package dependency is not
required when the installed PySide6 distribution provides `QtSvg`.

## 4. Piece and Position Model

### 4.1 Piece representation

Each logical piece shall contain:

- Color: `white` or `black`.
- Type: `king`, `queen`, `rook`, `bishop`, `knight`, or `pawn`.

A compact internal code may use:

- Uppercase `KQRBNP` for White.
- Lowercase `kqrbnp` for Black.

### 4.2 Position representation

The logical position shall map each UI square to zero or one piece.

UI square coordinates remain:

- Row `0`: rank 8.
- Row `7`: rank 1.
- Column `0`: file a.
- Column `7`: file h.

The position model shall be independent of camera-space physical coordinates.

### 4.3 Standard initialization

The initial position shall be:

- Row 0: `r n b q k b n r`
- Row 1: eight black pawns.
- Rows 2 through 5: empty.
- Row 6: eight white pawns.
- Row 7: `R N B Q K B N R`

Initialization shall occur only after:

- Camera/projector calibration is verified.
- Occupancy is stable.
- Exactly the 32 standard starting squares are occupied.
- Piece-color orientation has placed White at the bottom.

If physical occupancy does not match the starting layout, the application
shall not pretend that the UI position is synchronized.

After initialization, the logical position and its last accepted occupancy
baseline shall remain fixed across camera glitches, rejected observations,
occupancy-detector resets, calibration interruptions, and automatic board
remapping. Those events may pause move detection, but shall not clear,
reinitialize, or replace the logical position.

The application shall persist the last accepted logical position and restore
it after a process restart. A matching canonical reference image shall be
restored only when camera identity, resolution, orientation, and logical
position all match. Invalid persisted data shall be rejected rather than
partially loaded.

### 4.4 Startup verification and Offline readiness

**Status:** Implemented. This section supersedes conflicting startup behavior
described elsewhere in this document and in the related specifications.

#### 4.4.1 Orientation and persisted state

The logical UI board has rank 8 at the top, rank 1 at the bottom, file a at
the left, and file h at the right. Physical rows and columns are the indices
of the calibrated camera grid, not absolute directions in the room or on the
monitor. The complete board orientation is the number of clockwise
quarter-turns mapping a logical UI square to a physical grid square:

| Quarter-turns | Rotation | Logical h1 maps to physical `(row, column)` |
| --- | --- | --- |
| 0 | 0 degrees | `(7, 7)` |
| 1 | 90 degrees clockwise | `(7, 0)` |
| 2 | 180 degrees | `(0, 0)` |
| 3 | 270 degrees clockwise | `(0, 7)` |

This value is relative to the camera-grid indexing; it is not itself the
camera's angle in the room. Camera-grid redetection or changed calibration may
change the required value. Square-color parity determines only whether the
value is even or odd; identifying the White side resolves the remaining
180-degree ambiguity.

The persisted piece map uses logical UI coordinates. The complete orientation,
Offline legal FEN, Offline clock values, and canonical move reference are
stored separately but must describe a compatible saved session. The FEN
contains chess state such as side to move and castling rights; occupancy alone
cannot reconstruct that state. A saved reference is reusable only when its
camera identity, resolution, complete orientation, and logical position match.
Restored settings are provisional until the startup checks below succeed.

#### 4.4.2 Ordered startup checks

1. **Check calibration.** Check that the selected camera and projector are
   available, that saved calibration belongs to the active hardware and
   resolutions, and that live camera geometry and projector mapping are
   sufficiently verified for use. A matching saved record alone is not proof
   that the equipment or board has not moved. Keep projection black except
   for an authorized calibration or verification pattern. If a necessary
   check cannot be completed, or either calibration is invalid or unavailable,
   report the specific condition non-modally and do not enter startup Offline
   readiness.

   **Startup.** Whenever the camera starts or the selected projector
   changes, and a projector display is selected, the application loads the
   saved calibration and verifies it live as described below. It never
   starts a calibration on its own. Calibration and the focus pattern are
   available only from the Calibration menu, as described in
   `adaptive-board-cropped-calibration-specification.md`. Without a usable
   saved calibration, the application stays uncalibrated, reports "Not
   calibrated. Use Calibration > Full calibration." in the projector status,
   and keeps board features disabled.

   **Device loss.** When the active camera disappears or reports an error,
   any running calibration is cancelled and the projector turns black. When
   the camera disappears, the calibration is also deactivated, so no clocks,
   markers, or indicators are projected; it is loaded again when the camera
   returns. When the selected projector display disappears, the projector
   window is closed instead of moving to another screen, any running
   calibration is cancelled, and the calibration is deactivated. When that
   display returns, the projector window reopens black and the saved
   calibration is loaded again. The losses are reported in the status line
   without a dialog.

   The saved calibration is loaded when its projector display
   name and resolution match the selected projector. Camera-only operation is
   not supported. A loaded saved calibration keeps occupancy, startup checks,
   and move tracking paused until the alignment monitor reports a live
   camera-grid check with the projector black. A fresh explicit calibration
   counts as live verified. Load failures and hardware mismatches are shown
   in the projector status and runtime diagnostics, never in a dialog.

   **Automatic alignment recovery.** A valid projector calibration lets the
   application recover the camera grid and the board position without a full
   calibration:

   - When the saved camera identity and resolution also match and the stable
     live camera grid is within 1.5 camera pixels of the saved grid, the
     saved calibration is accepted directly.
   - When the live grid differs by more than that, or the saved record
     belongs to a different camera identity or resolution, the application
     shall not assume what moved. With the projector black it captures a
     black reference, then projects a verification pattern cropped to the
     saved projector board grid, followed by its inverse. The pattern lights
     a fixed set of 32 of the 64 board squares. Unlike a plain checkerboard,
     the set is asymmetric: under every other board rotation or reflection,
     and under every one-square shift, it agrees with itself on only about
     half the squares (absolute correlation at most 0.1). A replaced camera's
     grid indexing can therefore be identified.
   - Each pattern is measured from the first camera frame whose capture
     started at least 0.5 seconds after the projector confirmed that pattern
     is displayed; earlier frames are ignored. If a pattern is not displayed
     within 3 seconds of its request, recovery fails.
   - The pattern frame `P`, inverse frame `Q`, and black reference `B` give
     the normalized response `(P - Q) / (P + Q - 2B + 20)`. It is near +1
     where the pattern lit the surface, near -1 where the inverse did, and
     near 0 where no projected light falls, regardless of whether the surface
     is a light or dark board square.
   - Each hypothesis predicts a camera grid. For every board square that the
     saved piece map marks as empty (all squares when fewer than 16 are
     empty), the median response is sampled over the central 40% of the
     predicted square and multiplied by +1 or -1 according to the pattern.
     The hypothesis score is the mean of these values.
     - **Camera moved or replaced** (projector and board fixed): the
       pattern stays on the same physical squares, so it is predicted on the
       newly detected camera grid. Because a replaced camera may index the
       grid differently, all eight grid orderings are tested. Adoption keeps
       the projector grid unchanged and refits the camera-to-projector
       homography to the new camera grid.
     - **Board moved** (camera and projector fixed): the pattern stays where
       the board used to be, so it is predicted on the saved camera grid.
       Adoption uses the existing board remap of the projector grid. This
       hypothesis is not tested after a camera hardware change.
   - The best hypothesis is selected when its score is at least 0.4 and
     every rival whose predicted grid differs by more than a quarter square
     scores at least 0.2 lower. Rivals within a quarter square predict nearly
     the same mapping and do not make the result ambiguous. If three pattern
     pairs yield no selectable hypothesis, recovery fails.
   - The selected hypothesis is only a coarse explanation; when the camera
     and board both moved (or the projector moved), neither explanation is
     exact. The remaining offset is therefore measured, not assumed:
     **Offset measurement (A).** A measurement pattern and its inverse are
     projected through the hypothesis mapping. Its 7 x 7 cells are shifted by
     half a square, so every pattern edge crosses the middle of a uniformly
     coloured board square: optical blur pushes an edge that coincides with
     a light/dark board edge toward the darker square, which biased a
     square-aligned measurement by up to 8% of a square in simulation. A
     +1/-1 template of the cells predicted on the live board grid is
     registered onto the normalized, saturated response by
     enhanced-correlation (ECC) affine alignment, restricted to the eroded
     interior of cells lying entirely on scored squares (all cells when
     fewer than eight qualify). Patterns and templates are rendered with
     exact supersampled pixel coverage; OpenCV's anti-aliased polygon fill
     fattens squares asymmetrically. The largest displacement of a scored
     square corner is the residual offset.
     - A residual of at most 5% of a square adopts the hypothesis mapping.
     - Otherwise the measured warp corrects the hypothesis mapping:
       the projector point of each board grid point is taken where the
       hypothesis would place the registered camera point. If the
       registration does not converge, the uncorrected hypothesis mapping is
       used as the starting point.
   - **Fringe refinement (B).** Starting from that corrected mapping, the
     application projects, cropped to the board, four phase-shifted cosine
     fringes along the board columns and four along the board rows, each
     with a period of one square. For every scored square, 7 x 7 camera
     samples over its central 60% decode the phase `atan2(I3 - I1, I0 - I2)`,
     which is independent of the square's reflectance. The phase is
     unwrapped against the corrected mapping's prediction and gives the
     exact projector pixel seen at that board position. Samples with fringe
     modulation below 4 grey levels or below 15% of the projected brightness
     (pieces, shadows, glare) are rejected, and a square needs at least half
     of its samples. At least eight squares spanning at least three rows and
     three columns are required. A robust board-to-projector homography
     (RANSAC, threshold 10% of a projector square) is fitted, each projector
     grid point is corrected by the median residual of inlier samples within
     1.5 squares, and the median fit error must be at most 8% of a square.
     If refinement fails these checks, the corrected mapping from A is kept.
   - **Final check.** The measurement pattern and its inverse are projected
     again through the new projector grid and must appear on the live board
     grid with a cell score of at least 0.4 and a measured offset of at most
     10% of a square; otherwise recovery fails. The camera and board
     may therefore both have moved, and the projector too, as long as the
     live board grid is detected and enough squares are empty.
   - An adopted result is activated as live verified and saved as the new
     calibration record for the active camera and projector, after the
     previous record is copied to
     `calibrations/before-recovery-<timestamp>.json`. The camera calibration
     indicator keeps showing which hypothesis was recovered, or that the
     alignment was re-measured. The projector returns to black and startup
     continues with occupancy.
   - A failure marks the calibration unusable for this run, keeps the saved
     record, and tells the user non-modally to run a calibration explicitly.
   - Every step is written to the runtime diagnostics event file
     `diagnostics/runtime-events.jsonl` with camera images: the recovery
     start (reason, displacement, number of scored squares), the measurement
     (black, pattern, inverse, normalized-response, measurement-response,
     first column and row fringe, and final-check response images, display delays, every
     hypothesis's score and grid center, the selected hypothesis, the
     measured residual offset, the refinement sample counts, fit error and
     change, the final-check score and offset, and the last rejection
     reason), the backup file name, and the adoption or failure.
   - Recovery runs only for startup verification of a loaded saved
     calibration. During play, board movement is still remapped as a board
     movement and verified by the projected-square alignment checks.

   *Implementation status:* implemented in `AlignmentMonitor`
   (`recovery_pattern_image`, `score_recovery_hypotheses`,
   `select_recovery_hypothesis`, `measure_pattern_offset`,
   `pattern_offset_fraction`, `offset_pattern_score`,
   `corrected_projector_grid`, `recovery_fringe_image`,
   `refine_projector_grid`, `camera_moved_mapping`) and `CameraWindow`
   (`_restore_saved_calibration`, recovery signal handlers), with
   `CalibrationStore.load_for_projector` reporting whether the camera also
   matches and `CalibrationStore.backup` preserving the replaced record.
   History: the first implementation projected one square and measured it
   near each hypothesis's own expected point using the first frame after the
   request. In a real run the square was not yet visible, camera noise
   "confirmed" the wrong hypothesis, and the bad mapping overwrote the saved
   record. A second implementation measured one light centroid after a
   settling delay; on a board moved by about 1 cm the square straddled a light
   and a dark board square, the reflectance difference biased the centroid
   by 17% of a square, and the correct hypothesis was rejected. The
   normalized, whole-board pattern comparison replaces both. A third run with
   camera and board both moved selected "camera moved" with score 0.78 but
   left a 13% offset, because hypothesis selection alone cannot measure a
   combined movement; offset measurement, fringe refinement, and the final
   check address this. Tests:
   `tests/test_alignment_monitor.py` (pattern asymmetry, hypothesis
   selection with board reflectance, frames before a pattern is visible are
   ignored, camera change with different grid indexing, combined camera and
   board movement refined to the exact projector grid with and without
   occupied squares, failed fringe refinement falling back to the measured
   correction, a re-measured alignment failing the final check, missing pattern fails after three attempts, display
   timeout, small drift without recovery) and `tests/test_projector_calibration.py` (store matching and
   backup, occupied-square exclusion, dialog-free failure). Known
   limitations: the coarse step must still select a hypothesis, so a
   movement of more than about a quarter square relative to both
   explanations requires explicit calibration; with few empty squares (for
   example the starting position) the outer ranks are extrapolated from the
   board plane; and if no camera frame is available when recovery completes,
   the recovered calibration is used but not saved.
2. **Load the saved session provisionally.** Read and validate any saved
   physical piece map, orientation, legal FEN, reference metadata, and Offline
   clocks without overwriting them. Missing, invalid, or mutually inconsistent
   data shall be reported and shall not be treated as a verified saved game.
   The checkerboard's parity can constrain the saved orientation, but cannot
   alone prove which side is White.
3. **Observe occupancy.** With the projector black, discard exposure
   transitions and wait for a stable, quiet, unobstructed observation. Use the
   confirmed camera-grid parity to select orientation-aware empty-square
   models. An inconclusive observation, hand, motion, or calibration loss shall
   wait or report the problem; it shall not modify the saved game or reference.
4. **Evaluate the starting layout first.** If the stable occupancy matches
   all 32 standard starting squares under the geometric parity, determine
   which side is White from piece-color evidence and establish the complete
   orientation. With automatic starting-position reset enabled, atomically
   accept the standard physical and legal position, capture a clear black-frame
   reference, reset both Offline clocks to their configured initial values,
   persist the new session, and enter Offline play with White to move and
   clocks paused. If White-side evidence is inconclusive, do not reset or
   project moves; continue waiting and report the reason. When automatic reset
   is disabled, leave the session unchanged until the user explicitly requests
   a reset.
5. **Otherwise compare with the saved position.** Map the stable physical
   occupancy through the validated saved orientation and compare it with the
   occupied squares of the saved piece map. When the saved Offline FEN and
   physical piece map agree, their orientation is compatible with current
   geometry, and the physical occupancy matches, restore the saved Offline
   game and its saved clock values paused. Verify or establish a compatible
   clean reference before enabling move tracking; never adopt an obstructed
   or projector-contaminated frame. The implementation performs this
   comparison through physical reconciliation against the saved piece map,
   which also rejects confident contradictory piece-color evidence when a
   compatible saved reference is available.
6. **Handle mismatch without changing authority.** If no valid saved session
   matches, keep the previous saved position, FEN, clocks, orientation, and
   reference unchanged. Report the mismatch non-modally and wait for the user
   to arrange the board or explicitly choose a recovery action. Do not start
   a new game, infer or submit a move, or mark an inferred illegal move.
   Mismatching squares may be listed in the UI and briefly projected with the
   existing red reconciliation markers. Arranging the standard start (with
   automatic reset enabled) ends the pending check under step 4.

The starting-layout check intentionally takes precedence over a saved game
when automatic reset is enabled. Occupancy cannot distinguish same-color piece
types, swapped pieces, or every different position with identical occupied
squares. Consequently the automatic starting reset assumes the user has
physically arranged the correct pieces, and may replace a saved game even if a
different physical position happens to have the same occupancy. A saved-game
occupancy match likewise confirms occupied squares, not the identity of every
piece. Where available, confident contradictory color evidence shall prevent
adoption rather than be ignored.

This startup Offline readiness sequence requires both verified camera and
projector calibration. It does not alter Lichess game authority and
reconciliation; an active Lichess game cancels any pending startup Offline
check, and the sequence shall not interrupt an active Lichess game or
automatically reset it.

#### 4.4.3 Deferred single-move recovery

Recognizing one physical move made while the application was stopped and
marking it as an erroneous move is deferred. A normal relocation generally
changes two occupancy squares, not one; captures and special moves may have
different or ambiguous occupancy signatures. A future design must use
sufficient evidence to distinguish legal and illegal candidates, captures,
special moves, and visually indistinguishable states before offering
correction. Until then, a mismatch shall follow step 6 above.

#### 4.4.4 Acceptance criteria

- Missing or incompatible projector calibration blocks startup Offline
  readiness and reports which check failed without opening a startup dialog
  or automatically projecting a calibration pattern.
- With a valid projector calibration, a moved board, a moved camera, or a
  replaced camera, or a combination of these, is recovered by an asymmetric
  verification pattern and its inverse scored over the empty board squares
  independently of board reflectance, a measured residual offset, fringe
  refinement when that offset exceeds 5% of a square, and a final check
  within 10% of a square; the chosen
  hypothesis is saved, reported in the status text, and recorded in the
  runtime diagnostics file. An unmatched verification keeps the saved record
  and asks for explicit calibration without a dialog.
- After explicit successful calibration, startup proceeds to stable occupancy;
  interruptions or failed calibration do not replace saved game state.
- The four orientation values map UI squares to camera-grid squares as listed,
  and parity alone never establishes the White side.
- At each orientation, a clear standard setup is recognized from live evidence,
  White-side orientation is determined, and automatic reset creates the
  starting game with configured clocks paused.
- A non-starting physical board matching a compatible saved Offline session
  restores its FEN and saved clocks paused without counting application
  downtime or resetting the game.
- A changed, missing, invalid, or incompatible saved session never silently
  overwrites prior state; a non-starting occupancy mismatch remains pending.
- Motion, obstruction, insufficient piece-color evidence, projected light,
  and incompatible reference metadata prevent unsafe acceptance.
- Starting-layout precedence and the limits of occupancy-only identity
  checking remain explicit to the user.
- No single-move correction is inferred as part of this first implementation.

## 5. UI Rendering

Both the Physical position and Lichess position boards shall expose an
extensible right-click context menu with:

- **Copy FEN**: copy the complete current FEN to the system clipboard. The
  Physical board shall serialize its displayed pieces with White as the
  default side to move, castling rights for every king and corresponding rook
  still on their home squares, one en-passant target when the placement makes
  exactly one target physically plausible for that side, and default move
  counters (`0 1`). The Lichess board shall use the authoritative server FEN
  when connected and otherwise infer the same metadata from its displayed
  pieces.
- **Copy Board**: copy the square rendered board, including its current pieces
  and visible overlays but excluding layout margins, as an image.
- The Lichess board shall additionally expose **Current Lichess game...**.
  The action shall be enabled when current or most recently finished game
  metadata is available and shall open a selectable, read-only dialog showing
  the game ID, players, clocks, game status/result, and physical
  synchronization status.
- **Flip board**: a checkable action on both boards that shows the boards
  from Black's side. One shared orientation applies to the Physical and
  Lichess boards; toggling it on either board flips both.

The menu shall remain available on the display-only Lichess board; enabling
the context menu shall not make its squares selectable.

### 5.0 Board orientation

- Flipping is a view transform only. Positions, moves, highlights, guidance,
  and clicks keep UI coordinates (row 0 = rank 8); the widget draws UI square
  `(row, column)` at view square `(7 - row, 7 - column)` when flipped and maps
  clicks back before emitting them. The physical-board rotation, projector
  output, move detection, and Lichess synchronization are unaffected.
- When flipped, rank labels run 1-8 from top to bottom along the right edge
  and file labels run h-a along the bottom edge. Pieces stay upright. Copy
  Board copies the board as displayed; Copy FEN is unchanged.
- The Lichess panel shows the player row of the side at the top of the board
  above it: Black above and White below when not flipped, White above and
  Black below when flipped.
- The orientation is stored in the application settings and restored at
  startup.
- When a Lichess game becomes the attached game, set the orientation to the
  player's color (flipped for Black). A manual flip afterwards persists until
  another game is attached.
- The Current Lichess game picker preview shows each game from the player's
  color.
- Implemented in `ChessboardWidget` (`set_flipped`, view-to-UI square
  mapping) and `CameraWindow._set_board_flipped` /
  `_arrange_lichess_player_rows`. Covered by
  `tests/test_chessboard_widget.py` (flipped drawing and clicks, linked flip
  with row swap and persistence, picker preview) and
  `tests/test_projector_calibration.py` (automatic flip once per newly
  attached game).

### 5.1 Rendering order

For each square, the widget shall draw:

1. Square background.
2. Persistent last-move source and destination lightening.
3. Selected-square fill and border.
4. Piece SVG.
5. File and rank labels.

Coordinate labels and the selected-square border shall remain visible.

After an accepted move, both its source and destination squares shall remain
visibly lighter on the UI board until another move is accepted. An explicit
reset to the standard starting position shall clear the highlight.

In Offline legal mode, accepting a move shall illuminate its source and
destination on the projector for exactly one second. The optional next-turn
dot may appear concurrently. The projector shall then remove the move
illumination, return to the configured idle board illumination, and allow one
fixed second for camera exposure to settle before verification resumes. The
automatic starting-position detector shall discard every observation gathered
before the accepted move; stale starting-position evidence must not reset the
newly accepted logical position. The current side-to-move dot shall remain
steady during red illegal-move correction and every other temporary gameplay
projection mode.

If that next manipulation produces a clear relocation which does not match
any legal physical move signature, including a move by the wrong side, the
accepted position and reference shall remain unchanged and both observed
source and destination squares shall be projected red.

### 5.2 Size and alignment

Each piece shall:

- Be centered in its square.
- Preserve the SVG aspect ratio.
- Fit within approximately 88% to 94% of the square size.
- Scale smoothly with widget resizing.
- Use transparent SVG background.

### 5.3 Cache

Rendered images may be cached by:

- Piece code.
- Target pixel size.
- Device pixel ratio.

The cache shall be invalidated when the effective rendering size changes.

### 5.4 Occupancy dots

When a logical position is synchronized:

- Piece artwork replaces black occupancy dots for known occupied squares.
- A dot shall not be rendered underneath a known piece.

When occupancy detects a square for which the logical position has no known
piece:

- The normal UI shall report that synchronization is lost.
- A diagnostic marker may identify the unmatched occupied square.
- The application shall not invent a piece identity.

## 6. Physical-to-Logical Mapping

Camera occupancy results use physical board coordinates. The position model
uses UI coordinates.

Every confirmed occupancy set shall be converted through the current complete
quarter-turn orientation before movement comparison.

The White-side orientation established from the starting-position piece colors
shall be retained until:

- Calibration is invalidated.
- The board is deliberately reoriented.
- The user explicitly starts a new setup.

Activating a newly completed or loaded calibration shall always invalidate the
previous physical-grid orientation, even when the logical piece position is
already initialized. Before square projection or occupancy resumes, the
application shall re-detect square-color parity from the current camera frame
so that logical light squares map to physical light squares. When a complete
or sufficiently populated starting-side arrangement is subsequently visible,
the application shall also re-evaluate the 180-degree White-side ambiguity
without clearing or replacing the initialized logical position. Parity alone
shall remain transient and shall not overwrite the persisted complete
quarter-turn orientation. Square projection shall remain unavailable while the
White-side ambiguity is unresolved.

When parity detection agrees with the persisted orientation modulo 180
degrees, the application shall retain the persisted complete orientation
rather than collapsing it to the parity detector's `0` or `1` representative.
This preserves the known White-side direction during midgame restart, when the
starting-rank piece evidence needed to resolve the 180-degree ambiguity is no
longer available.

That retained direction is provisional when the restored logical position is
the standard starting position. A stable physical starting arrangement shall
always re-run piece-color orientation before automatic starting-position
acceptance, even when parity agrees with the saved orientation. The measured
White side shall replace the saved 180-degree direction and be persisted.
Parity agreement alone shall never mark White-side orientation complete for a
visible standard starting position.

The projection restriction applies to every physical projection entry point,
including Offline accepted-move marking and pending Lichess guidance. Logical
guidance may remain visible in the application while physical projection is
withheld. A move withheld for unresolved orientation shall remain eligible for
projection after orientation resolves; it shall not be recorded as already
projected.

Changing the complete quarter-turn orientation or the active projector mapping
shall invalidate any cached pending-move projection. If a Lichess move remains
pending, the application shall render it again using the new logical-to-
physical mapping. A cached UCI move shall not prevent this refresh.

## 7. Move Tracking

Post-initialization move inference is defined exclusively by
[Reference-image move tracking](./reference-image-move-tracking-specification.md).

Absolute 64-square occupancy shall initialize and verify the starting setup,
but shall never mutate an initialized logical position. The reference tracker
shall:

- Detect activity and wait until the board is visibly clear.
- Compare a settled image with the last accepted canonical reference.
- Rank per-square changes.
- Require one dominant known source and one dominant empty destination.
- Support non-legal one-piece relocations.
- Reject ambiguous or multi-piece changes without replacing the reference.
- Update the logical position and reference atomically.

Captures, castling, en passant, and promotion remain outside the first
reference-tracking milestone.

Offline legal mode shall support a completed ordinary capture when one legal
capture signature uniquely matches the settled image. After legal application,
the piece synchronizer shall replace its logical position with the complete
authoritative post-move position returned by the rules engine rather than
requiring the destination to have been empty. The logical position, accepted
reference, persisted FEN, UI pieces, and last-move indication shall commit
together. Any failure before reference commit shall roll back the legal game
and preserve the previous logical position and reference.

For an authoritative legal signature, a vacated source may have weaker visual
change evidence than its destination because removing a piece can reveal a
similar-colored square. The source shall use a lower evidence floor while every
other expected changed square retains the normal movement threshold. The
signature must remain unique, and the existing unexpected-change, obstruction,
released-piece, and consecutive-observation safeguards shall still pass.
Board setup / diagnostic mode shall continue to require the normal threshold
for both its source and destination.

For a capture, destination occupancy remains occupied. The visual transition
is the destination changing from the captured piece's color appearance to the
mover's color appearance. The legal signature supplies both before and after
piece colors. When the source is visibly vacated and current color
classification confidently identifies the destination as the mover's color,
the destination may use a lower change floor than an ordinary move because
similarly shaped pieces can produce little pixel change on an occupied square.
The destination must still exceed the capture noise floor, the complete
signature must remain unique, and source/destination color state must match the
authoritative capture result. A source-only lift with the captured piece still
present must remain an illegal removal. Regression cases include `a3xb4`,
`Nb4xd3`, and the low-contrast `Rd1xh1` replacement of one rook by another.

For a two-action capture, removing only the captured destination piece shall
display pending-capture guidance naming the empty destination and every
eligible attacker source. The UI pieces, FEN, turn, and accepted reference
shall remain unchanged. Moving an eligible attacker to the destination shall
complete the normal authoritative capture commit against the original
reference. Restoring the captured piece instead shall return to the accepted
position without requiring a reset.

When the only settled change is removal of a piece and the removal is not the
captured-piece-first stage of a pseudo-legal capture, the removed square shall
be treated as an illegal physical removal. That square shall be marked red on
the UI and projected red, and the persistent correction text shall require
restoring the piece. A side-to-move piece lifted without being placed on a
destination after the hand is withdrawn uses the same illegal-removal
correction.

Board setup / diagnostic mode shall retain its one-source/one-empty-destination
contract; an occupied destination without legal-game authority remains
ambiguous.

The application provides two local movement policies:

- `Board setup / diagnostic` retains legality-independent one-piece relocation
  for position setup and camera testing.
- `Offline legal game` validates a visual candidate against an authoritative
  `python-chess` position before changing the logical position or reference.

After an illegal relocation, the accepted position and reference remain
unchanged. The user may move the physical piece directly from the rejected
square to a legal destination; returning it to its original square first is
not required. The authoritative source piece remains at its logical square and
that square is marked red on the UI. When a calibrated projector is available,
both the authoritative source square and the physically observed illegal
destination shall be projected in red together without a fixed timeout. Both
red squares shall remain projected while the player reaches onto the board and
restores or redirects the piece. Only after confirmed hand activity has become
quiet shall the projector return to black so camera verification can resume.
If verification shows that the same rejected position remains, both red
squares shall be projected again. The UI mark clears when the board returns to
the accepted reference or a legal move is accepted. Restoration uses a
separate conservative tolerance: every square except the authoritative source
must be below the normal movement threshold, while the source may retain a
small lighting residual. This tolerance does not change ordinary move
detection.

Physical reconciliation may mark multiple squares red simultaneously. Missing
expected pieces, unexpected pieces, and confidently opposite-color pieces
shall remain marked until a stable observation matches the reconciliation
target. These reconciliation marks are independent of the single-square
illegal-move marker and clear atomically when the authoritative position is
adopted.

An otherwise clear illegal relocation shall still reach this correction
workflow when unrelated settled differences are classified as board-edge
camera-remap residuals. Such residuals shall not suppress the red authoritative
source marker. Genuine interior obstruction continues to prevent move
evaluation.

Offline legal mode shall also detect a completed pseudo-legal capture,
promotion, or capture-promotion that is illegal only because it fails to resolve
check or exposes the moving king. The physical signature preserves the occupied destination transition needed to localize that
action, but only the authoritative legal-move check may commit it. A rejected
signature shall use the same red source marker and red projector correction on
both involved squares as any other illegal relocation.

While correction is required, persistent UI guidance shall identify the
observed incorrect square, the authoritative source square, the side to move,
and the source piece's legal destinations. If that piece has no legal move for
the current side, the guidance shall explicitly require returning it to its
authoritative source before making a move with the correct side. Camera
restabilization and alignment messages shall not replace this guidance. After
restoration, the guidance shall confirm that the board matches the accepted
position and state the side to move.

User-facing tracking guidance shall remain readable. Short-lived internal
`activity`, `settling`, and `candidate` phases shall not replace the last
actionable message. Pending captures, obstructions, ambiguous board changes,
illegal corrections, accepted moves, and restored-board confirmation shall
remain visible until another durable or actionable state replaces them.

The complete offline FEN and the last successfully activated mode are
persisted. A matching accepted physical position shall restore Offline legal
mode automatically. A mismatch shall preserve the saved offline FEN and mode
until the user explicitly chooses one of the recovery outcomes below; mismatch
detection alone shall not write setup/diagnostic mode into persistent settings.
Visual signatures for captures and special moves are specified as later
milestones in the
[Lichess physical-board integration roadmap](./lichess-integration-roadmap.md).

Offline legal mode is the normal local gameplay mode. Board setup / diagnostic
mode exists for arbitrary position setup, camera diagnosis, and recovery
without legal turn enforcement. It shall not silently replace an active or
persisted offline game.

When the saved legal and accepted physical piece maps differ, the application
shall first run live starting-position verification with the projector black.
If all pieces occupy the standard starting squares, it shall automatically
reset legal, physical, reference, and UI state together, activate Offline
legal mode, persist that state, and be ready for White's first move. If the
verified occupancy is not the standard starting position, the application
shall show a modal recovery choice:

Successful explicit or automatic reset shall replace the board widget's
logical pieces and camera-occupancy overlay atomically with the verified
standard squares. Occupancy-only dots from the preceding physical position
shall not remain visible after reset.

When automatic starting-position reset is enabled, the same stable occupancy
check shall run passively while the projector is black and no Lichess game is
active, including in Board setup / diagnostic mode. Physically arranging the
standard setup shall therefore trigger reset and Offline legal readiness
without requiring a mode-selection attempt. This passive start-only detector
may use a lower-resolution canonical image and shorter observation window than
general occupancy tracking because it accepts only the complete standard
32-square pattern. It shall still require two agreeing classified observations
after motion settling, rebuild its square-appearance model after board motion
so a model learned from a midgame position cannot prevent recognition of a
newly arranged starting position, report provisional recognition immediately,
and target completion within one second after the board becomes clear.

- **Restore saved offline position** keeps the saved FEN authoritative and
  enters persistent correction guidance until the camera confirms that
  position.
- **Reset to starting position** verifies the standard physical setup, resets
  the legal game and accepted physical/reference state together, activates
  Offline legal mode, and persists the resulting state.
- **Start Offline game from current position** makes the accepted physical
  position the Offline legal position without rearranging the board. The side
  to move remains undetermined until the first accepted move. During that
  pending state, the application shall recognize legal moves for either side,
  hide both turn indicators, and retain the pending state across restart.
  The first accepted move selects its source piece's color as the side to move
  and advances the resulting legal game. An illegal first move shall be
  rejected with ordinary red correction guidance and shall leave the turn
  undetermined. The standard starting position is the exception: it starts
  immediately with White to move.

Selecting **Start Offline game from current position** shall not immediately
enable move tracking against a restored reference. The application shall
black the projector, discard at least one second of camera exposure
transition, wait for a stable quiet observation window, and capture a fresh
move reference for the accepted current position. The previous appearance
reference shall not participate in this capture because it may be stale or
projector-contaminated. Turn inference and ordinary Offline move tracking
shall begin only after the clean reference is installed.

The same clean-reference requirement applies when a matching saved Offline
game is restored during application startup. After live calibration
verification, startup shall first compare stable physical occupancy with the
saved piece map, as specified in section 4.4. Only a matching observation,
taken after projector blackout and exposure settling, replaces and persists
the appearance reference. Move tracking shall remain paused until then. If
occupancy detection is unreliable for a visibly correct board, the user may
explicitly choose **Stop reconciliation** to accept the current board and
capture a clean reference; this is never done automatically.

For an Offline game started from the current position, the provisional legal
position shall grant castling rights for each king and corresponding rook
occupying their home squares. For each candidate side to move, it shall retain
an en-passant target only when the placement identifies exactly one physically
plausible preceding two-square pawn move that can now be captured; ambiguous
or absent candidates shall use no target. Once the first move is accepted,
ordinary authoritative move history controls both rights.

Cancelling this choice shall preserve both the current active mode and all
persisted mode, FEN, physical-position, and reference data.

Offline legal mode shall display a persistent turn indicator beside the move
rules selector:

- White fill with a dark border means White to move.
- Black fill with a dark border means Black to move.
- Gray means the authoritative legal game is complete.
- The indicator is hidden in Board setup / diagnostic mode.
- The indicator is also hidden while an Offline game is waiting for its first
  move to determine the side to move.

The indicator is driven only by the authoritative legal position. Camera,
alignment, occupancy, and correction messages shall not change it. Its tooltip
and accessible name shall state the same turn or game-over information.

When the persistent physical turn-indicator setting is enabled, an active game
shall also project one white dot on the logical a-file side beside the clocks:
half a square beyond a1 for White's turn and half a square beyond a8 for
Black's turn. The dot's location, rather than its color, identifies White or
Black. Calibrated board orientation shall rotate the logical corners to the
physical board. The dot shall be defined as a circle in calibrated board-plane
coordinates and pre-warped into projector pixels so projector perspective does
not turn it into a physical ellipse. It shall remain outside the 8x8
camera-evidence area and remain steadily visible through accepted-move
illumination, move guidance, illegal correction, reconciliation, promotion,
blackout, settling, and other temporary gameplay projection states. These
states shall not blink or temporarily suppress the dot. The application may
delay changing the dot from one side to the other until the newly authoritative
turn is ready for presentation, but once changed the new dot shall remain
steady. Calibration patterns and calibration blackout may suppress the dot.
Board setup / diagnostic mode, completed games, and an Offline game waiting
for first-move turn inference shall project no turn dot.

## 14. Recovery and Reset

When automatic starting-position reset is disabled, the UI shall provide an
explicit action equivalent to:

`Reset pieces to starting position`

When automatic starting-position reset is enabled, this Camera-menu action
shall be hidden.

The action shall:

- Immediately make the projector output black and mark black as the
  authoritative projector state before live occupancy verification begins.
- Enter a dedicated live absolute-occupancy verification mode.
- Preserve the current logical position and reference while verification is
  pending.
- Require a stable physical starting-position occupancy.
- Re-run White-side orientation.
- Recreate the standard logical position.
- Capture a new canonical reference image.
- Leave the previous logical position and reference unchanged when
  verification fails.

This explicit action is the only way to discard an initialized logical
position other than starting a new application session. Automatic occupancy,
calibration, camera, or remapping events shall not discard it.

A future setup editor may support arbitrary positions. The first implementation
shall not infer arbitrary piece identities from occupancy dots.

## 15. Interaction with Projector Selection

Clicking a UI square shall continue to project the corresponding physical
square. The projection and its selected-square UI marker shall remain active
for one second, then automatically return to authoritative black and clear the
temporary marker. A timeout belonging to an older selection shall not cancel
a newer projected square.

Any action that renders black projector output, including **Open projector
(black)** and **Black**, shall also mark black as the authoritative projector
state used by the occupancy gate. The displayed projector status and the
occupancy gate shall not disagree about whether the projector is black.
The **Black** action shall open or raise the selected projector window before
rendering black, so a closed, obscured, or screen-saver-covered output cannot
remain visible.
Rendering black shall clear the active selected-square marker from the
physical-position UI while retaining the remembered square internally for
later reuse. It shall not clear the accepted last-move highlighting.

Piece rendering shall not:

- Change click hit testing.
- Change logical-to-physical orientation mapping.
- Trigger a projector update by itself.
- Prevent the selected-square highlight from being shown.

When an automatically moved piece changes UI squares, the remembered selected
square shall remain unchanged.

## 16. Diagnostics

For each candidate move, diagnostics shall include:

- Per-square reference-image change scores.
- Source piece code.
- Candidate source and destination names.
- Calibration, orientation, and reference identity.
- Reference-tracker state before and after evaluation.
- Acceptance or rejection reason.

Asset diagnostics shall include:

- Expected and loaded filenames.
- Pinned upstream revision.
- Recorded SHA-256 checksums.
- SVG load failures.

## 17. Tests

### 17.1 Asset tests

- All twelve files exist.
- Every file is valid SVG.
- Every file can be rendered by Qt.
- Checksums match the recorded manifest.
- Attribution and license notice exist.

### 17.2 Position tests

- Standard position contains 32 pieces.
- Every expected piece occupies the correct square.
- White is at the bottom.
- All four physical board rotations map to the same logical position.

### 17.3 Rendering tests

- Every piece type renders.
- Piece transparency preserves the square background.
- Resizing preserves aspect ratio.
- Coordinate labels and selection remain visible.
- Clicking a piece-occupied square emits the same square as before.

### 17.4 Movement tests

- `e2-e4`.
- `g1-f3`.
- Black move after a White move.
- Lifted-piece transient state is ignored.
- One source and one destination update atomically.
- Deliberately non-legal one-piece relocation is accepted.
- Multiple removals are rejected.
- Multiple additions are rejected.
- Source-only capture evidence is rejected.
- Large stationary obstructions are rejected.
- Thin grid-edge residuals after remapping are not treated as obstructions.
- Camera glitches preserve the logical position and reference.
- Occupancy resets and calibration interruptions preserve the logical position.
- Reference metadata mismatches prevent restoration.
- Explicit reset uses live occupancy rather than a cached pre-reference result.
- Explicit reset automatically returns the projector to black before occupancy
  verification.
- With automatic reset enabled and no active Lichess game, stable standard
  occupancy triggers the same reset from Board setup / diagnostic mode.
- Disabling automatic reset restores the explicit Camera-menu action.
- Opening black projector output resumes occupancy processing without requiring
  a separate **Black** action.
- Failed reset verification preserves the logical position and reference.
- Offline legal mode rejects moves that are illegal for the authoritative FEN.
- A pseudo-legal capture-promotion that does not resolve check is localized and
  marked illegal.
- Rejection preserves both the accepted position and reference.
- Restoring a rejected move clears its UI marker despite a small residual on
  the authoritative source square.
- The persistent turn indicator follows the authoritative side to move and is
  unaffected by transient camera status.
- A wide or tall released king is not mistaken for a hand.
- An opposite-color adjacent piece whose base signature remains unchanged can
  explain a third-square silhouette residual.
- The same-color equivalent remains ambiguous.

## 18. Acceptance Criteria

The piece-synchronization workflow is accepted when:

1. The UI displays the official cburnett SVG pieces in the standard position.
2. White appears at the bottom after camera-derived side detection.
3. An ordinary physical move updates exactly one UI piece, while supported
   special moves update their complete legal signature atomically.
4. The UI update occurs only after a clear, settled reference comparison.
5. Hand motion does not create intermediate UI moves.
6. Ambiguous physical changes preserve the previous logical position.
7. Detector resets and transient camera or calibration failures preserve the
   initialized logical position.
8. Square clicking and projector indication still work.
9. Operation requires no internet connection.
10. Asset provenance, checksums, attribution, and GPLv2-or-later licensing are
   recorded.

## 19. Implementation Milestones

The twelve pinned SVGs, checksum manifest, GPLv2-or-later text, and attribution
notice are vendored with the application. The standard position model, Qt SVG
rendering, reference-image one-source/one-destination synchronization,
ambiguity handling, and explicit and automatic starting-position reset are
implemented.
Offline legal authority, persistent turn indication, illegal-move correction,
targeted opposite-color residual resolution, legal physical signatures, and
single-action captures are implemented and have passed live physical-board
acceptance. Reference-relative physical piece-color classification has also
passed live acceptance for white and black captures and wrong-color rejection;
arbitrary-position setup remains future work.

Physical-board reconciliation supports both manual restart and automatic
Lichess attachment. Without an active Lichess target, stable absolute
occupancy is compared with the accepted legal position and may replace only
the accepted camera reference. With an active Lichess target, stable occupancy
is compared with the authoritative current server position. Missing expected
pieces, unexpected occupied squares, and confidently opposite-color pieces are
marked red on both the UI and projector. Uncertain colors do not block
adoption. When no discrepancy remains, reconciliation may replace the stale
logical piece map and legal state with the authoritative Lichess position,
capture a new accepted camera reference, and initialize synchronization at the
current server cursor. Same-color piece identity differences are intentionally
accepted because occupancy and color evidence cannot distinguish piece types.

### Milestone 1: Asset import

- Download the twelve SVG files from the pinned Lichess revision.
- Record checksums.
- Add attribution and license information.
- Validate Qt SVG rendering.

Implemented.

### Milestone 2: Static starting position

- Add a typed piece and position model.
- Initialize the standard position.
- Render all pieces in the chessboard widget.
- Replace occupied dots with known piece artwork.

Implemented.

### Milestone 3: Reference-image move tracking

- Store the last accepted canonical image.
- Detect board activity, clearance, and settling.
- Rank changed squares and identify one known source and empty destination.
- Move the known UI piece and commit the new reference atomically.
- Report accepted and rejected changes.

Implemented for ordinary relocations and the multi-square legal signatures
listed in Milestone 6.

### Milestone 4: Recovery

- Add explicit starting-position reset.
- Detect synchronization loss.
- Preserve logical state through temporary motion.
- Restore only metadata-compatible persisted references.
- Verify reset requests asynchronously using the current camera frame.

Implemented. Reset requires confirmed starting-position occupancy. When the
persistent automatic-reset setting is enabled, stable starting occupancy
triggers the same atomic reset and activates Offline legal mode without an
explicit command.

### Milestone 5: Offline legal moves

- Add authoritative FEN, side-to-move, legal-move generation, and game status.
- Reject illegal physical moves without changing logical/reference state.
- Add persistent turn and correction guidance.

Implemented for all currently supported legal physical signatures.

### Milestone 6: Captures and special physical signatures

- Commit completed ordinary legal captures from a unique legal signature and
  the authoritative post-move position.
- Add stronger destination color-transition evidence for captures.
- Support castling, en passant, and promotion workflows.
- Reconcile multi-square physical signatures before atomic commit.

Implemented. Ordinary and captured-piece-first captures, king-first castling,
captured-pawn-first en passant, auto-queen promotion, and projected local
underpromotion selection preserve atomic legal/reference commits. Exact
incoming underpromotion suffixes are retained; dedicated incoming
replacement-piece projection remains staged.

## 20. Open Decisions

- Whether move history is persisted between application starts.
- Whether later arbitrary-position setup uses FEN entry, a graphical editor,
  or piece recognition.
