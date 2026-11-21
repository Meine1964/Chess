# Chess-Piece UI Rendering and Move Synchronization Specification

**Version:** 0.3
**Status:** Implemented ordinary-move and offline legal-game workflow
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

The first milestone supports one ordinary, non-capturing physical piece move
at a time.

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

## 5. UI Rendering

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
piece colors; reference-image evidence must show a material destination change
even though no empty-destination transition occurs. Regression cases include
`a3xb4` and `Nb4xd3`.

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
destination are projected in red for two seconds. The projector then returns
to black so camera verification can resume. The UI mark clears when the board
returns to the accepted reference or a legal move is
accepted. Restoration uses a separate conservative tolerance: every square
except the authoritative source must be below the normal movement threshold,
while the source may retain a small lighting residual. This tolerance does not
change ordinary move detection.

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

When automatic starting-position reset is enabled, the same stable occupancy
check shall run passively while the projector is black and no Lichess game is
active, including in Board setup / diagnostic mode. Physically arranging the
standard setup shall therefore trigger reset and Offline legal readiness
without requiring a mode-selection attempt.

- **Restore saved offline position** keeps the saved FEN authoritative and
  enters persistent correction guidance until the camera confirms that
  position.
- **Reset to starting position** verifies the standard physical setup, resets
  the legal game and accepted physical/reference state together, activates
  Offline legal mode, and persists the resulting state.
- **Remain in board setup** explicitly activates and persists Board setup /
  diagnostic mode without altering the saved physical position.

Cancelling this choice shall preserve both the current active mode and all
persisted mode, FEN, physical-position, and reference data.

Offline legal mode shall display a persistent turn indicator beside the move
rules selector:

- White fill with a dark border means White to move.
- Black fill with a dark border means Black to move.
- Gray means the authoritative legal game is complete.
- The indicator is hidden in Board setup / diagnostic mode.

The indicator is driven only by the authoritative legal position. Camera,
alignment, occupancy, and correction messages shall not change it. Its tooltip
and accessible name shall state the same turn or game-over information.

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

The first piece-synchronization milestone is accepted when:

1. The UI displays the official cburnett SVG pieces in the standard position.
2. White appears at the bottom after camera-derived side detection.
3. An ordinary physical move updates exactly one UI piece.
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
ambiguity handling, and explicit starting-position reset are implemented.
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

Implemented for non-capturing one-piece relocations.

### Milestone 4: Recovery

- Add explicit starting-position reset.
- Detect synchronization loss.
- Preserve logical state through temporary motion.
- Restore only metadata-compatible persisted references.
- Verify reset requests asynchronously using the current camera frame.

Implemented. Reset requires confirmed starting-position occupancy.

### Milestone 5: Offline legal moves

- Add authoritative FEN, side-to-move, legal-move generation, and game status.
- Reject illegal physical moves without changing logical/reference state.
- Add persistent turn and correction guidance.

Implemented for ordinary non-capturing moves.

### Milestone 6: Captures and special physical signatures

- Commit completed ordinary legal captures from a unique legal signature and
  the authoritative post-move position.
- Add stronger destination color-transition evidence for captures.
- Support castling, en passant, and promotion workflows.
- Reconcile multi-square physical signatures before atomic commit.

Ordinary single-action legal capture commit is implemented. Stronger
piece-color classification, multi-action capture handling, and special moves
remain future work.

## 20. Open Decisions

- Whether move history is persisted between application starts.
- Whether later arbitrary-position setup uses FEN entry, a graphical editor,
  or piece recognition.
