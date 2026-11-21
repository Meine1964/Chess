# Lichess Physical Move Synchronization Specification

**Version:** 0.4
**Status:** Full physical move synchronization and local promotion selection implemented; extended move kinds awaiting live acceptance
**Platform:** Python on Windows

**Related specifications:**

- [Lichess physical-board integration roadmap](./lichess-integration-roadmap.md)
- [Reference-image move tracking](./reference-image-move-tracking-specification.md)
- [Chess-piece UI rendering and move synchronization](./chess-piece-ui-tracking-specification.md)
- [Projector/camera chessboard system](./projector-camera-chessboard-specification.md)

## 1. Purpose

This specification defines the physical-move synchronization slices of Phases
7 and 8 of the Lichess physical-board integration: synchronizing moves already
accepted by Lichess onto the physical board and submitting local physical
moves, including captures, castling, en passant, and promotion.

When Lichess advances the remote game, the application shall:

1. Advance the authoritative server and Lichess UI positions.
2. Keep the accepted physical position and camera reference unchanged.
3. Queue each remote move not yet confirmed physically.
4. Project guidance for the first pending move.
5. Verify the expected physical change with the camera.
6. Advance the physical position and replace the camera reference only after
   exact confirmation.

Outgoing ordinary moves, non-promotion captures, castling, en passant,
auto-queen promotion, and auto-queen capture-promotion are supported.
Local physical underpromotion selection is supported through the projected
virtual-rank workflow specified below.

## 2. Design Principles

### 2.1 Lichess is authoritative

Lichess determines the game ID, initial position, complete move history, side
to move, game status, and result.

The application shall never rewrite server history from camera observations.

### 2.2 Physical state advances only from camera evidence

A remote update shall not:

- Advance the accepted physical position.
- Replace the accepted camera reference.
- Pretend that a projected move was completed.

Only an unobstructed, settled camera observation matching the expected physical
signature may advance the physical state.

### 2.3 Verification is narrower than inference

When a remote move is pending, its UCI move and legal physical signature are
known. The camera shall verify that one expected signature rather than choose
among every legal move.

### 2.4 Network and camera commits remain separate

The server/UI commit occurs when the move appears in authoritative Lichess
history. The physical/reference commit occurs later, after camera confirmation.

Failure in the second commit shall not roll back or alter Lichess state.

### 2.5 Projected light is not camera evidence

Colored guidance may change the observed appearance of squares. The projector
shall be black while frames are evaluated for physical confirmation.

Pending legal-move source and destination squares shall be shown in white on
both the physical-position UI board and the projector. Red is reserved for
illegal correction and reconciliation discrepancies. The physical UI guidance
shall remain visible while the projector is black for move verification and
shall clear only when the move is confirmed, replaced, or enters correction.
If projected guidance is blacked out after manipulation settles but the
settled camera image still matches the accepted reference, the same pending
guidance shall be projected again. Camera or board drift shall not permanently
hide an unperformed opponent move.

### 2.6 Authoritative game information

The Lichess panel shall display both players with their colors and ratings when
available, separate White and Black clocks, side to move, game status, and
result. Full snapshots establish player identity and state; incremental state
events update clocks, turn, status, and result without discarding player
identity.

When polling cannot provide authoritative clocks, both clocks shall display
`--:--` rather than `0:00`. Zero is a valid displayed clock only when the
transport identifies clock values as available. Before an active game is
loaded, the panel shall explicitly report that no game information is
available rather than displaying stale values from another selected game.

## 3. Scope

### 3.1 First implementation

The first implementation shall support:

- A standard Lichess game that starts from the normal initial position.
- Streamed or read-only-polled Lichess snapshots.
- Complete move-history prefix validation.
- One or more newly reported moves in a single update.
- An ordered queue of moves not yet confirmed physically.
- Ordinary moves, non-promotion captures, castling, en passant, auto-queen
  promotion, and auto-queen capture-promotion for both colors.
- Direct submission of those supported local-player move kinds and exact local
  underpromotion suffixes selected through the projected virtual rank.
- Opponent-move projection and physical confirmation.
- Durable UI guidance.
- Incorrect physical-action correction.
- Preservation of the accepted physical reference until confirmation.
- Reconnection while a move remains pending.
- Explicit reconciliation when remote and physical positions cannot be joined
  safely.

### 3.2 Staged extensions

Dedicated projected replacement-piece guidance for incoming server-directed
underpromotion remains staged. Incoming underpromotion already preserves the
exact authoritative suffix and uses ordinary source/destination guidance.

### 3.3 Excluded

The implemented synchronization slice shall not:

- Accept incoming challenges.
- Infer missing server moves from a camera image.
- Automatically assign piece identity from occupancy alone.
- Accept a physical move that is absent from Lichess history.

## 4. State Model

The application shall maintain three distinct positions.

### 4.1 Server position

The position produced by applying the complete validated Lichess move history
to the game's initial FEN.

This is the ultimate authority during a connected game.

### 4.2 Lichess UI position

The position rendered in the `Lichess position` board panel.

It normally equals the server position immediately after each network update,
even when the physical board is still behind.

### 4.3 Physical position

The position associated with:

- The last camera-confirmed logical piece map.
- The accepted canonical camera reference.
- The legal state from which the next expected physical signature is derived.

The physical position may temporarily be behind the server position by one or
more queued moves.

## 5. Core Data

The synchronization controller shall own or receive:

- Lichess game ID.
- Player color.
- Opponent name.
- Initial FEN.
- Last validated complete server move tuple.
- Server legal game.
- Physical legal game.
- Number of server plies confirmed physically.
- Ordered pending UCI move queue.
- Active expected physical signature.
- Current synchronization state.
- Last durable user guidance.

The controller shall not own:

- Camera frames.
- Projector windows.
- Qt widgets.
- HTTP connections.
- Authentication tokens.

It should be a pure, directly testable state machine.

### 5.1 Active-game selection

When polling reports exactly one ongoing game, the application shall select it
automatically. When polling reports multiple ongoing games, the application
shall not assume that the first API result is the game visible in the user's
browser. It shall display all ongoing games in an explicit selector and pause
game snapshots, board updates, and physical synchronization until the user
selects one.

The selector shall identify each game by opponent, player color, turn state,
and game ID. Changing the selection shall discard the prior game's
synchronization controller before the selected game's snapshot is applied.
If the selected game is no longer ongoing, polling shall return to the
selection-required state.

## 6. Runtime States

### 6.1 `disconnected`

No active Lichess synchronization session exists.

### 6.2 `attaching`

An active game has been discovered, but the relationship between the remote
and physical positions has not yet been established.

### 6.3 `reconciliation-required`

The current remote position cannot be joined safely to the accepted physical
position.

No physical move shall be committed automatically in this state.

### 6.4 `synchronized`

The server, Lichess UI, physical legal position, and accepted reference all
represent the same confirmed position. No remote move is pending.

### 6.5 `guidance`

At least one remote move is pending. The application is presenting projector
and UI instructions for the first queued move.

### 6.6 `observing`

The projector is black and the camera is evaluating physical evidence for the
active expected signature.

### 6.7 `correction`

A settled physical action does not produce the expected final position.

The server and physical authorities remain unchanged. Red correction guidance
remains visible until the physical board is restored or reaches the expected
position.

### 6.8 `game-complete`

Lichess reports a terminal status and all reported moves have been confirmed
physically. The final result remains visible.

## 7. Attaching to a Game

### 7.1 Safe attachment

On initial attachment, the application shall compare:

- The current server piece map.
- The accepted physical logical piece map.
- Available persisted game identity and physical legal state.

Attachment is safe when either:

1. The game is new and both positions are the standard initial position; or
2. The persisted game ID, confirmed ply count, physical legal state, and
   accepted reference form a verified continuation of the same server history;
   or
3. The current server and accepted physical legal positions match exactly and
   the server position can safely initialize the synchronization cursor.
4. During current-position reconciliation, stable live occupancy exactly matches the
   current server position and no occupied square has a confidently detected
   opposite-color piece. In this case, reconciliation may adopt the
   authoritative Lichess piece map even though individual same-color piece
   types cannot be verified optically.

### 7.2 Unsafe attachment

If the positions differ and there is no verified move-history relationship,
the application shall enter `reconciliation-required`.

Without stable live reconciliation, it shall not:

- Queue the entire historical game automatically.
- Replace the accepted reference with the latest camera frame.
- Infer piece identity from occupancy.
- Assume that a matching occupied-square set means the positions are equal.

### 7.3 Current-position adoption

When attachment requires reconciliation and calibrated live observation is
available, the application shall automatically black the projector, reset
occupancy stabilization, and compare stable live observations with the current
authoritative Lichess piece map rather than with a stale persisted physical
map. **Reconcile physical board** remains available to restart or repeat this
process manually.

Discrepancies are:

- A Lichess-occupied square observed empty.
- A Lichess-empty square observed occupied.
- A square occupied in both positions where appearance classification
  confidently reports the opposite piece color.

Stable occupancy and appearance classification shall be combined before
creating discrepancies. A confident White or Black appearance shall establish
that a square is occupied even when the unsupervised occupancy detector calls
it empty. A confident empty appearance shall establish that it is empty even
when the occupancy detector calls it occupied. Uncertain appearance shall
leave the stable occupancy result unchanged. This prevents clearly visible
pieces from producing widespread false red reconciliation markers.

All discrepancy squares shall be marked red simultaneously on the physical UI
board and, when available, on the projected board. Uncertain color
classification shall not create a discrepancy. A same-color incorrect piece
type is intentionally undetectable and shall be accepted as the authoritative
Lichess identity.

After projected red reconciliation marks return to black, the application
shall discard occupancy and color observations for one second, reset occupancy
stabilization, and only then evaluate the board again. Red projector light and
the camera's resulting exposure transition shall not create a subsequent
reconciliation discrepancy.

When no discrepancy remains, reconciliation shall replace the stale physical
piece map with the complete Lichess map, capture the settled live image as the
accepted reference, initialize the synchronization cursor at the current
server history, clear every red mark, and continue from the current position
without replaying historical moves.

While reconciliation is active, the reconciliation control shall read
**Stop reconciliation**. Pressing it is an explicit user confirmation that the
physical board visibly matches the authoritative Lichess position even when
automatic camera comparison has not converged. The application shall then
remove every projected reconciliation mark, return the projector to black,
discard at least one second of camera exposure transition, and require a
stable quiet observation window. Only that later clean frame may replace the
move reference and continue synchronization from the current server history.
Because the user explicitly accepted the visible position, final clean-frame
capture shall not consult the previous appearance reference; that reference
may itself be contaminated and be the reason reconciliation was requested. If
no calibrated current frame or required Lichess metadata is available, the
application shall keep reconciliation active and report that manual acceptance
cannot yet complete. Closing the application before clean capture completes
shall leave the previous accepted state unchanged.

The first live acceptance shall use a fresh Lichess game and a physical board
in the standard starting position.

## 8. Server History Processing

### 8.1 Complete-history validation

Each Lichess snapshot shall provide a complete move tuple from the initial
position.

For an existing game, the previously accepted server move tuple shall be an
exact prefix of the new tuple.

### 8.2 Appended moves

Moves after the validated prefix shall be:

1. Parsed against the server legal position in order.
2. Applied to the server position.
3. Added to the physical-confirmation queue.
4. Rendered immediately in the Lichess UI position.

### 8.3 Inconsistent history

The application shall stop automatic synchronization and surface an explicit
error if:

- The game ID changes unexpectedly.
- The initial FEN changes.
- Existing history is rewritten or shortened unexpectedly.
- A reported move is illegal in the preceding server position.
- A polled export and previously accepted history cannot be joined.

It shall not silently replace the known history.

### 8.4 Multiple appended moves

Streaming or a history-bearing reconnection may reveal multiple new plies.

They shall be queued and confirmed physically in server order. Only the first
pending move may be active in the projector and camera workflow.

Current-position polling reports only the authoritative latest move. If its
new FEN is not exactly the result of that legal move from the prior polling
FEN, the application shall establish a new reconciliation baseline instead of
creating a multi-ply queue from unavailable history.

Polling shall obtain the current FEN and authoritative last move from the
authenticated ongoing-games endpoint. The first observation establishes a
current-position baseline. Later observations append `lastMove` only when that
legal move transforms the prior FEN into the newly reported FEN. A transition
that cannot be verified as one legal move shall establish a new baseline and
require physical reconciliation rather than inventing missing history. The
public game-export endpoint shall not be used for live synchronization because
Lichess deliberately delays ongoing-game exports by multiple moves.

## 9. Expected Physical Move

For the first pending UCI move, the controller shall:

1. Generate legal physical signatures from the physical legal game.
2. Find the signature with the exact UCI move.
3. Reject the synchronization transition if no exact legal signature exists.
4. Expose that signature as the only expected physical action.

The physical legal game is intentionally behind the server game while queued
moves remain. This allows each queued move to be validated against the correct
preceding physical position.

## 10. Projection Workflow

### 10.1 Guidance presentation

For an ordinary move or capture, the projector shall identify:

- Source square.
- Destination square.

The UI shall provide durable text, for example:

```text
Stockfish played e7-e5. Move the physical piece from e7 to e5.
```

For a move made by the local player in the browser:

```text
Reproduce your Lichess move e2-e4 on the physical board.
```

Opponent-move guidance shall remain projected while the player performs the
move. A hand or similarly large moving object approaching or entering the
physical board shall start manipulation tracking but shall not remove the
guidance. The guidance shall be removed only after the manipulation has become
quiet for the required consecutive observations, indicating that the hand has
left the board. It shall not use a fixed display timeout.

Intent detection shall cover the calibrated 8x8 board plus a
perspective-aware one-square margin around every edge. Motion elsewhere in the
camera image shall be ignored. This margin ensures that a hand approaching a
piece on the nearest rank or an outer file starts manipulation tracking even
when only fingertips enter the 64-square board area.

The manipulation detector may compare frames captured under the same static
guidance projection, but its frames and scores shall never update occupancy,
infer a chess move, or replace the accepted reference. The detector shall arm
only after one second of static projected observations so projector startup
cannot be mistaken for hand activity. It shall then require material activity
in consecutive frames, keep the guidance projected throughout that activity,
and request projector blackout only after the scene is quiet for the required
consecutive observations.

### 10.2 Camera observation

After manipulation has started and then settled, the guidance projection shall
be removed and the projector returned to black. The application shall discard
frames during a one-second projector-settling interval, reset transient
tracker activity, and only then evaluate change scores, color evidence,
obstruction, or settling. The durable UI guidance shall remain visible during
physical confirmation.

Guidance may be shown again when:

- No physical activity was observed.
- The user requests the instruction again.
- A correction was restored and the expected move remains pending.

Captures shall support both complete single-action evidence and
captured-piece-first handling. Removing only the captured piece shall preserve
the authoritative state and display durable instructions to complete the move
from the expected source. Moving the attacker to the destination shall then
confirm the capture against the original reference.

Castling shall support complete final-position evidence and a king-first,
rook-second workflow. Moving the king to its castling destination shall enter
a non-authoritative pending-castling state while retaining the original
position and camera reference. Moving the rook from its castling source to its
castling destination shall then confirm the complete four-square signature
against the original reference.

Complete castling may use the reduced legal-source change threshold on all
four changed squares when classified post-move occupancy and piece colors
match the expected castled position exactly. This permits lower-contrast rook
movement without weakening the exact four-square state check.

En passant shall support complete final-position evidence and a
captured-pawn-first workflow. Removing only the bypassed pawn from its square
shall enter a non-authoritative pending-en-passant state while retaining the
original position and camera reference. Moving the attacking pawn from its
source to the en passant destination shall then confirm the complete
three-square signature against the original reference.

Complete en passant may use the reduced legal-source change threshold on all
three changed squares only when classified post-move occupancy and piece
colors exactly match the expected source-empty, destination-attacker, and
captured-pawn-square-empty state.

### 10.3 Special-move projection

Castling guidance shall mark the king source/destination and rook
source/destination in white on both the physical UI board and projector.

En passant guidance shall mark the attacking pawn source, destination, and
captured-pawn square in white on both the physical UI board and projector.

Later slices shall derive involved squares from the expected signature:

- Promotion: source and destination.

### 10.4 Promotion modes

The persistent promotion mode shall be either **Auto-queen** or
**Select promoted piece**. Auto-queen shall remain the default.

#### 10.4.1 Auto-queen

A pawn reaching the last rank shall be interpreted and submitted as a queen
promotion. The camera shall require the promotion source to become empty and
the destination to contain a piece of the moving color. Because occupancy and
color evidence cannot distinguish a pawn from a same-color queen, that evidence
shall commit the authoritative queen without claiming optical piece-type
verification.

Capture-promotion shall support complete final-position evidence and the
existing captured-piece-first workflow. Removing the destination piece alone
shall preserve the original authoritative state and reference until the
attacking pawn reaches the promotion destination.

#### 10.4.2 Select promoted piece

For a local legal promotion or capture-promotion, source-empty and
promotion-destination-occupied evidence shall begin promotion selection rather
than commit or submit a queen. The authoritative game position, accepted
physical position, and accepted camera reference shall remain unchanged until
selection and physical replacement are complete.

The projector shall display icons for queen, rook, bishop, and knight in four
virtual squares outside the promotion edge. These virtual squares shall:

- Form one extra rank immediately beyond the calibrated playing grid.
- Use the calibrated square width and height at that edge.
- Have adjacent centers separated by exactly one square pitch.
- Bind each occupancy region to the piece shown in that region; icon order is
  therefore not independently significant.
- Scale each icon to fit completely inside its virtual square without crossing
  an edge.
- Orient every icon toward the physical player's viewing side, independently
  of which promotion edge contains the virtual rank. In Offline legal game
  mode, the viewing side is White's calibrated starting side. In a Lichess
  game, the viewing side is the configured local player's calibrated starting
  side. An icon projected on the opposite edge shall therefore have the same
  player-facing orientation as an icon projected on the player's own edge,
  rather than being rotated toward the edge that contains it.
- Render the piece as a bright white filled silhouette. The glyph shall use
  substantially more illuminated area than a thin outlined chess symbol while
  remaining fully inside its virtual square.
- Projectively warp each icon into all four corners of its calibrated
  quadrilateral rather than drawing it in an axis-aligned bounding rectangle.
  The icon shall therefore inherit the board's rotation, perspective, skew,
  and local square size. Each icon uses its own quadrilateral so calibration
  may legitimately make squares on the far side larger than squares on the
  near side.
- Remain inside both the projector output and camera image. If all four regions
  cannot be mapped safely, selection shall stop with an explicit calibration
  or visibility error rather than defaulting to queen.

The icons shall remain projected while the player moves the pawn, including
while a hand is visible or board activity continues. When the scene becomes
unobstructed and settled, the projector shall turn black and the camera shall
evaluate the four virtual squares:

The choice icons and the selected-piece icon on the promotion destination shall
remain steady during their respective manipulation phases. Projection startup,
projector refresh variation, and camera sampling of projected pixels shall not
be treated as hand activity and shall not cause repeated black/reproject
flashing. Black output is permitted only for the deliberate settled evidence
measurement.

- Exactly one occupied virtual square selects its displayed promotion piece.
- No occupied virtual square shall select nothing; the icons shall reappear and
  selection shall continue.
- Multiple occupied or otherwise ambiguous virtual squares shall select
  nothing, report the ambiguity, and reproject the icons for another attempt.
- Projector-induced icon pixels shall never be accepted as occupancy evidence;
  every choice measurement shall use a black-projector frame.

After a choice is detected, the application shall retain the selected piece
provisionally and provide immediate visible confirmation:

- The occupied virtual square containing the detected pawn shall be illuminated
  as a solid white square.
- The promotion destination on the playing grid shall display the icon of the
  selected piece.
- The other three choice icons shall remain projected and selectable.

The player may move the pawn from the selected virtual square to any other
projected choice before completing replacement. After each manipulation
settles, the projector shall briefly turn black and evaluate all four virtual
squares. Exactly one occupied choice shall become the new provisional
selection, move the solid white highlight to that square, and replace the icon
on the promotion destination. Multiple occupied choices shall preserve the
previous selection and request an unambiguous choice.

This reversible confirmation projection shall remain visible while the player
changes the choice or removes the pawn from the virtual row and places the
corresponding physical piece on the promotion destination. Final replacement
verification shall begin only when all four virtual choice squares are empty.
If completion evidence is not yet present, all options shall reappear together
with the last selected white highlight and selected-piece destination icon.

Completion shall require all four virtual squares to be empty, the promotion
source to remain empty, and the destination to be occupied by the moving color.
The camera cannot prove physical piece type, so the last unambiguous virtual
selection is authoritative for the UCI promotion suffix; the destination icon
is instruction and confirmation feedback, not optical proof. Only then shall
the exact move be submitted to Lichess and, after server acceptance, committed
to the physical position and camera reference.

Cancellation, loss of calibration, an uncertain network result, or failure to
complete replacement shall not silently apply a queen. The existing authority,
failure, retry, and reconciliation rules shall apply without weakening.

For an incoming Lichess promotion, the server-specified promotion piece is
already authoritative. The application shall guide replacement with that
specific piece and shall not present four selectable choices.

## 11. Camera Confirmation

### 11.1 Exact expected match

The existing reference tracker shall evaluate the settled observation against
the active expected signature.

Acceptance shall require:

- All expected changed squares satisfy their evidence requirements.
- Source vacancy is confirmed.
- Destination occupancy and mover color are confirmed where required.
- No unexplained changed square exceeds the allowed residual policy. A visual
  residual is explained when the color classifier confidently confirms that
  the square still has the occupancy and piece color required by the
  authoritative position after the candidate move.
- The board is unobstructed and settled.

When legal physical signatures are available, the tracker shall accept only a
matching signature. If no signature matches, it shall report ambiguous or
correction evidence rather than falling back to an unrestricted source and
destination inferred from visual change scores.

Whole-board inter-frame motion is advisory for legal-signature tracking and
shall not prevent evaluation of a clear final position. Hand safety for this
path is enforced by settled observations, large-obstruction rejection,
released-piece-footprint checks, and exact signature evidence. Diagnostic
tracking without legal signatures remains gated by inter-frame motion.

### 11.2 Commit

After exact confirmation, the application shall:

1. Apply the UCI move to the physical legal game.
2. Update the physical piece synchronizer using the authoritative resulting
   position.
3. Commit the settled canonical image as the new accepted reference.
4. Increment the physically confirmed server ply count.
5. Remove the move from the queue.
6. Clear correction state.
7. Activate the next queued move, or enter `synchronized`.

### 11.3 Reference invariant

No frame captured while guidance is projected, while a hand is present, or
while an incorrect physical position is visible may replace the reference.

## 12. Incorrect Physical Actions

### 12.1 Unexpected move

If a settled action represents a move other than the active expected move:

- Do not advance the physical legal game.
- Do not remove the pending move.
- Do not replace the accepted reference.
- Mark the relevant incorrect squares red.
- Keep durable correction guidance visible.

### 12.2 Incomplete multi-action move

An isolated settled piece removal remains illegal unless it is a recognized
first step of the active expected capture workflow.

A completed king relocation remains pending rather than illegal when it is the
king-first step of an active expected castling signature. The authoritative
state and reference shall remain unchanged until the rook completes castling.

A captured pawn removed from its separate square remains pending rather than
illegal when it is the captured-pawn-first step of an active expected en
passant signature. The authoritative state and reference shall remain
unchanged until the attacker reaches the en passant destination.

### 12.3 Correction completion

The user may:

- Restore the previous physical position; or
- Complete a final board position that exactly matches the expected signature.

After restoration, the application shall resume guidance for the same pending
move. After an exact expected final position, it may commit normally.

## 13. Local Supported-Move Submission

Local submission is available only when the physical and server positions are
synchronized, it is the configured player's turn, and the camera identifies a
unique ordinary move, non-promotion capture, completed castling signature, or
completed en passant signature, queen promotion, or queen capture-promotion.

The application shall use a two-phase commit:

1. Preserve the settled candidate image without changing the accepted
   reference, physical position, server position, UI position, or turn.
2. Submit the UCI move once with
   `POST /api/board/game/{gameId}/move/{move}`.
3. After a successful response, advance the controller's server and physical
   positions, commit the preserved image as the accepted reference, update the
   UI, persist synchronized state, and display the accepted move.
4. After an explicit HTTP rejection, preserve the previous authoritative state
   and reference and require the physical move to be restored.

The application shall pause further camera move inference while submission is
pending. It shall never retry a move automatically.

If the POST outcome is uncertain because the response is interrupted or
invalid, the submission worker shall fetch the authoritative game snapshot:

- If history equals the previously confirmed history plus the submitted move,
  treat the move as accepted and commit it once.
- If history still equals the previously confirmed history, treat the move as
  rejected and require restoration.
- If the refresh fails or history has another shape, enter reconciliation and
  preserve both the previous reference and the pending physical discrepancy.

Polling fallback remains capable of submitting moves. Polling affects only how
authoritative game updates are received; it does not make a valid Board API
session read-only.

After Lichess accepts a local submission, the application shall advance the
polling cursor through that move after verifying that it transforms the
cursor's prior FEN into the committed server FEN. This update shall occur
before a later opponent poll is interpreted, so an immediate reply reported as
`lastMove` remains a verifiable one-ply transition rather than appearing to
shorten or rewrite history.

## 14. Reconnection and Polling

### 14.1 Network interruption

Network loss shall not clear:

- Confirmed server history.
- Physical legal state.
- Pending move queue.
- Active correction state.
- Accepted camera reference.

Projection may be cleared while the remote state is uncertain.

### 14.2 Reconnection

After reconnection, the application shall:

1. Fetch the complete current move history.
2. Validate the previous history as a prefix.
3. Append newly discovered moves to the existing queue.
4. Resume the first unconfirmed physical move.

It shall not duplicate already queued or confirmed moves.

### 14.3 Polling fallback

Current-position polling shall verify each reported `lastMove` against the
prior FEN. A verified transition appends one move to the server cursor. An
unverifiable transition replaces the polling baseline and enters physical
reconciliation; it shall not fabricate or queue unreported intermediate moves.

The five-second polling interval may delay guidance but shall not alter
correctness.

## 15. UI Requirements

The UI shall distinguish:

- Lichess/server position.
- Physical/reference position.
- The active side to move using both a color indicator and visible text.
- Number of pending physical moves.
- Active expected move.
- Player color and opponent.
- Network mode: stream or polling.
- Synchronization state.

The `Lichess position` panel shall advance immediately from remote history.

The `Physical position` panel shall advance only after camera confirmation.

The main window shall present both boards simultaneously in a four-quadrant
dashboard:

- Top left: camera preview.
- Top right: physical position.
- Bottom left: scrollable interaction controls and status.
- Bottom right: Lichess position.

Transient camera phases such as activity, settling, and candidate scoring shall
not overwrite durable move guidance or correction text.

## 16. Persistence

The implementation should persist:

- Lichess game ID.
- Initial FEN.
- Last validated complete move tuple.
- Physically confirmed ply count.
- Physical legal FEN.
- Pending UCI queue.

Persisted synchronization state is valid only when it can be associated with
the same accepted physical reference and calibration context.

If that association cannot be verified after restart, the application shall
enter `reconciliation-required`.

Persistence may be implemented after the first in-process ordinary-move slice,
but restart recovery is required before Phase 7 is complete.

## 17. Diagnostics and Security

Diagnostics may include:

- Game ID.
- Initial and current FEN.
- Confirmed and pending ply counts.
- Pending UCI moves.
- Expected physical signature.
- Tracker scores and rejection reason.
- Reference commit or preservation decision.
- Stream or polling mode.

Diagnostics shall never include:

- Access token.
- Authorization header.
- Complete credential-bearing requests.

The application shall continuously maintain an ignored local runtime diagnostic
bundle:

- `diagnostics/runtime-state.json`: complete current token-free state.
- `diagnostics/runtime-events.jsonl`: bounded recent state-transition history.
- `diagnostics/runtime-camera.png`: latest camera frame at a recorded physical
  event.
- `diagnostics/runtime-accepted-board.png`: last accepted canonical board
  reference.
- `diagnostics/runtime-physical-ui.png`: rendered physical-position UI board.
- `diagnostics/runtime-lichess-ui.png`: rendered Lichess-position UI board.

The JSON state shall include the server and physical FEN values, complete
server and pending move tuples, square-by-square logical maps, tracker state,
last rejected move, calibration availability and quality metrics, projector
mode, selected camera and display metadata, raw-frame dimensions, application
runtime versions, selected and pending projected squares, and durable UI
messages. Structured events shall cover application and camera startup,
hardware selection and errors, saved-calibration restoration, every
calibration operation and outcome, board-snapshot detection, physical-board
clicks, square-projection outcomes, Lichess transitions, and physical
move-tracker decisions. Illegal-correction diagnostics shall expose the
current projection/blackout phase, generation, projected physical squares,
and detector manipulation state. One structured event shall be recorded for
initial projection or reprojection, first confirmed hand entry, blackout after
hand departure, black-settling completion, and accepted-reference
restoration. Repeated camera frames within one phase shall not generate
duplicate lifecycle events. Events may carry token-free operation details
needed to reconstruct the failure context. Runtime files shall remain
excluded from version control, image files shall be overwritten rather than
accumulated, and event history shall be bounded.

## 18. Implementation Boundaries

The preferred implementation surfaces are:

- A new pure Lichess synchronization controller for state and queue decisions.
- The existing Lichess client and worker for authoritative snapshots.
- The existing legal-game model for server and physical legal cursors.
- The existing legal signature matcher for expected physical evidence.
- The existing reference tracker for obstruction, settling, and reference
  commit.
- The existing projector for guidance and red correction.
- The main camera application for Qt signal wiring and presentation only.

The state machine shall be unit-tested without Qt, a camera, a projector, or
network access.

When an active Lichess game acquires physical-board authority, it shall cancel
any transient Offline starting-position reset, saved-position restoration, or
automatic-reset recovery request left from startup or mode selection. Those
Offline requests shall not take priority over Lichess reference tracking or
leave the camera loop waiting for a standard setup. Cancelling transient
recovery shall not alter the accepted Lichess position or reference.

## 19. First-Slice Acceptance Tests

The first ordinary-move slice is accepted when all of the following pass.

### 19.1 Initial synchronization

- A fresh standard Lichess game attaches to a standard physical position.
- A mismatching active game enters `reconciliation-required`.
- Attachment does not replace the accepted camera reference.

### 19.2 Remote history

- One appended legal UCI move becomes pending.
- Two appended moves are queued in exact server order.
- Duplicate snapshots do not duplicate pending moves.
- Rewritten or shortened history is rejected.

### 19.3 Projection and confirmation

- The active source and destination are projected.
- The projector is black before camera evaluation.
- The exact physical move commits the physical legal state and reference.
- The Lichess UI position may be ahead while the physical panel remains behind.
- After confirmation, both panels show the same position when the queue is
  empty.

### 19.4 Incorrect actions

- A wrong destination is rejected and marked red.
- Moving a different piece is rejected.
- Isolated removal is rejected.
- The accepted reference remains unchanged.
- Restoring the board resumes the expected guidance.

### 19.5 Queue and reconnect

- The next queued move activates only after the first is confirmed.
- Reconnection preserves the pending move.
- Polling-discovered moves follow the same queue behavior.
- Confirmed moves are not replayed.

## 20. Live Acceptance Sequence

The initial live test shall use a new casual game against Stockfish:

1. Reset the physical board to the standard starting position.
2. Start a new standard Lichess game.
3. Connect and verify both positions are synchronized.
4. Make one ordinary move in the browser.
5. Verify that the Lichess panel advances while the physical panel does not.
6. Verify source/destination guidance.
7. Reproduce the move physically and wait for confirmation.
8. Verify both panels synchronize.
9. Wait for Stockfish's ordinary reply.
10. Reproduce that move physically.
11. Repeat one move with an intentionally wrong destination.
12. Verify red correction, reference preservation, restoration, and successful
    completion.

Captures and special moves shall receive separate live acceptance before their
Phase 7 slices are declared complete.

## 21. Implementation Status

The ordinary-move, non-promotion capture, castling, en passant, and auto-queen
promotion slices are implemented:

- A pure synchronization controller validates complete history prefixes.
- Server and physical legal cursors advance independently.
- Up to two moves discovered during initial fresh-game attachment may queue.
- Later polling updates may append multiple moves in server order.
- Multiple ongoing games require explicit selection; a single game is selected
  automatically.
- Polling maintains a current-position legal cursor from authenticated FEN and
  `lastMove` observations without using delayed public game exports.
- The Lichess board advances immediately while the physical board remains
  unchanged.
- Stable live occupancy can adopt the authoritative current Lichess position,
  with confident opposite-color discrepancies marked red.
- **Stop reconciliation** lets the user explicitly confirm a visibly matching
  board and perform the same authoritative current-position adoption when
  automatic camera comparison does not converge.
- The active ordinary signature is the only legal signature offered to the
  reference tracker.
- Guidance remains projected throughout confirmed board-or-halo manipulation;
  only after consecutive quiet observations does the projector return to black
  and settle before camera evaluation resumes.
- Pending legal guidance remains white on both the physical UI board and
  projector; red remains reserved for correction and reconciliation.
- Correct physical evidence commits the physical position and reference.
- Premature and incorrect physical moves enter durable red correction without
  changing the authoritative state.
- Incoming and local captures use the same two-phase authority boundaries as
  ordinary moves, including one-action and captured-piece-first camera
  evidence.
- Incoming and local castling use four-square guidance and accept either the
  complete final position or king-first, rook-second physical handling.
- Incoming and local en passant use three-square guidance and accept either
  the complete final position or captured-pawn-first physical handling.
- Incoming and local queen promotion use source/destination guidance and
  commit the authoritative queen from exact occupancy/color evidence.
- Queen capture-promotion accepts complete evidence or captured-piece-first
  physical handling.
- Auto-queen remains the default. Select promoted piece implements the
  virtual-rank projection, settled black-frame choice, physical replacement,
  perspective-correct player-facing bright icons, and exact UCI suffix
  requirements in section 10.4.

The validated automated suite contains 311 passing tests. Live ordinary-move,
polling, reconciliation, and guidance acceptance has been exercised on the
target camera/projector setup. Lichess capture, castling, en passant, and
auto-queen behavior is automated-test validated and awaits live acceptance.
The dedicated player, clock, status, and result display is automated-test
validated and awaits live acceptance. Draw-offer, draw-response, and
resignation and abort controls are automated-test validated and await live
acceptance. Local select-promoted-piece geometry on all four physical edges,
viewer-side-oriented bright fitted icons, steady manipulation projection,
settled black-frame choice, projection feedback, exact UCI suffix handling, and
offline/Lichess commit paths are automated-test validated and await live
acceptance. Incoming
server-directed underpromotion uses the exact authoritative suffix and ordinary
source/destination physical guidance; dedicated projected replacement-piece
guidance remains staged.
