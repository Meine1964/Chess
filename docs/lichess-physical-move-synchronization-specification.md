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

Opponent-move guidance shall remain projected until camera evidence shows that
a real hand or arm entered the board region or a bounded guidance-display
interval elapsed.
Lichess guidance shall use the calibrated boundary-crossing detector, which
also drives the visible hand indicator while guidance is projected. It shall:

- arm against the projected scene before accepting activity;
- confirm entry when one connected component contains material activity inside
  the board, outside the board, and at the outer rim of the one-square
  observation halo;
- begin blackout immediately on confirmed entry, before waiting for the hand
  to depart;
- suspend and clear boundary-indicator observation before changing to black so
  frames containing the entering hand cannot become a new detector baseline;
- keep guidance steadily projected while no board entry or manipulation is
  detected; and
- after blackout settling, use the ordinary reference tracker to wait for a
  clear, stable board and verify the required move.

Hand departure, board-confined changes, the completed opponent move, projector
spill, piece silhouettes, and shadows shall therefore be handled by the same
black-board reference-tracking path used by Offline play rather than by a
second guidance-specific departure gate. A timer shall not black out and
reproject an untouched pending move because that creates visible flashing. If
the black, settled image after actual manipulation still matches the accepted
reference, guidance shall be reprojected and boundary-indicator observation
shall resume with a fresh baseline. The unchanged accepted scene shall not
reactivate the indicator. Runtime diagnostics shall record initial projection
or reprojection, confirmed boundary entry, blackout reason, and black-frame
verification readiness once per lifecycle. The edge-indicator diagnostics
remain the authoritative boundary-component measurements.

When exactly one Lichess move is pending, its legal physical signature and
resulting position are authoritative verification inputs, distinct from the
set of candidate signatures used for ordinary move discovery. After the
projector is black and the scene is stable, the tracker shall first apply its
full signature checks. If those checks are inconclusive but geometric
relocation identifies exactly the required source and destination, the tracker
shall accept that move only when the required result is legal and no
unexplained physical change exists outside the signature's expected-change
mask. It shall not relabel that exact authoritative relocation as illegal.
Wrong relocations, additional moved pieces, and unsupported or illegal pending
results shall remain rejected. Offline and local-move discovery shall retain
their ordinary candidate-ranking rules.

Implementation status (2026-11-21): authoritative required-signature
verification, shared boundary-detector guidance lifecycle, smooth local
photometric-drift normalization, outer-halo entry qualification,
entry-triggered blackout, steady untouched guidance, and decision diagnostics
are implemented. Focused regressions cover exact `e8-f8` verification,
wrong-relocation rejection, halo-only activity, genuine boundary crossing,
completed edge-piece changes, no-hand guidance stability, and transition
diagnostics. Automatic camera measurement of projector-specific optical spill
remains a possible later refinement rather than an acceptance dependency.

### 2.6 Authoritative game information

The Lichess panel shall display both players with their colors and ratings when
available, separate White and Black clocks, side to move, game status, and
result. Full snapshots establish player identity and state; incremental state
events update clocks, turn, status, and result without discarding player
identity.

During an active game, polling shall treat the authenticated ongoing-game FEN
and last move as authoritative for the current position. It shall retrieve a
non-streaming JSON game export on every polling interval to enrich that
position with player identity, clock samples, increments, status, and result.
When that export describes the current position, polling shall replace both
clock values with the Lichess values. Those values already include the mover's
increment, so polling shall never add an increment to a same-position Lichess
clock sample.

The export may temporarily lag behind the authenticated ongoing-game position.
When its FEN is exactly one reported legal move behind, polling shall accept and
emit the newer ongoing position. It shall first count down the previous mover's
clock to the observation time, then add that mover's increment exactly once,
switch the active countdown to the opponent, and use authenticated
`secondsLeft` for the local player's clock. Repeated polls of the same move
shall continue decreasing only the active player's estimated clock and shall
not add the completed mover's increment again. The next same-position export
shall replace both estimates with Lichess values without adding any increment.
If polling resumes with multiple missing plies, it shall count each side's
missing moves from the export position and add each corresponding increment
once to the raw export clocks. It shall derive the local player's elapsed time
from that increment-adjusted raw clock and authenticated `secondsLeft`, then
assign the remaining elapsed wall time since the export sample to the
opponent. This provides a bounded two-clock estimate while the export catches
up and continues counting down the current side locally. A later
same-position Lichess sample replaces both estimates directly. If any required
raw clock, sample timestamp, or local `secondsLeft` value is unavailable,
polling shall mark both clocks unavailable rather than inventing a value. A
lagging export shall never reject an official move, reset a valid polling
cursor, or terminate the polling worker. If a valid same-game export differs
from the authenticated ongoing position but its exact relationship cannot be
proven, polling shall preserve and emit the ongoing position, turn, and latest
move with both clocks unavailable. A later same-position export shall restore
authoritative clocks. A malformed ongoing position, invalid export FEN, or
wrong-game export shall still fail explicitly rather than be merged
speculatively.

Export clock samples shall retain their server sample timestamp. Elapsed time
shall be applied only after the worker has established which position the
sample describes; a stale export shall not charge all time since its last move
to a player who has already moved. Zero is a valid displayed clock only when an
authoritative source or the local monotonic countdown reaches zero. Before an
active game is loaded, the panel shall explicitly report that no game
information is available rather than displaying stale values from another
selected game.

Between authoritative stream events or polling snapshots, the displayed clock
for the side to move shall count down using locally elapsed monotonic time. The
inactive clock shall remain unchanged. Every new authoritative state shall
replace both clock values and restart the local elapsed-time baseline; an
estimated polling transition shall instead carry forward the elapsed active
clock as specified above. The display shall stop at zero. Reaching zero only
through the local monotonic
countdown shall not declare a timeout. However, when the authenticated
ongoing-games endpoint reports `secondsLeft == 0` while it is the local
player's turn, polling shall emit an `outoftime` terminal snapshot immediately,
with the opponent as winner and the corresponding result, even if the endpoint
still carries its lagging `started` status. Time cannot be restored without an
accepted move, so this authenticated zero is terminal clock authority. The UI
shall state the time-forfeit result, stop both local and projected countdowns,
and enable **New AI game** while later polling reconciles Lichess's terminal
status propagation. Local countdown shall stop when clocks are unavailable or
the effective game state is no longer active.

During an active timed Lichess game, the projector shall also display both
clocks immediately outside the board's logical a-file: White beside a1-a2 and
Black beside a7-a8. Clock placement shall read the already-established logical
board orientation solely to map that edge into calibrated projector geometry.
It shall neither request board-orientation detection nor modify the orientation
used to map move squares. Both clocks shall be readable from the board center.
Their projective quadrilaterals shall preserve winding and shall never reflect
the seven-segment glyphs.
Within those non-reflecting quadrilaterals, both complete clock faces shall be
rotated 180 degrees. Their digit and colon content shall be scaled to 70% of
its prior height around the clock region's unchanged center. Rotation and
scaling shall be composed in canonical clock space before the perspective warp
so they cannot rotate or translate the complete region around the projector
origin. Clock digits shall use a geometric seven-segment design rather than
depend on an operating-system font. A time below one hour shall use
`minutes:seconds`, for example `59:59`. At one hour or above, both on-screen
and projected clocks shall use `hours.minutes:seconds`, for example `1.29:20`;
minutes and seconds within the hour shall each use two digits. Projected times
below one hour shall retain a fixed two-digit minute field so both clocks
allocate identical width to each digit regardless of whether the minute value
has one or two digits. The projector shall render the hour separator as one
geometric full-stop dot and the minute/second separator as the existing colon.
This visual normalization shall not change clock values, visibility, placement,
or countdown behavior. Segment thickness, separator-dot radii, and result
fraction-bar thickness shall compensate for the calibrated projector-pixel
density at each clock or result quadrilateral. Compensation shall be
direction-aware: vertical segments use the local horizontal density and
horizontal segments use the local vertical density. The fixed White-clock,
Black-clock, and center-result layout shall always define the density
reference, regardless of whether a result is currently visible. The densest
region in that fixed layout on each
axis shall retain the established base thickness; regions with lower density
shall widen inversely, subject to a conservative upper bound that prevents
segments from merging. Showing or hiding a timeout or draw result shall not
change either clock face's rasterized strokes. Compensation shall not move or
resize a clock or result quadrilateral, change its text, or alter its
perspective transform. For a calibration with unequal near-side and far-side
density, corresponding projected strokes shall rasterize to comparable
projector-pixel thickness within antialiasing tolerance. The clocks shall be
bright white on the otherwise black edge, shall not replace or suppress
projected move and correction markings, and shall inherit the same locally
advancing display values as the on-screen clocks.

Implementation status (2026-11-21): calibrated direction-aware stroke
compensation is implemented for clocks, result glyphs, colon dots, and draw
fraction bars. The fixed full edge layout supplies a content-independent
density reference. Focused raster tests verify inverse density compensation,
bounded enlargement, unchanged clock rasterization when a result appears, and
preservation of timeout and agreed-draw rendering. The remaining limitation is
that compensation uses calibrated geometric pixel density rather than a
camera-measured photometric response map, so projector optical falloff that is
independent of geometry may still require later brightness equalization.

Projected clocks shall remain hidden until a complete White-side orientation
and a calibrated projector grid are available. They shall also remain hidden
when clocks are unavailable or calibration and diagnostic patterns own the
projector output. Any authoritative terminal result shall keep both clocks at
their last reported values, remove the projected turn dot, and show the result
in the edge gap. If authoritative clocks are unavailable, the same result
shall appear alone in the calibrated edge-result position. Locally displayed
zero shall not project or imply an authoritative timeout result.

Offline play shall use the same result-only edge rendering for every terminal
legal-game result, including White or Black checkmate wins and draws. A local
Lichess move that delivers checkmate shall refresh the result projection
immediately from the locally verified legal result rather than waiting for a
later server snapshot. Official Lichess terminal snapshots, successful
resignation, and successful draw acceptance shall converge on the same edge
result rendering.

When Lichess accepts a resignation request, that successful response is
immediate terminal authority; the application shall not wait for the resigned
game to remain present in the ongoing-games list. It shall infer the result
from the resigning local player (`0-1` for White, `1-0` for Black), stop the
local clocks, update local terminal state, and project the result. A later
official terminal snapshot may confirm or enrich this state.

Likewise, when Lichess accepts the local player's acceptance of an opponent
draw offer, that successful response is immediate authority for status `draw`
and result `1/2-1/2`. The application shall stop the clocks and project the
draw without waiting for another polling snapshot. Submitting a draw offer is
not terminal and shall continue to wait for the opponent's authoritative
response.

For an authoritative `outoftime` result, both on-screen and projected clocks
shall remain visible and stop advancing. The losing player's clock shall be
shown as `0:00`, while the winner's last reported time shall remain unchanged.
The projected losing clock shall also show a geometric expiration flag
immediately beside it on the side toward the middle of the board. The flag
shall use the same player-relative perspective and rotation as the clock.
The projected turn dot shall disappear. The result shall instead be shown as
an unspaced seven-segment score in the edge gap between the two clocks, with
`1` beside the winner's clock, `0` beside the losing `0:00` clock, and the dash
directly between them. Read from winner to loser, the displayed score is
exactly `1-0`. Its glyphs shall have the same physical size and orientation as
the clock glyphs. Its quadrilateral shall be derived from the calibrated board
plane, including fractional board coordinates and the extrapolated outer edge,
so the projected result retains the board's perspective and parallax
correction.

For an authoritative agreed draw with status `draw` and result `1/2-1/2`,
both on-screen and projected clocks shall remain visible at their last reported
values and shall stop advancing. The projected turn dot shall disappear. The
edge gap between the clocks shall instead show the conventional unspaced
fractional score `1/2-1/2`: each `1/2` shall be drawn as a stacked fraction
with a horizontal fraction bar, with a dash directly between the two
fractions. The numerator and denominator glyphs shall be smaller than the clock
glyphs so that each complete stacked fraction has the same physical height and
orientation as a clock glyph. Each visible numerator `1` shall be horizontally
centered above its denominator `2`, accounting for the asymmetric
seven-segment shape of `1`. The result quadrilateral shall use the same
calibrated board-plane geometry, including fractional board coordinates and
the extrapolated outer edge, as the timeout score. No expiration flag shall be
shown. Other drawn-game statuses shall retain their existing terminal-game
projection behavior.

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
- Single-visit verification and submission of one pending opponent move
  followed by one uniquely identifiable legal local reply from the final clear
  frame only.
- Current-position reconciliation that can adopt the authoritative opponent
  move and submit one uniquely identifiable legal local reply already present
  on the physical board.
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
Changing games shall not invalidate, redetect, or modify the established
physical-board orientation. A physical position retained from another game may
be compared with the selected game's current position and may enter the
existing reconciliation flow, but clock lifecycle events shall not alter the
coordinate mapping used by move guidance.

If the selected game is no longer present in the ongoing-games response,
polling shall retrieve one final authoritative snapshot by the remembered game
ID before clearing its polling cursor. Any final opponent move, including a
move that ends the game, shall enter the normal pending physical-move queue and
remain available for projection and confirmation. Only after that snapshot is
emitted may polling return to the selection-required state. If final-snapshot
retrieval fails or still reports an active game, the cursor shall remain intact
and polling shall retry rather than silently dropping the final move.

A terminal time-forfeit export may contain one more clock sample than chess
moves because the final sample records the flag fall without another move.
That sample shall update the clock of the side that was due to move and shall
not create a fictitious move or make the payload invalid.

The final authoritative snapshot shall also supersede an optimistic local move
that was submitted concurrently with game termination but is absent from the
official move history. Polling shall rebuild its relative move suffix from the
official history and its remembered baseline, remove the uncommitted local
move, and emit the terminal status, winner, result, and clocks. In particular,
a time-forfeit snapshot ending before the optimistic move shall be accepted as
authoritative rather than treated as an impossible backward transition.

After a terminal snapshot, the main Lichess status and retained current-game
information shall state the result and termination reason. A time forfeit shall
be presented explicitly, for example `White lost on time; result 0-1`, and
shall not be overwritten by the generic polling text `waiting for a game`.

## 6. Runtime States

### 6.1 `disconnected`

No active Lichess synchronization session exists.

### 6.2 `attaching`

An active game has been discovered, but the relationship between the remote
and physical positions has not yet been established.

### 6.3 `reconciliation-required`

The current remote position cannot be joined safely to the accepted physical
position.

No unexplained physical move shall be committed automatically in this state.
The unique legal-successor adoption defined in section 7.3.1 is the only
automatic local-move exception.

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

### 6.9 `compound-submission`

A clear, settled frame uniquely matches the pending opponent move followed by
one legal local reply, and that reply is being submitted to Lichess. The final
camera frame and compound result are provisional until Lichess confirms the
local move. Camera move inference remains paused.

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
occupancy stabilization, discard camera observations for one second while the
projector and exposure settle, reset stabilization again, and only then compare
stable live observations with the current authoritative Lichess piece map
rather than with a stale persisted physical map. This initial blackout wait is
required even when the preceding output contained only clocks or turn
indicators; projected light must not create a false first reconciliation grid.
**Reconcile physical board** remains available to restart or repeat this
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

#### 7.3.1 Reconciliation with an already executed local reply

Current-position reconciliation shall also handle a physical visit in which
the board was previously behind by one authoritative opponent move and the
player has already executed both that opponent move and one legal local reply.
The application shall not require the player to reconstruct the intermediate
opponent-only position when the final physical position proves one unique
reply safely.

The current server position remains authoritative. The application shall first
test the ordinary exact match described above. Only when that comparison has
stable discrepancies may it test legal successors of the current server
position, and only when all of the following are true:

- The active Lichess game is non-terminal.
- The server position says that it is the local player's turn.
- No submission, promotion-selection, correction, or other reconciliation
  acceptance is already pending.
- The projector is black, the camera exposure has settled for the required
  interval, the board is quiet, and no hand or obstruction is present.
- The server history still has the same game identity, move tuple, and current
  position used to generate the candidates.

The application shall enumerate every supported legal local move from the
current server position and compare each complete successor position with the
stable live occupancy and piece-color evidence. It shall use the same
discrepancy rules as ordinary current-position adoption. A candidate matches
only when every occupied and empty square agrees and no square has a
confidently detected opposite-color piece. The move identity, including any
promotion suffix, must be unique; several legal moves that are optically
indistinguishable remain ambiguous even when they produce the same occupancy
and color map.

When exactly one legal successor matches, the application shall:

1. Adopt the current server history and position, thereby accepting the
   server-confirmed opponent move without inferring or replaying it.
2. Initialize the synchronization controller at that server position.
3. Treat the uniquely matched successor as a provisional local reply and
   enter `compound-submission`.
4. Retain the clear final camera frame as a provisional reference and submit
   the exact local move to Lichess exactly once.
5. Commit the local legal position and accepted reference only after
   authoritative submission confirmation, then clear reconciliation marks and
   continue in `synchronized` or `game-complete`.

The existing uncertain-submission rules apply if the request fails after its
outcome may have reached Lichess. A definite rejection, changed server
history, invalid candidate, missing required promotion choice, or loss of the
clear settled frame shall commit neither the local reply nor its reference and
shall continue or restart reconciliation against the newest authoritative
server position.

When zero or more than one legal successor matches, the application shall
submit nothing, preserve the previous accepted physical state and reference,
and remain in `reconciliation-required`. Existing red discrepancy guidance
shall continue against the authoritative server position. Reconciliation
shall never invent a missing opponent move, choose a reply by score ordering,
or infer an unobservable promotion choice.

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

Because a terminal move removes the game from the ongoing-games endpoint, the
worker shall use the authenticated Board API game snapshot for exactly one
final refresh when the previously tracked game disappears. That snapshot shall
be processed before the polling cursor is cleared, preserving the terminal
move and result without relying on delayed public exports.

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

Physical move guidance shall be projected only after the complete White-side
board orientation is resolved. While orientation is unresolved, the logical
source and destination shall remain highlighted in the Physical position UI,
the projector shall remain black, and the pending move shall remain unmarked
as projected so that it can be rendered after orientation resolution.

Whenever the complete board orientation or active projector mapping changes,
pending guidance shall be regenerated from its logical move. For example,
`f8-d8` at a two-quarter-turn orientation shall be rendered through the
corresponding physical cells for that orientation; coordinates cached from a
temporary one-quarter-turn state shall never remain displayed as `h3-h5`.

Intent detection shall cover the calibrated 8x8 board plus a
perspective-aware one-square margin around every edge. Motion elsewhere in the
camera image shall be ignored. This margin ensures that a hand approaching a
piece on the nearest rank or an outer file starts manipulation tracking even
when only fingertips enter the 64-square board area.

The shared boundary detector may initiate blackout from the first confirmed
entry across the calibrated board edge. Its frames and scores shall never
update occupancy, infer a chess move, or replace the accepted reference.
Guidance blackout shall not wait for hand departure; after confirmed entry,
black-board tracking shall handle obstruction, departure, settling, and final
position verification. Without confirmed entry or manipulation, guidance
shall remain continuously visible.

### 10.2 Camera observation

After manipulation has started and then settled, the guidance projection shall
be removed and the projector returned to black. The application shall discard
at least 400 ms of frames, then compare consecutive downscaled black-projector
board observations for global luminance stability and per-square exposure
stability. Three consecutive stable observations may end settling early. Any
unstable observation resets that count. A one-second timer remains the
mandatory fallback and shall end settling when adaptive measurement is
unavailable or has not stabilized. Transient tracker activity shall be reset
when black settling completes, and only then may change scores, color
evidence, obstruction, or move settling be evaluated. The durable UI guidance
shall remain visible during physical confirmation. This adaptive timing shall
apply only to projected Lichess move guidance; startup, reconciliation,
illegal-correction, and other blackout workflows retain their existing timing.

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
against the original reference. Incoming castling guidance shall mark only the
king's source and destination from the authoritative Lichess move; the rook
source and destination shall remain part of verification but shall not be
projected or highlighted as guidance.

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

When authoritative game clocks are available, both edge clocks shall remain
visible during promotion choices and promotion confirmation. Promotion
projection shall not clear or suppress the clock state.

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

### 10.5 Single-visit opponent move and local reply

When exactly one opponent move is pending physically, the player may perform
both that authoritative opponent move and one legal local reply during one
continuous hand visit. The feature exists to avoid requiring the player to
remove the hand after the opponent move and then re-enter the board for the
reply.

#### 10.5.1 No interpretation under obstruction

The application shall not interpret, classify, infer, accept, reject, commit,
or submit any move while a hand, arm, or other material obstruction remains
over the board. In particular, it shall not:

- attempt to identify the opponent move as an intermediate position;
- infer the local reply from partial square changes;
- update the physical or UI position from an obstructed frame;
- use a momentary gap between actions as an intermediate observation; or
- submit a local move before a clear, settled final frame exists.

Confirmed boundary entry shall use the existing guidance-blackout lifecycle.
All frames acquired from entry until confirmed departure and settling are
motion and obstruction evidence only. They shall never be chess-position
evidence.

The camera is required to observe only:

1. the accepted reference before hand entry; and
2. one unobstructed, stable final position after hand departure.

No intermediate camera image is required or authoritative.

The projected side-to-move dot shall remain steadily visible throughout
guidance, blackout, settling, compound verification, submission, correction,
and reconciliation. These temporary gameplay states shall not blink or
suppress it. The application may delay changing sides until the newly
authoritative turn is ready for presentation. Calibration output, unresolved
turn authority, a disabled turn-indicator setting, and game completion may
suppress the dot.

While one or more authoritative server moves remain pending physically, the
turn indicator shall show the side to move at the physical cursor rather than
the newer server cursor. For a single-visit compound action, it therefore
continues to show the opponent's pending side while the player performs the
opponent move and local reply. After the final physical result is committed,
the indicator shall change directly to the side to move in that final result.
It shall not briefly display the intermediate server side.

#### 10.5.2 Candidate construction

Compound verification is eligible only when:

- exactly one supported opponent move is active in the pending queue;
- applying that move legally makes it the configured local player's turn;
- the opponent move does not end the game;
- no local submission, correction, promotion selection, reconciliation, or
  calibration operation is active; and
- the accepted physical reference still represents the position immediately
  before the opponent move.

The application shall apply the authoritative opponent move to a temporary
legal game and enumerate every currently supported legal local reply from that
intermediate position. Each compound candidate shall contain:

- the required opponent signature and result;
- the local reply signature, UCI, and result;
- the complete final logical position after both plies;
- the net expected physical changes relative to the accepted reference; and
- the complete set of squares on which residual visual change is permitted.

The ordinary opponent-only result shall remain a separate candidate so the
existing one-move workflow continues to work when the player removes the hand
after performing only the opponent move.

#### 10.5.3 Final-frame verification

After the board is clear and settled, the application shall evaluate the final
frame against the required opponent-only result and all eligible compound
results. Verification shall use complete result-state evidence rather than
assuming that both component relocations remain individually visible.

This is necessary when the local reply:

- moves or captures the opponent's just-moved piece;
- immediately recaptures on the opponent move's destination;
- uses a square changed by both plies;
- completes castling or en passant; or
- otherwise causes the two plies' change masks to overlap.

A compound result may be accepted only when exactly one legal local reply
explains the complete final board and no unexplained change exists outside its
net expected-change mask. The known opponent move shall remain mandatory; a
final position reachable only by omitting or replacing that move shall not be
accepted.

When exactly one compound result matches the complete final board, an
opponent-only match covering a strict subset of the same changed squares shall
not make the result ambiguous and shall not be committed instead. The unique
compound result takes precedence because it explains the additional local-reply
evidence. If more than one compound result matches, the result remains
ambiguous.

Occupancy-equivalent or color-equivalent candidates that the camera cannot
distinguish shall remain ambiguous even if one seems more likely. The
application shall never choose among compound replies by score ordering,
timing, move popularity, or engine preference.

#### 10.5.4 Outcomes

The settled final frame shall produce exactly one of these outcomes:

1. **Opponent-only match:** confirm the opponent move through the existing
   workflow, commit its reference, and then permit ordinary local-move
   detection.
2. **Unique compound match:** retain the final frame and two-ply result
   provisionally, confirm the physical opponent move in the synchronization
   controller, enter `compound-submission`, and submit the uniquely identified
   local reply once.
3. **Ambiguous compound match:** preserve the prior accepted reference and
   physical state, submit nothing, and require the player to restore the
   previous physical position or enter explicit reconciliation.
4. **No valid match:** preserve the prior accepted reference and physical
   state and use existing correction or reconciliation behavior.

The opponent and local move shall be displayed as two ordered plies. They
shall not be collapsed into one synthetic chess move.

#### 10.5.4.1 Incorrect compound attempts

Correction shall distinguish whether the required opponent move itself is
proven by the settled final frame. The application shall not reject the whole
visit merely because the subsequent local action is illegal or does not match
any legal reply.

When the required opponent move is proven but the subsequent local action is
an unambiguous incorrect relocation:

- the opponent move shall be committed to the physical logical position;
- the incorrect local action shall not be submitted or committed;
- only the incorrect local action's origin and destination shall be marked
  red on the physical-position UI and projector, using the ordinary illegal
  correction behavior;
- the pre-visit camera reference shall remain authoritative while correction
  is active because the incorrect final frame is not a valid opponent-only
  reference; and
- after the incorrect local action is undone and a clear settled frame proves
  the opponent-only result, that frame shall become the accepted reference and
  normal local-move tracking shall resume.

When the required opponent move is incorrect, missing, or cannot be proven
independently of the subsequent action:

- neither physical ply shall be committed;
- no local move shall be submitted;
- the pre-visit logical position and camera reference shall remain
  authoritative;
- the known opponent move's origin and destination and the unambiguous
  subsequent relocation's origin and destination shall all be marked red,
  with duplicate or overlapping squares shown only once; and
- the player shall restore the complete pre-visit accepted position before
  the pending opponent guidance resumes.

The combined correction therefore marks at most four distinct origin or
destination squares for ordinary two-ply attempts, and fewer when squares
overlap. If the subsequent relocation cannot be identified unambiguously, the
application shall mark every unambiguous origin or destination from the failed
attempt and shall not guess the remaining squares. Castling, en passant, and
promotion correction shall continue to include any additional required
special-move correction squares defined by their existing workflows.

No logical move or camera reference may be committed from a frame in which the
required opponent move is unproven. Ambiguity about whether the opponent move
was executed correctly shall use the second, non-committing outcome.

#### 10.5.5 Submission and commit

Compound local submission shall retain the existing network two-phase commit:

- The settled final image, local UCI, and complete two-ply result remain
  provisional while the POST and authoritative refresh are pending.
- Camera inference remains paused throughout submission.
- Lichess confirmation of the expected post-reply FEN atomically commits the
  two-ply physical position, final UI position, preserved final camera
  reference, polling cursor, and durable state.
- The final reference shall be captured from the already settled unobstructed
  frame. The application shall not ask the player to recreate the intermediate
  opponent-only position after successful submission.
- An explicit rejection shall not pretend that the observed final board is
  synchronized. The opponent move remains authoritative; the player shall be
  instructed to undo the local reply to reach the opponent-only position.
- Because no intermediate camera reference exists, rejection recovery shall
  verify the opponent-only result geometrically against the original reference
  before capturing it as the new accepted reference.
- An uncertain response, changed authoritative history, or inability to prove
  opponent-only restoration shall enter reconciliation.

The application shall never retry the local reply automatically.

#### 10.5.6 Supported and excluded compound moves

Compound candidates may use the move kinds already supported for both pending
opponent verification and local submission: ordinary moves, captures,
castling, en passant, auto-queen promotion, and auto-queen
capture-promotion.

Select-promoted-piece mode shall continue to allow compound local replies that
do not promote. Only a compound local promotion or capture-promotion is
excluded in that mode because its projected virtual-rank interaction requires
a separate settled selection and replacement workflow. A pending queue
containing more than one unconfirmed server move is also excluded. Excluded
replies and queues shall fall back to the existing sequential workflow.

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

Inter-frame motion shall gate diagnostic, legal-signature, and compound
tracking alike. A moving hand frame may update motion and obstruction state but
shall not be classified, matched, accepted, rejected, or used as a reference.
After motion ends, settled observations, large-obstruction rejection,
released-piece-footprint checks, and exact signature evidence shall verify the
clear final position.

When all changed-square occupancy and piece colors confidently match one legal
final result, a weak photometric score at a destination shall not discard that
result. Source departure shall retain a minimum change requirement, while the
destination's exact final-state classification supplies the missing evidence.
This applies to compound results such as `e7-e5` followed by `f2-f3`, where the
destination pawn may be visually similar to its new square.

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

This includes an illegal capture in which a piece leaves its occupied source
and replaces an opposing piece on an occupied destination without matching
any legal physical signature. The source and occupied destination shall be
reported as an illegal relocation so both squares receive the same durable red
correction used for other incorrect moves.

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

While red correction is projected, the application shall use the shared
calibrated boundary detector to begin blackout immediately when a hand enters.
A generation-guarded two-second fallback shall begin blackout when entry is
missed. Boundary observation shall be suspended before changing to black so a
hand-containing frame cannot become its baseline. After the fixed
illegal-correction blackout settling interval, the ordinary black-board
reference tracker shall verify restoration. If restoration is incomplete, red
correction shall be reprojected and boundary observation rearmed against that
static projected scene. If restoration is complete, correction state shall
clear and boundary observation shall rearm against the restored black-board
scene.

After restoration, the application shall resume guidance for the same pending
move. After an exact expected final position, it may commit normally.

## 13. Local Supported-Move Submission

Local submission is available only when the physical and server positions are
synchronized, it is the configured player's turn, and the camera identifies a
unique ordinary move, non-promotion capture, completed castling signature, or
completed en passant signature, queen promotion, or queen capture-promotion.
Section 10.5 is the sole exception to the initial synchronization precondition:
it may identify the local reply from a unique final two-ply result after first
confirming the one authoritative pending opponent move.

The application shall use a two-phase commit:

1. Preserve the settled candidate image without changing the accepted
   reference, physical position, server position, UI position, or turn.
2. Submit the UCI move once with
   `POST /api/board/game/{gameId}/move/{move}`.
3. After a successful response, fetch the authenticated ongoing-game position.
   Advance the controller's server and physical positions, commit the preserved
   image as the accepted reference, update the UI, persist synchronized state,
   and display the accepted move only when its FEN equals the expected
   post-move FEN. If the game is no longer ongoing, use the terminal export for
   confirmation instead.
4. After an explicit HTTP rejection, preserve the previous authoritative state
   and reference and require the physical move to be restored.

The application shall pause further camera move inference while submission is
pending. It shall never retry a move automatically.

After every POST response, including an apparently successful response, the
submission worker shall fetch the authenticated ongoing-game position:

- If its FEN equals the expected post-move FEN, treat the move as accepted and
  commit it once.
- If its FEN still equals the previously confirmed FEN, report an uncertain
  active result without committing either state.
- If the game is absent, fetch the terminal export and use its official FEN,
  history, status, and result to confirm or reject the move.
- If either refresh fails or the position has another shape, enter
  reconciliation and preserve both the previous reference and the pending
  physical discrepancy.

Polling fallback remains capable of submitting moves. Polling affects only how
authoritative game updates are received; it does not make a valid Board API
session read-only.

After Lichess accepts a local submission, the application shall advance the
polling cursor through that move after verifying that it transforms the
cursor's prior FEN into the committed server FEN. This update shall occur
before a later opponent poll is interpreted, so an immediate reply reported as
`lastMove` remains a verifiable one-ply transition rather than appearing to
shorten or rewrite history. If the ongoing-games endpoint temporarily continues
to report the position immediately before that confirmed local move, polling
shall retain the advanced cursor rather than resetting it backward. A
same-position export may then enrich the retained position with clocks, and
polling shall continue until the ongoing endpoint catches up. Only an
authoritative terminal history may roll back a locally submitted move.

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
- The active physical side to move using both a color indicator and visible
  text, while the Lichess panel separately retains the authoritative server
  side to move.
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
accumulated, and event history shall be bounded. Routine move-tracker
observations shall not write images. Significant physical decisions shall
queue immutable state and image snapshots to a bounded single-writer
background queue so diagnostic compression and file output cannot delay
physical move acceptance. Queue overflow or writer failure shall be reported
in the runtime UI.

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

### 19.6 Single-visit opponent move and local reply

- Frames containing a hand or obstruction never produce a chess move,
  intermediate position, local submission, or reference update.
- A clear final frame matching only the opponent move uses the existing
  opponent-only workflow.
- A clear final frame uniquely matching the required opponent move plus one
  legal local reply submits that reply exactly once.
- A unique compound result is selected when the opponent-only signature also
  passes as a strict subset; score fluctuation shall not later downgrade that
  final board to the opponent-only result.
- Immediate recapture and overlapping-square sequences are matched from their
  complete final result rather than requiring the intermediate board.
- Two legal replies with indistinguishable final camera evidence submit
  nothing and report ambiguity.
- An extra unexplained changed square rejects the compound candidate.
- A proven opponent move followed by one unambiguous incorrect local
  relocation commits only the opponent move, marks the incorrect relocation's
  origin and destination red, and accepts a new reference only after the
  incorrect action is undone.
- An incorrect or unproven opponent move commits neither ply and marks the
  deduplicated origin and destination squares of both identifiable
  relocations red.
- Checkmate or another terminal opponent result offers no compound reply.
- Multiple pending opponent moves use the sequential workflow.
- In select-promoted-piece mode, ordinary, capture, castling, and en-passant
  compound replies remain eligible; a local promotion or capture-promotion
  uses the sequential promotion workflow.
- Explicit local rejection requests restoration to the opponent-only result;
  uncertain submission enters reconciliation.

### 19.7 Reconciliation with a local reply

- An exact physical match with the server position uses ordinary
  current-position adoption before any legal successor is considered.
- When the accepted physical state is one opponent ply behind, a clear final
  board containing that server-confirmed move plus one uniquely matching legal
  local reply attaches at the server position and submits only the local reply.
- Immediate recapture is accepted when its complete final position uniquely
  matches a legal successor, even though the intermediate opponent-only
  position was never observed.
- Zero matching successors retain reconciliation and submit nothing.
- Two optically indistinguishable legal successors retain reconciliation and
  submit nothing.
- Promotion candidates remain ambiguous unless the exact suffix is uniquely
  established by the configured promotion workflow.
- A hand, obstruction, unsettled frame, projected light, or server-history
  change prevents successor adoption.
- A confirmed submission commits the final legal state and captured reference;
  a definite rejection preserves the previous accepted state and reference,
  while an uncertain outcome follows the existing submission-recovery rules.

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
- Incoming castling uses two-square king-move guidance while verification
  still accepts only the complete four-square final position or king-first,
  rook-second physical handling.
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
- Single-visit opponent-plus-local handling generates complete two-ply legal
  candidates, interprets only the clear settled final frame, blocks acceptance
  until both move footprints are released, submits only a unique reply, and
  atomically commits both plies after authoritative Lichess confirmation.

The current validated automated suite contains 455 passing tests. Live
ordinary-move, polling, reconciliation, and guidance acceptance has been
exercised on the target camera/projector setup. Lichess capture, castling, en
passant, and auto-queen behavior is automated-test validated and awaits live
acceptance.
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
guidance remains staged. Single-visit opponent-plus-local movement is
automated-test validated and awaits live acceptance. Select-promoted-piece mode
retains single-visit ordinary, capture, castling, and en-passant replies while
reserving local promotion replies for the sequential selection workflow.
Incorrect compound attempts now distinguish a proven opponent ply from an
incorrect or unproven one, preserve the appropriate authoritative reference,
and project the specified two- or four-square red correction; this correction
workflow is automated-test validated and awaits live acceptance.
Reconciliation of a server-confirmed opponent move plus an already executed
unique legal local reply is implemented and automated-test validated according
to section 7.3.1; live acceptance remains pending.
