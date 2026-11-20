# Lichess Physical Move Synchronization Specification

**Version:** 0.2
**Status:** Incoming and outgoing ordinary-move slices implemented; live acceptance pending
**Platform:** Python on Windows

**Related specifications:**

- [Lichess physical-board integration roadmap](./lichess-integration-roadmap.md)
- [Reference-image move tracking](./reference-image-move-tracking-specification.md)
- [Chess-piece UI rendering and move synchronization](./chess-piece-ui-tracking-specification.md)
- [Projector/camera chessboard system](./projector-camera-chessboard-specification.md)

## 1. Purpose

This specification defines the ordinary-move slices of Phases 7 and 8 of the
Lichess physical-board integration: synchronizing moves already accepted by
Lichess onto the physical board and submitting local physical moves.

When Lichess advances the remote game, the application shall:

1. Advance the authoritative server and Lichess UI positions.
2. Keep the accepted physical position and camera reference unchanged.
3. Queue each remote move not yet confirmed physically.
4. Project guidance for the first pending move.
5. Verify the expected physical change with the camera.
6. Advance the physical position and replace the camera reference only after
   exact confirmation.

The first outgoing slice submits ordinary non-capturing moves only. Captures,
castling, en passant, and promotion remain reconciliation-only until their
outgoing slices are implemented.

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

## 3. Scope

### 3.1 First implementation

The first implementation shall support:

- A standard Lichess game that starts from the normal initial position.
- Streamed or read-only-polled Lichess snapshots.
- Complete move-history prefix validation.
- One or more newly reported moves in a single update.
- An ordered queue of moves not yet confirmed physically.
- Ordinary moves for both colors.
- Browser-first reproduction of the local player's moves.
- Opponent-move projection and physical confirmation.
- Durable UI guidance.
- Incorrect physical-action correction.
- Preservation of the accepted physical reference until confirmation.
- Reconnection while a move remains pending.
- Explicit reconciliation when remote and physical positions cannot be joined
  safely.

### 3.2 Staged extensions

After ordinary moves are live-accepted, the same state machine shall be
extended to:

- Captures.
- Castling.
- En passant.
- Queen promotion.
- Capture-promotion.

The legal-signature model already represents these move kinds. Their rollout
may be staged because projection and multi-action physical workflows require
additional acceptance testing.

### 3.3 Excluded

Phase 7 shall not:

- Submit moves to Lichess.
- Create or accept challenges.
- Offer or accept draws.
- Abort or resign games.
- Infer missing server moves from a camera image.
- Automatically assign piece identity from occupancy alone.
- Accept a physical move that is absent from Lichess history.
- Support underpromotion selection.

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

### 7.2 Unsafe attachment

If the positions differ and there is no verified move-history relationship,
the application shall enter `reconciliation-required`.

It shall not:

- Queue the entire historical game automatically.
- Replace the accepted reference with the latest camera frame.
- Infer piece identity from occupancy.
- Assume that a matching occupied-square set means the positions are equal.

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

Polling or reconnection may reveal multiple new plies.

They shall be queued and confirmed physically in server order. Only the first
pending move may be active in the projector and camera workflow.

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

For an ordinary move, the projector shall identify:

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

### 10.2 Camera observation

The guidance projection shall be removed and the projector returned to black
before the camera evaluates change scores, color evidence, obstruction, or
settling.

Guidance may be shown again when:

- No physical activity was observed.
- The user requests the instruction again.
- A correction was restored and the expected move remains pending.

### 10.3 Special-move projection

Later slices shall derive involved squares from the expected signature:

- Capture: source and destination.
- Castling: king source/destination and rook source/destination.
- En passant: source, destination, and captured-pawn square.
- Promotion: source and destination plus queen-replacement guidance.

## 11. Camera Confirmation

### 11.1 Exact expected match

The existing reference tracker shall evaluate the settled observation against
the active expected signature.

Acceptance shall require:

- All expected changed squares satisfy their evidence requirements.
- Source vacancy is confirmed.
- Destination occupancy and mover color are confirmed where required.
- No unexplained changed square exceeds the allowed residual policy.
- The board is unobstructed and settled.

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

### 12.2 Isolated removal

An isolated settled piece removal remains illegal unless it is a recognized
first step of the active expected capture workflow.

### 12.3 Correction completion

The user may:

- Restore the previous physical position; or
- Complete a final board position that exactly matches the expected signature.

After restoration, the application shall resume guidance for the same pending
move. After an exact expected final position, it may commit normally.

## 13. Local Ordinary-Move Submission

Local submission is available only when the physical and server positions are
synchronized, it is the configured player's turn, and the camera identifies a
unique ordinary non-capturing legal move.

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

The read-only polling fallback may discover several moves at once. The same
prefix validation and ordered queue shall apply.

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
move-tracker decisions. Events may carry token-free operation details needed
to reconstruct the failure context. Runtime files shall remain excluded from
version control, image files shall be overwritten rather than accumulated,
and event history shall be bounded.

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

The first ordinary-move slice is implemented:

- A pure synchronization controller validates complete history prefixes.
- Server and physical legal cursors advance independently.
- Up to two moves discovered during initial fresh-game attachment may queue.
- Later polling updates may append multiple moves in server order.
- The Lichess board advances immediately while the physical board remains
  unchanged.
- The active ordinary signature is the only legal signature offered to the
  reference tracker.
- Guidance is projected for two seconds and the projector returns to black
  before camera evaluation resumes.
- Correct physical evidence commits the physical position and reference.
- Premature and incorrect physical moves enter durable red correction without
  changing the authoritative state.
- Captures and special moves explicitly enter reconciliation until their staged
  slices are implemented.

Live acceptance and restart persistence remain pending.
