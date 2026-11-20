# Lichess Physical-Board Integration Roadmap

**Version:** 0.1  
**Status:** Planning  
**Platform:** Python on Windows  

**Related specifications:**

- [Reference-image move tracking](./reference-image-move-tracking-specification.md)
- [Chess-piece UI rendering and move synchronization](./chess-piece-ui-tracking-specification.md)
- [Chess-square occupancy and UI marking](./chess-square-occupancy-ui-specification.md)
- [Projector/camera chessboard system](./projector-camera-chessboard-specification.md)

## 1. Goal

Enable a player to complete a normal game on Lichess using:

- A physical chessboard as the local move-input device.
- A camera to detect and verify physical moves.
- A projector to indicate remote moves and synchronization errors.
- The desktop UI to display the authoritative game position, clocks, status,
  and required user actions.
- The Lichess Board API as the remote game service.

The implementation shall support all legal chess moves, recover safely from
ambiguous physical changes and network failures, and never corrupt the
accepted physical reference or authoritative game position.

## 2. Guiding Principle

The move detector shall not first infer an arbitrary relocation and then ask
whether it is legal. It shall instead ask:

> Which currently legal move best explains the observed physical changes?

For every legal move, the application shall generate an expected physical
change signature and score the settled camera evidence against that signature.
The move may be accepted only when one legal move explains the evidence
unambiguously.

The camera layer shall remain responsible for visual evidence. A separate
chess-rules and game-control layer shall be responsible for legality, turn
state, and interaction with Lichess.

## 3. Authoritative State Model

The completed application shall distinguish three positions.

### 3.1 Server position

The position reported by Lichess. During a connected game, this is the
ultimate authority.

### 3.2 Logical/UI position

The position displayed by the application. It normally matches the server
position but may temporarily advance before the physical board when an
opponent move arrives.

### 3.3 Physical/reference position

The last camera-confirmed physical position and its accepted canonical
reference image.

These positions are usually equal. They may differ temporarily while:

- The user is reproducing an opponent move.
- A local move is awaiting server acceptance.
- The application is reconnecting.
- The physical board requires reconciliation.

The states shall be represented explicitly rather than overloading one
position object.

## 4. Phase 1: Legal Chess Model

Use `python-chess` as the chess-rules engine.

**Implementation status:** Implemented. The application now provides
diagnostic movement and offline legal-game modes. The persisted offline game
stores its full FEN and is restored only when its piece map matches the saved
logical position.

The legal game model shall provide:

- Current board position.
- Side to move.
- Legal move generation.
- Check, checkmate, stalemate, and draw state.
- Castling rights.
- En passant state.
- Half-move and full-move counters.
- FEN import and export.
- UCI move parsing and formatting.
- Promotion choices.
- Move application and rollback.

The existing visual piece dictionary shall remain useful for rendering and
physical-state comparison, but shall not be the authority for legality.

### 4.1 Game modes

The application should distinguish:

- `diagnostic`: arbitrary one-piece relocation may be accepted for camera
  testing.
- `offline-game`: only legal chess moves may be accepted.
- `lichess-game`: only moves legal in the authoritative server position may be
  submitted.

Offline legal mode displays a persistent black or white square for the
authoritative side to move. The square is independent of transient camera and
alignment messages, is hidden in diagnostic mode, and becomes gray when the
legal game is complete.

### 4.2 Phase 1 acceptance gate

- Legal moves are accepted in offline-game mode.
- Illegal physical actions are rejected.
- Rejection changes neither the logical position nor the accepted reference.
- The UI reports why no legal move matched and tells the user to move the
  piece directly to a legal destination or restore the previous position.
- The authoritative source piece remains at its logical square and is marked
  red on the UI. A calibrated projector briefly illuminates the physically
  observed incorrect destination in red and the authoritative source in white
  before returning to black.
- Diagnostic mode can remain legality-independent.

Implemented. Physical capture, castling, en passant, and promotion evidence
remain Phase 2 through Phase 4 work; the rules authority already generates and
applies those move types.

## 5. Phase 2: Legal-Move Physical Signatures

For each currently legal move, generate the physical changes expected after
the move is complete.

### 5.1 Ordinary move

For `e2e4`:

- `e2` becomes empty.
- `e4` becomes occupied by White.

### 5.2 Capture

For `f3e5`:

- `f3` becomes empty.
- `e5` remains occupied.
- The piece-color appearance on `e5` changes from Black to White.

### 5.3 En passant

For White playing `e5d6`:

- `e5` becomes empty.
- `d6` becomes occupied by White.
- `d5` becomes empty.

### 5.4 Castling

For White kingside castling:

- `e1` becomes empty.
- `g1` becomes occupied by the king.
- `h1` becomes empty.
- `f1` becomes occupied by the rook.

### 5.5 Promotion

For `e7e8q`:

- `e7` becomes empty.
- `e8` becomes occupied by White.
- The user selects the promotion type because exact physical piece-type
  recognition is not required.

The visual matcher shall score all expected changed squares, vacancy evidence,
piece-color evidence, unexpected third-square changes, and candidate
dominance. It shall accept only a unique best legal explanation.

## 6. Phase 3: Physical Piece-Color Classification

The camera does not need to distinguish knights from bishops. It needs a
conservative per-square classification:

- Empty.
- White piece.
- Black piece.
- Uncertain.

The classifier should use known logical piece colors and the accepted
reference as calibration evidence.

Capture acceptance requires:

1. A known source piece.
2. A logically occupied destination containing the opposing color.
3. Strong source-vacancy evidence.
4. Continued destination occupancy.
5. Destination appearance changing to the mover's color.
6. Dominant reference differences on the expected squares.
7. A unique matching legal move.

Uncertain color evidence shall preserve the previous position and reference.

## 7. Phase 4: Complete Physical Move Workflows

### 7.1 Captures

Support both:

- Removing the captured piece and moving the attacker in one continuous
  action.
- Removing the captured piece, withdrawing the hand, and moving the attacker
  in a second action.

A destination-only removal is an incomplete capture. It shall not replace the
reference. Later evidence shall still be compared with the original accepted
reference until the capture is completed or reconciliation is requested.

### 7.2 Castling

The initial physical interaction contract shall require moving the king first
and the rook second.

The king move signals castling intent. The application shall wait for all four
expected square changes before committing. Moving the rook first can be
ambiguous with an ordinary legal rook move and should not be the initial
supported workflow.

### 7.3 En passant

The application shall wait for all three expected square changes before
committing the move.

### 7.4 Promotion

After detecting the pawn relocation:

1. Enter a pending-promotion state.
2. Ask the user to select queen, rook, bishop, or knight.
3. Apply the selected promotion.
4. Commit the logical position and reference together.

Queen may be preselected but shall not be committed without confirmation.

### 7.5 Phase 4 acceptance gate

Complete an offline game containing:

- Ordinary moves for both colors.
- Captures by both colors.
- A capture performed as two physical actions.
- Kingside and queenside castling.
- En passant.
- Promotion and capture-promotion.
- Check and checkmate.
- Deliberate illegal physical actions.

## 8. Phase 5: Physical-Board Reconciliation

Reconciliation shall be completed before network move submission.

When physical occupancy does not match the authoritative position:

1. Switch the projector to black.
2. Capture stable absolute occupancy.
3. Compare physical occupancy with the authoritative logical occupancy.
4. Illuminate only mismatching physical squares.
5. Return the projector to black before camera verification.
6. Repeat until occupancy matches.
7. Capture and persist a new accepted reference.

The same discrepancy color may be used for:

- Expected occupied but physically empty.
- Expected empty but physically occupied.

The UI shall identify the expected pieces and squares.

Occupancy cannot detect two pieces exchanged between occupied squares. Piece
color may detect opposite-color swaps, but exact piece identity remains
dependent on accepted move history and the legal position.

## 9. Phase 6: Read-Only Lichess Connectivity

Use the official Lichess Board API, which is intended for physical boards and
third-party clients using normal Lichess accounts.

Initial authentication may use a personal access token with only the required
scopes. The token shall:

- Never be committed.
- Never appear in diagnostics or logs.
- Be redacted from errors.
- Be loaded from an environment variable or secure operating-system credential
  storage.

All requests shall include an identifying `User-Agent`.

Implement:

- Account-event NDJSON stream.
- Active-game stream.
- Initial FEN and complete move list.
- Player names and colors.
- Clock values.
- Side to move.
- Game status and result.
- Reconnection with bounded backoff.
- Lichess rate-limit handling.

The first networking milestone shall observe games without submitting moves.

Official API reference:
<https://github.com/lichess-org/api/blob/master/doc/specs/lichess-api.yaml>

### 9.1 Phase 6 acceptance gate

The application follows a live game and reconstructs exactly the same position
as Lichess after every streamed event and after reconnecting.

## 10. Phase 7: Incoming Opponent Moves

When Lichess reports an opponent move:

1. Parse the UCI move against the current server position.
2. Advance the server and UI positions.
3. Keep the physical/reference position unchanged.
4. Project all involved physical squares.
5. Wait for the expected physical change signature.
6. Keep the projector black while evaluating camera frames.
7. Commit the physical position and reference after exact confirmation.
8. Enable local move detection when it becomes the player's turn.

Projection shall cover:

- Source and destination for ordinary moves and captures.
- King and rook source/destination squares for castling.
- Source, destination, and captured-pawn square for en passant.

Because the incoming move is known, the camera verifies one expected
signature rather than inferring among all legal moves.

## 11. Phase 8: Submit Local Moves

Local move submission requires a two-phase commit.

### 11.1 Prepare

1. The camera produces one pending legal candidate.
2. Preserve the settled candidate image.
3. Do not yet change the accepted logical position or reference.
4. Submit the UCI move to Lichess.

### 11.2 Commit

After Lichess accepts the move:

- Advance the server position.
- Advance the logical/UI position.
- Commit the pending candidate image as the new reference.
- Persist the synchronized game state.

### 11.3 Reject

If Lichess rejects the move or the request outcome is uncertain:

- Do not advance the authoritative position.
- Do not replace the accepted reference.
- Refresh the server move stream.
- Determine whether the move was already accepted.
- Enter reconciliation when the physical and server positions differ.

The application shall not retry a move blindly after an uncertain network
outcome.

## 12. Phase 9: Game Controls and Lifecycle

Add UI support for:

- Connect and disconnect.
- Active-game selection.
- Challenge notification.
- Accept and decline challenge.
- Player color.
- Player names.
- Clocks.
- Draw offer and response.
- Resign.
- Abort when permitted.
- Game result.
- Reconnection status.
- Return to setup after game completion.

For the first playable version, challenge creation may remain on the Lichess
website. The application may initially detect and join an already active game.

## 13. Diagnostics

Diagnostics shall include:

- Server game ID without authentication credentials.
- Server, logical, and physical positions.
- FEN for each position.
- Side to move.
- Legal moves considered.
- Expected signature for each leading candidate.
- Per-square change and color scores.
- Selected move and runner-up margin.
- Pending network operation without token or authorization headers.
- Lichess response status.
- Reference commit or preservation decision.
- Reconnection and reconciliation transitions.

Credentials and complete authorization headers shall never be recorded.

## 14. Test Strategy

### 14.1 Chess rules

- Legal and illegal ordinary moves.
- Moves blocked by check.
- Pins.
- Captures.
- Both castling sides and loss of castling rights.
- En passant timing.
- All four promotion choices.
- Checkmate, stalemate, repetition, and insufficient material.
- FEN and UCI round trips.

### 14.2 Physical evidence

- Ordinary source/destination localization.
- White and Black captures.
- One-action and two-action captures.
- Same-color piece/square combinations.
- Tall pieces and shadows.
- Castling with king-first workflow.
- En passant three-square signature.
- Promotion selection.
- Incomplete actions preserve the reference.
- Illegal actions preserve the reference.

### 14.3 Lichess protocol

- NDJSON split across arbitrary network chunks.
- Initial full game event.
- Incremental game-state event.
- Clock updates.
- Move acceptance and rejection.
- Rate limiting.
- Connection loss and reconnect.
- Duplicate or delayed events.
- Uncertain POST outcome.
- Completed, aborted, and resigned games.

### 14.4 End-to-end

- Play a complete casual game.
- Include at least one capture.
- Include castling.
- Exercise promotion or its complete tested workflow.
- Reject an illegal physical move.
- Confirm an opponent move physically.
- Recover from temporary network loss.
- Restart the application during a game.
- Reconcile a mismatching physical board.
- Finish with the same final position and result as Lichess.

## 15. Definition of Ready for Lichess Play

The system is ready for normal Lichess use when:

1. All legal move types have verified physical workflows.
2. Illegal actions cannot change the authoritative position or reference.
3. Server, logical, and physical positions are represented separately.
4. Incoming moves are projected and physically confirmed.
5. Local moves are committed only after server acceptance.
6. Captures, castling, en passant, and promotion are supported.
7. Reconnection cannot duplicate or lose a move.
8. Restart restores or safely reconstructs the active game.
9. Reconciliation identifies physical occupancy discrepancies.
10. Authentication credentials remain secure and absent from diagnostics.
11. A complete casual game can be played without manually correcting internal
    application state.

## 16. Recommended Implementation Order

1. Add `python-chess` and the legal position authority.
2. Change reference tracking to score legal move signatures.
3. Implement physical captures and piece-color evidence.
4. Implement castling, en passant, and promotion.
5. Complete a fully legal offline game.
6. Implement physical/server reconciliation.
7. Add read-only Lichess event and game streaming.
8. Project and confirm opponent moves.
9. Submit local moves using two-phase commit.
10. Add clocks, game controls, restart recovery, and reconnection hardening.

