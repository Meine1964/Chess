# Lichess Physical-Board Integration Roadmap

**Version:** 0.6
**Status:** Phases 2-6 live-accepted
**Platform:** Python on Windows  

**Related specifications:**

- [Reference-image move tracking](./reference-image-move-tracking-specification.md)
- [Chess-piece UI rendering and move synchronization](./chess-piece-ui-tracking-specification.md)
- [Chess-square occupancy and UI marking](./chess-square-occupancy-ui-specification.md)
- [Projector/camera chessboard system](./projector-camera-chessboard-specification.md)
- [Lichess physical move synchronization](./lichess-physical-move-synchronization-specification.md)

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
It shall also generate queen-only pseudo-legal signatures for physical
candidate detection. Pseudo-legal signatures obey piece movement and occupancy
rules but may leave the moving side's king in check. They exist only to identify
a completed physical action for correction guidance; the move may be accepted
only when the rules authority confirms that the matched candidate is legal.

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
  observed incorrect destination and the authoritative source in red before
  returning to black.
- Diagnostic mode can remain legality-independent.

Implemented. Physical capture, castling, en passant, and promotion evidence
remain Phase 2 through Phase 4 work; the rules authority already generates and
applies those move types.

Phase 1 also includes the implemented ordinary-move reliability layer:

- Candidate source/destination squares are localized before general obstruction
  rejection, so a large released king or queen is not itself classified as a
  hand.
- Released-piece footprints allow the largest tested chess pieces while
  retaining the hand-sized rejection bound.
- A high-scoring adjacent square can be explained only when it contains an
  opposite-color logical piece whose lower/base color signature still matches
  the accepted reference.
- Absolute occupancy remains available for initialization and reset, but does
  not run as a competing authoritative move tracker.

## 5. Phase 2: Legal-Move Physical Signatures

For each currently legal move, generate the physical changes expected after
the move is complete.

**Implementation status:** Signature generation is implemented. Every legal
move exposes a typed signature containing UCI, SAN, move kind, source,
destination, promotion choice, and the exact before/after piece value for each
changed square. Signatures are derived by applying each legal move to a
temporary `python-chess` board and diffing its piece map, so the rules are not
duplicated and the authoritative game remains unchanged.

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
- The initial implementation assumes promotion to a queen.
- Rook, bishop, and knight underpromotion remain future extensions.

The visual matcher shall score all expected changed squares, vacancy evidence,
piece-color evidence, unexpected third-square changes, and candidate
dominance. It shall accept only a unique best legal explanation.

The implemented Phase 1 adjacent-residual check is reusable evidence for this
phase, but it is not a complete capture or per-square color classifier. Phase 2
shall generalize that targeted reference evidence into legal physical
signatures.

Implemented signature kinds are:

- `ordinary`
- `capture`
- `castling`
- `en-passant`
- `promotion`
- `capture-promotion`

Focused tests verify ordinary source/destination transitions, destination color
replacement on capture, both rook transitions during castling, the separate
captured-pawn square for en passant, queen promotion, and queen
capture-promotion.

### 5.6 Camera-side legal-signature matching

For every settled observation, the matcher shall:

1. Read the per-square reference-change scores.
2. Score every currently legal physical signature.
3. Require every expected changed square to exceed the normal movement
   threshold.
4. Penalize the strongest unexplained changed square outside the signature.
5. Accept only one best signature separated from the runner-up by the configured
   dominance margin.
6. Generate only the queen-promotion signature for the initial implementation;
   underpromotion signatures remain disabled until a promotion-choice workflow
   is implemented.

Squares already proven to be retained adjacent-piece residuals may be excluded
from the unexpected-change penalty. They shall still satisfy the Phase 1
logical-color and lower/base reference-signature checks.

Offline legal mode shall match the settled image against queen-only
pseudo-legal physical signatures, including signatures excluded from the legal
move set because they fail to resolve check or expose the king. This allows an
illegal capture, promotion, or capture-promotion to retain its occupied
destination evidence and reach correction guidance. Every matched candidate
shall still be submitted to the legal authority before commit. If no physical
signature matches, the existing one-piece inference may still identify an
occupied-to-empty relocation. Neither path may commit an illegal move or replace
the accepted reference.

**Implementation status:** Implemented. Offline legal mode caches the current
legal signatures and pseudo-legal physical signatures, scores physical
signatures against each settled per-square change map, and submits a unique
match to the legal authority before arbitrary relocation inference. Retained
adjacent-piece residuals are excluded only for the signature whose source
satisfies the existing logical-color and lower/base reference check.

If no physical signature uniquely matches, ordinary source/destination
inference remains available solely to identify and explain an illegal physical
move to an empty destination. It cannot commit the legal game or accepted
reference. Physical promotion defaults to queen; underpromotion signatures are
not generated. A regression shall cover a black `e2xf1=Q` capture-promotion
while the black king on `e8` remains in check from a white pawn on `d7`; the
physical action must be detected and marked illegal.

**Live acceptance:** Completed. Ordinary captures, castling, en passant,
queen promotion, capture-promotion, check-constrained illegal moves, correction
restoration, and camera remapping behavior have been exercised on the physical
board.

## 6. Phase 3: Physical Piece-Color Classification

The camera does not need to distinguish knights from bishops. It needs a
conservative per-square classification:

- Empty.
- White piece.
- Black piece.
- Uncertain.

The classifier should use known logical piece colors and the accepted
reference as calibration evidence.

The first implementation shall learn reference-relative appearance samples for
empty squares, white pieces, and black pieces. Each sample shall use a
background-relative LAB descriptor from the square interior so classification
does not depend only on absolute brightness or board-square color. Empty
samples shall be compared within the same board-square parity. A current square
shall be classified only when the best class is within the configured distance
limit and separated from the runner-up by a confidence margin; otherwise its
label shall be `uncertain`.

Capture acceptance requires:

1. A known source piece.
2. A logically occupied destination containing the opposing color.
3. Strong source-vacancy evidence.
4. Continued destination occupancy.
5. Destination appearance changing to the mover's color.
6. Dominant reference differences on the expected squares.
7. A unique matching legal move.

Uncertain color evidence shall preserve the previous position and reference.

For the first integration slice, completed captures and capture-promotions
shall require the observed source to classify as empty and the observed
destination to classify as the moving piece's color. This evidence supplements,
and does not replace, the physical signature thresholds or authoritative legal
move check.

**Implementation status:** The reference-relative classifier and first capture
integration slice are implemented. Focused tests cover empty, white, black, and
uncertain labels; successful source-vacancy and mover-color evidence; rejection
of uncertain capture destinations; and the existing ordinary-capture and
capture-promotion tracker workflows.

**Live acceptance:** Completed with a legal white capture, a legal black
capture, and a wrong-color destination replacement. Both legal captures
committed successfully, while the wrong-color destination preserved the
accepted position, FEN, turn, and reference.

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

The tracker shall report a `pending-capture` state when a settled observation
uniquely identifies one captured-piece destination, even if multiple eligible
attacker sources could complete the capture. Pending evidence requires:

- A capture or capture-promotion physical signature for the current side.
- The signature destination to exceed the movement threshold and classify
  empty.
- Each candidate attacker source to remain below the movement threshold and
  retain the mover's color.
- No unexplained square above the normal residual limit.
- No hand or multi-square obstruction.

The pending state shall identify the removed destination and eligible source
squares, but shall not change the FEN, logical position, turn, UI pieces, or
accepted reference. Returning the captured piece shall return tracking to the
accepted position. Moving an eligible attacker to the destination shall be
evaluated as the original complete capture signature against the unchanged
reference.

If the only settled change is removal of a piece and the removal is not the
captured-piece-first stage of a pseudo-legal capture, the tracker shall report
an illegal removal rather than an ambiguous or unfinished change. This includes
removing the side-to-move's own piece without placing it on a destination after
the hand is withdrawn. The removed square shall be marked red on the UI and
projected red when projection is available. The correction shall instruct the
user to restore the piece. The FEN, logical position, turn, and accepted
reference shall remain unchanged.

The first capture slice supports a completed ordinary capture visible in one
settled observation. When a unique legal capture signature matches, the
application shall commit the complete authoritative post-move piece map and
the settled reference atomically. It shall not use the diagnostic relocation
method, because that method intentionally rejects occupied destinations.

If legal application succeeds but logical/reference commit fails, the legal
game shall roll back and all previously accepted state shall remain unchanged.
**Implementation status:** Single-action captures are live-accepted.
Two-action captured-piece-first tracking is implemented and awaiting live
acceptance.

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

The initial promotion mode is **Auto-queen**. After exact source-empty and
destination-moving-color evidence, apply queen promotion and commit the
logical position and reference together. The camera does not claim to
distinguish the same-color physical pawn from a queen.

Selection of rook, bishop, or knight is deferred to a later extension.

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

### 8.1 First reconciliation slice

The first implementation shall be explicitly started by the user. It shall:

- Preserve the authoritative FEN, turn, and piece identities throughout.
- Compare stable absolute physical occupancy with the authoritative position.
- Report each expected occupied square that is physically empty, including the
  expected piece color and type.
- Report each expected empty square that is physically occupied.
- Project every mismatching square in red for two seconds, then return the
  projector to black before collecting the next occupancy observation.
- Allow reconciliation to be cancelled without changing accepted state.
- Replace the accepted camera reference only after physical occupancy matches
  the authoritative position exactly.

The first slice shall not claim to detect same-occupancy piece swaps. Color
evidence and exact piece-identity reconciliation remain later Phase 5 work.

**Implementation status:** The explicit occupancy-reconciliation workflow,
structured discrepancy guidance, cancellation, repeated red mismatch
projection, and reference replacement after an exact match are implemented and
live-accepted. A physical missing/extra-square mismatch was identified,
projected, corrected, reverified with the projector black, and completed without
changing the authoritative game.

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

The application shall never request or store a Lichess username/password
credential pair. The first implementation shall read a personal access token
from `LICHESS_TOKEN`. The official Board game stream requires the `board:play`
scope even when the client only observes events; the read-only milestone shall
therefore enforce safety by exposing no move, challenge, chat, abort, resign, or
other mutating API methods.

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

### 9.1 First read-only client slice

The first client slice shall:

- Open `GET /api/stream/event` and
  `GET /api/board/game/stream/{gameId}` only.
- Send `Authorization: Bearer ...`, `Accept: application/x-ndjson`, and an
  identifying `User-Agent` on every request.
- Ignore empty NDJSON keep-alive lines.
- Parse and validate `gameStart`, `gameFinish`, `gameFull`, and `gameState`
  events into typed immutable models.
- Reconstruct the complete position from the initial FEN plus the complete UCI
  move list, rejecting invalid or internally inconsistent streams.
- Surface HTTP 401/403/404 errors without including the token.
- Convert HTTP 429 into an explicit retry delay, preferring `Retry-After` and
  otherwise waiting at least 60 seconds.
- Provide bounded exponential reconnect delays without sleeping inside the
  parser or HTTP transport.

The transport shall be injectable so automated tests never contact Lichess.

**Implementation status:** The typed read-only account and Board game stream
client is implemented. It reconstructs standard-chess FEN from the complete UCI
move list, exposes players, colors, clocks, side to move, status, winner, and
result, validates game identity and event ordering, handles keep-alives, and
surfaces redacted HTTP/protocol errors.

The desktop UI now provides explicit connect and disconnect controls, a
persistent Lichess status line, and simultaneous `Physical position` and
`Lichess position` board panels. The dashboard uses four quadrants: camera,
physical position, scrollable interaction panel, and Lichess position. Remote
updates render their reconstructed FEN and last move on the Lichess board
without changing the accepted physical position or camera reference. Network
iteration runs in a cancellable Qt worker thread and never on the camera/UI
thread. The worker keeps at most one HTTP stream open: it closes the account
stream after `gameStart`, follows that game, and returns to the account stream
after a terminal game state. It stops
permanently for authentication and protocol failures, honors `Retry-After` for
HTTP 429, and otherwise reconnects with bounded exponential backoff, including
after a socket or SSL read timeout. A successfully opened stream resets the
backoff sequence. Disconnect checks include blank keep-alive lines so an idle
stream can stop cooperatively.

If the network path blocks the idle account-event stream, the worker falls
back to polling. It discovers games through `GET /api/account/playing` and
uses that authenticated response's current FEN and authoritative `lastMove`
to maintain a legal cursor every five seconds. Unverifiable jumps establish a
new current-position baseline and require physical reconciliation. The public
game-export endpoint is not suitable because Lichess deliberately delays
ongoing-game exports.

The Lichess state drives the separate server cursor, current-position
reconciliation, pending physical-move guidance, and ordinary local move
submission. The physical cursor and accepted camera reference advance only
after camera confirmation or successful current-position adoption.

Official API reference:
<https://github.com/lichess-org/api/blob/master/doc/specs/lichess-api.yaml>

### 9.2 Phase 6 acceptance gate

The application follows a live game and displays exactly the same current
position as Lichess after every streamed event or verified polling transition.
When polling cannot verify a one-move transition, it establishes the reported
current FEN as a new reconciliation baseline rather than reconstructing
unreported history.

**Acceptance status:** Live-accepted against a standard correspondence game
with Stockfish. On the tested network, blocked streaming response headers
triggered current-position polling. It discovered the selected active game,
adopted its authenticated current FEN, verified subsequent authoritative
`lastMove` transitions, and updated the separate Lichess board panel.

## 10. Phase 7: Incoming Opponent Moves

Detailed behavior is defined in the
[Lichess physical move synchronization specification](./lichess-physical-move-synchronization-specification.md).

**Implementation status:** The ordinary-move synchronization slice is
implemented and live-accepted. Incoming and locally submitted non-promotion
captures are implemented and automated-test validated, including complete
single-action and captured-piece-first evidence, and await live acceptance.
Incoming and locally submitted castling is implemented with four-square white
guidance and king-first, rook-second confirmation, and awaits live acceptance.
Incoming and locally submitted en passant is implemented with three-square
white guidance and captured-pawn-first confirmation, and awaits live
acceptance.
Incoming and locally submitted queen promotion and capture-promotion are
implemented in persistent Auto-queen mode. Capture-promotion supports
captured-piece-first evidence. Underpromotion remains reconciliation-only
until computer-based piece selection is implemented.
The synchronization flow maintains separate server and physical legal cursors,
validates streamed history prefixes and verified polling transitions,
queues one or more reported moves in server order, shows one pending move at a
time in white on the physical UI board and projector, and commits the physical
position and camera reference only after exact expected evidence. It also
supports explicit game selection, automatic current-position reconciliation,
manual trusted acceptance through **Stop reconciliation** when the visible
board already matches Lichess, and
ordinary/capture/castling/en-passant/auto-queen local move submission.
Underpromotion selection remains staged work.

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

**Game-information slice:** Implemented and automated-test validated. The
Lichess panel displays White and Black player identity/rating, authoritative
clocks when available, side to move, game status, and result. Polling snapshots
without clocks show unavailable clocks rather than zero. Incremental state
updates retain player identity from the latest full snapshot.

**Start-new-game slice:** Implemented and automated-test validated. The native
menu action opens the Lichess website for game creation and then detects the
game through the existing account connection. Native challenge creation is
deferred until time-control, opponent, color, and rated/casual setup are
designed. The action does not discard completed-game information before the
replacement game is received. Existing ongoing-game IDs are excluded when
identifying the newly created game so unrelated older games do not force an
unnecessary manual selection.

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

