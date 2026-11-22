# Offline Clock Specification

## Purpose

Offline legal play may optionally use camera-controlled White and Black chess
clocks without requiring a Lichess game or external clock.

## Configuration

- The Offline clock is disabled by default.
- Settings shall provide an enable option, initial minutes, and increment
  seconds.
- Initial time and increment shall be positive or zero integer values accepted
  by the settings controls.
- Configuration defaults shall persist across application restarts.

## Lifecycle

- Entering or restoring Offline play initializes both clocks from the
  configured initial time plus increment and leaves them paused. For example,
  `10+5` initially displays `10:05` for both players.
- Start runs the clock of the side to move in the authoritative Offline legal
  game.
- If the clock has not yet been started after creating or resetting an
  Offline game, the first accepted move starts the next side's clock
  automatically. The untimed first mover receives no increment.
- Pause commits elapsed monotonic time and stops both clocks.
- An accepted move made after an explicit Pause switches the active side but
  leaves the clock paused.
- Reset restores both clocks to the configured initial time and leaves them
  paused.
- Leaving Offline mode pauses the Offline clock.
- Restoring an application session shall never count application downtime;
  restored clocks remain paused.

## Move authority

The clock shall switch only after the same accepted physical action that
advances the authoritative Offline `LegalGame`.

- Commit elapsed time against the mover.
- Include one increment in each player's initial display, then add the
  configured increment to the mover at the end of each timed accepted move.
- Start the next side's clock from the accepted legal-game turn.
- Ordinary moves, captures, castling, en passant, and confirmed promotions use
  the same switch rule.
- Candidate, ambiguous, obstructed, incomplete, illegal, and correction
  observations shall not switch clocks.

## Display and projection

- The control window shall show both Offline clock values and whether the
  clock is running or paused.
- Below one hour, clock values shall use `minutes:seconds`. At one hour or
  above, they shall use `hours.minutes:seconds`, with two-digit minutes and
  seconds within the hour.
- While enabled in Offline mode, project both clocks using the existing
  calibrated edge-clock geometry, orientation, perspective, parallax, and
  stroke compensation.
- Keep both clocks visible during projected promotion choices and promotion
  confirmation. Only the deliberate black evidence-measurement interval may
  temporarily replace the promotion display.
- Clock display updates may run periodically, but elapsed time shall be
  derived from a monotonic baseline rather than accumulated timer ticks.
- Offline terminal results shall retain the stopped clocks when clock data is
  available. Result-only rendering remains valid when the Offline clock is
  disabled.

## Timeout

- Reaching zero stops both clocks.
- White expiration produces `0-1`; Black expiration produces `1-0`.
- Project the existing expiration flag beside the losing clock and the result
  in the edge-result position.
- A clock timeout is authoritative for the Offline session even though the
  board-derived legal-game status is not checkmate or stalemate.
- Further Offline move acceptance shall remain disabled until the clock or
  Offline game is reset.

## Persistence

- Persist enabled state, configured initial time, configured increment, and
  committed remaining clock values, including whether the clock has already
  started.
- Persist no running monotonic timestamp.
- Restore remaining values only for the saved Offline game and restore them
  paused.

## Acceptance Criteria

1. Disabled Offline clocks preserve existing Offline behavior.
2. Start, pause, reset, and periodic display use monotonic elapsed time without
   tick accumulation.
3. Every accepted Offline move switches once and adds one increment.
4. Rejected or incomplete actions never switch the clock.
5. Promotion switches only after the promoted piece is confirmed.
6. Timeout stops both clocks and projects the correct result and expiration
   flag.
7. Restart restores saved values paused.
8. Lichess clocks and controls remain unchanged.

## Implementation Status

Implemented in the Offline game UI and projector integration.

- The Chess settings tab controls enablement, initial minutes, and increment.
- The Offline interaction panel provides Start, Pause, and Reset controls.
- Accepted ordinary moves and confirmed promotions switch the clock from the
  authoritative `LegalGame` turn.
- Timeout blocks further Offline move tracking until reset or a new game.
- Saved remaining values restore paused with the saved Offline game.

Validation:

- Offline clock, game UI, and projector behavior is covered by focused model,
  move-acceptance, timeout, persistence, and projection tests.
- Current full repository quality gate: 451 passed.
