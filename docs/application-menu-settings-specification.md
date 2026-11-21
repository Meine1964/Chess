# Application Menu and Settings Specification

**Status:** Implemented and automated-test validated

## 1. Purpose

The main window shall expose operational commands through a conventional menu
bar instead of permanent button grids. Live status, board displays, camera
preview, turn indication, and calibration progress shall remain visible in the
main window.

## 2. Menu Structure

### 2.1 File

- **Settings...** opens the settings dialog.
- **Exit** closes the application through the normal window shutdown path.

### 2.2 Camera

- **Take snapshot**
- **Resume live camera**
- **Reset pieces to starting position**
- **Reconcile physical board**, dynamically renamed to **Stop reconciliation**
  while reconciliation is active.

### 2.3 Game

- **Mode** contains mutually exclusive **Board setup / diagnostic** and
  **Offline legal game** actions. Offline legal game is the normal mode for
  playing chess. Board setup / diagnostic is reserved for arbitrary position
  setup, camera diagnosis, and recovery when no legal history is active; it
  shall not be presented as an equivalent gameplay mode.
- **Lichess** contains an **Active game** submenu populated from the current
  ongoing games.

The Lichess position panel shall contain a compact **Connect** or **Disconnect**
button centered in the Black information row between the player identity and
Black's clock. It shall display **Connecting...** while the initial connection
is pending and **Disconnecting...** while shutdown is pending; the pending
button shall be disabled. After connection succeeds it shall display
**Disconnect** and be enabled. The panel shall also contain a centered
Lichess-style icon control bar beneath the game status. It shall show takeback,
draw, resign, and new-game buttons with tooltips and accessible names. Takeback
remains a visible disabled placeholder and no takeback request shall be
submitted. Draw and resign shall be enabled only for an active selected game
with an authenticated Lichess client and while no other game-control request is
pending.

The draw button shall submit a draw offer through the Board API and disable
while the local player's offer remains active. When authoritative stream state
reports an opponent draw offer, the offer button shall be replaced by explicit
accept and decline buttons. Accept and decline shall use the Board API and
disappear when authoritative state clears the offer. Read-only polling does not
expose Lichess draw-offer flags, so incoming-offer controls are unavailable in
polling mode.

The resign button shall require explicit confirmation before submitting a
resignation. All game-control operations shall execute outside the UI thread,
disable game-control buttons while pending, report server rejection explicitly,
and report an uncertain outcome when the connection fails before confirmation.
Resignation shall remain disabled after submission until authoritative terminal
game state arrives.

An abort button shall be visible beside the other Lichess controls. It shall be
enabled only for an active standard-start game with fewer than two known plies,
which conservatively represents the opening interval in which Lichess may allow
an abort. It shall require confirmation and use the Board API asynchronously.
Lichess remains authoritative: a rejected abort shall leave the game active and
surface the rejection. After confirmed submission, abort shall remain disabled
until terminal game state arrives.

The new-game button shall be available while no Lichess game is loaded and
after the loaded game has finished, but not during active play or while a
creation request is pending. It shall open a native AI-game setup dialog with
level 1 through 8, player color, and clock choices. The dialog shall be
prefilled from the persistent defaults in File > Settings > Lichess; initial
defaults are level 4, White, and 10+5. Changes made only in the setup dialog
apply to that game and shall not replace the persistent defaults. Confirming
shall create a standard game through the authenticated Lichess AI challenge
API outside the UI thread. Cancelling shall make no state or network change.

When the account event stream reports the selected game finished, or polling
removes it from the ongoing-game list, the cached game state shall become
terminal before the new-game button is enabled. The final board may remain
visible, but stale `created` or `started` state shall not block a replacement
game after resignation or another terminal outcome.

After Lichess confirms creation, the application shall persist and follow the
returned exact game ID. The completed previous board, players, and result shall
remain visible until the replacement snapshot arrives. Rejected and uncertain
creation outcomes shall be reported explicitly without a browser fallback or
success-shaped local game.

When connection discovers multiple ongoing games and the exact remembered game
does not identify the intended selection,
the application shall open a modal game chooser. The chooser shall identify
each game by opponent, player color, and side to move rather than using the game
ID as its primary label, and shall show the current board position for the
highlighted choice. Confirming selects and persists that exact game. Cancelling
leaves every game unselected and shall not reopen the same chooser on every
poll; a changed ongoing-game set may prompt again.

### 2.4 Projector

- **Open projector black**
- **Show focus pattern**
- **Black projector**
- **Close projector**
- **Test patterns** contains **Project 8x8 grid** and
  **Project all 64 squares**.

### 2.5 Calibration

- **Full calibration - empty board**
- **Populated calibration - experimental**
- **Quick realign / assess**
- **Validate calibration**
- **Cancel calibration**

Existing availability rules shall continue to enable and disable the matching
menu actions. Moving a command shall not change its operational behavior.

## 3. Settings Dialog

**File > Settings...** shall open a modal settings dialog with:

- **Devices:** camera and projector display selectors.
- **Chess:** promotion mode with **Auto-queen** and
  **Select promoted piece** choices, plus **Automatically reset a detected
  starting position**. Auto-queen and automatic starting-position reset remain
  the defaults.
- **Diagnostics:** the **Save diagnostic snapshots** setting.
- **Lichess:** current connection status, persistent AI-game defaults for AI
  level, player color, and clock, plus a fresh configured/not-configured check
  of `LICHESS_TOKEN` from the process environment, without displaying or
  persisting the authentication token. Startup status shall use the same
  environment check and shall not claim that configuration is missing when a
  non-blank token is available.

**Apply** shall apply the selected values while leaving the dialog open.
**OK** shall apply them and close the dialog. **Cancel** shall close the dialog
without applying changes made since the most recent Apply.

The promotion mode shall persist. **Auto-queen** shall immediately interpret a
local promotion as queen promotion once its physical evidence is complete.
**Select promoted piece** shall start the projected virtual-rank selection
workflow defined by the Lichess physical-move synchronization specification.
Changing the setting while a promotion workflow is already pending shall not
change that workflow; the new mode shall apply to the next promotion.

The automatic starting-position reset setting shall persist and default to
enabled. When enabled and no Lichess game is active, the application shall
passively verify live occupancy while the projector is black. A stable standard
starting setup shall automatically trigger the reset even when the application
is already in Board setup / diagnostic mode; the user shall not have to select
Offline mode first. An Offline-mode position mismatch shall use the same live
verification. When disabled, the application shall show the existing recovery
dialog without attempting an automatic reset. While automatic reset is
enabled, the Camera menu shall hide **Reset pieces to starting position**
because the command is redundant. The command shall reappear when automatic
reset is disabled. The recovery dialog's **Reset to starting position** choice
shall remain available as a fallback after an automatic check does not find the
standard setup.

The dialog shall be populated from the current live selectors each time it is
opened so camera and screen hot-plug changes remain authoritative.

Camera and projector choices are persistent identities, not availability
hints. Startup and hot-plug handling shall never replace a remembered choice
with the system default, first available camera, primary display, calibrated
display, or any other available device. If the remembered device is absent or
switched off, its persisted identity shall remain unchanged, the corresponding
live selector shall have no active replacement, and dependent operations shall
remain unavailable. If that same device returns, the application may reattach
to it automatically because the selected identity has not changed. Only an
explicit user choice in Settings or an explicit camera command-line selector
may replace the persisted identity.

The Devices tab shall list the remembered camera and projector even while they
are unavailable, label them as unavailable, and keep those entries selected.
Available alternatives may also be listed, but merely opening, applying, or
accepting Settings without choosing one shall not replace the remembered
identity. Human-readable device labels shall be persisted alongside hardware
identities when available.

Opening Settings shall refresh Qt's current display enumeration before
building the Devices list. The application shall not rely solely on startup
enumeration or screen-added notifications, because Windows may activate a
connected projector without delivering that notification to the running Qt
window.

While Settings is open, the projector dropdown shall remain subscribed to
screen-added and screen-removed events. A projector reported by Qt shall appear
in the dropdown immediately without closing or reopening the dialog, while the
currently selected identity remains unchanged until the user explicitly
chooses and applies another display.

The diagnostic-snapshot checkbox and the selected Lichess game ID shall also
persist. A remembered Lichess game may be restored only when that exact game
is still present in the ongoing-game list; the application shall not replace a
remembered game with a different ongoing game. When no game has previously
been selected, the existing single-ongoing-game automatic selection may become
the remembered choice.

Transient operational state shall not persist. Startup shall not resume a
Lichess connection, projector output other than black, calibration,
validation, reconciliation, a frozen camera snapshot, or any one-shot command.
Automatic startup connection requires a separate future opt-in preference.

## 4. Main Window

The main window shall retain:

- Camera preview.
- Physical and Lichess boards.
- Turn indicator.
- Calibration progress and state indicators.
- Camera, detection, projector, alignment, occupancy, piece, Lichess
  connection, and synchronization status text.

The main-window placement shall persist on a normal close and restore after
the next startup. The application shall store the named screen, position
relative to that screen's available area, normal size, and maximized state.
Qt's native geometry state shall remain a fallback when the named screen or
explicit placement is unavailable. If all stored placement data is absent or
invalid, the application shall use its default 1500-by-900 size.

The lower-left interaction panel shall use a minimum 12-point font for its
turn indicator text, calibration state, progress text, status messages, and
instructions so they remain readable at the normal operating distance. This
font increase shall not enlarge text rendered inside the camera preview or
either chessboard.

The removed main-window controls shall not leave empty button, selector, or
interaction rows.

## 5. Failure Handling

- Invalid or unavailable settings selections shall preserve the persisted
  device identity and shall not select a substitute.
- A successfully activated game mode shall persist across normal restart.
  Startup shall not silently replace a persisted Offline legal game choice
  with Board setup / diagnostic.
- If the saved offline legal position and restored accepted physical position
  differ, startup or an attempt to activate Offline legal game shall first
  verify the live physical occupancy automatically. If it is the standard
  starting position, the application shall perform the Reset to starting
  position behavior without prompting and be ready for White's first move.
  Otherwise it shall show an explicit modal explanation and offer:
  - **Restore saved offline position:** retain the saved legal game and enter
    correction guidance until the physical board matches it.
  - **Reset to starting position:** verify the standard physical setup, reset
    both logical and physical game state, and then activate Offline legal game.
  - **Remain in board setup:** activate Board setup / diagnostic and persist
    that explicit choice.
- Closing or cancelling the mismatch dialog shall make no mode or persisted
  state change. Merely detecting a mismatch shall never rewrite the persisted
  mode.
- Existing command-level warnings and errors shall remain authoritative.
- A disabled command shall remain visible but unavailable in its menu.
- Lichess credentials shall continue to come from the existing environment
  configuration and shall never be copied into application settings.

## 6. Acceptance Criteria

1. Every former main-window button is available as a menu action.
2. Menu action text and enabled state follow the existing command state.
3. The active rules mode and active Lichess game are represented by checked
   menu actions.
4. File > Settings supports Apply, OK, and Cancel semantics for device and
   diagnostic settings; live game mode remains exclusively under Game > Mode.
5. No operational button grid or device/rules selector remains in the main
   interaction panel.
6. Existing camera, projector, calibration, game, reconciliation, and Lichess
   behavior remains unchanged.
7. Text in the lower-left interaction panel is rendered at no less than
   12 points.
8. Camera, projector, game mode, diagnostic snapshots, and exact Lichess game
   selection persist; unavailable identities are retained without substitution.
   The promotion mode also persists and offers **Auto-queen** and
   **Select promoted piece**, with Auto-queen as the default.
9. Connection and in-progress hardware operation state does not persist.
10. Remembered unavailable camera and projector identities remain visible and
    selected in the Devices tab.
11. A normally closed main window reopens with its previous screen, position,
    size, and maximized state.
12. A compatible saved Offline legal game reopens in Offline legal mode.
13. An incompatible saved Offline legal game produces the explicit recovery
    dialog and is never silently converted to Board setup / diagnostic.

## 7. Validation Status

The implemented menu bar exposes File, Camera, Game, Projector, and
Calibration menus. The complete automated suite contains 307 passing tests,
including menu structure, transactional selector-copy behavior, existing UI
state transitions, reconciliation controls, and the readable interaction-font
minimum. Token-status coverage confirms configured and missing environment
states without exposing credential values. Device-selection coverage confirms
that unavailable remembered hardware never causes a camera or projector
substitution. Preference coverage confirms diagnostic snapshot persistence and
exact Lichess game restoration without selecting a replacement. An offscreen
full-window smoke check also confirms that the complete Qt window constructs
the expected menu bar and File > Settings action. Projector-refresh coverage
confirms that current Qt display enumeration replaces stale cached availability
without changing the remembered selection. Window-state coverage confirms that
placement is written only on an accepted close and restored through the named
screen when it remains available. Auto-queen and Select promoted piece are
implemented persistent promotion settings. The local projected virtual-rank
workflow is automated-test validated and awaits live camera/projector
acceptance. The setup-mode rename, explicit offline-position recovery dialog,
camera-confirmed saved-position restoration, verified starting-position reset,
and no-silent-downgrade behavior are implemented and automated-test validated;
live recovery acceptance remains pending.
