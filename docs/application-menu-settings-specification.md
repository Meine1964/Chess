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

- **Mode** contains mutually exclusive **Diagnostic movement** and
  **Offline legal game** actions.
- **Lichess** contains the dynamic connect/disconnect action and an
  **Active game** submenu populated from the current ongoing games.

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
- **Diagnostics:** the **Save diagnostic snapshots** setting.
- **Lichess:** current connection status plus a fresh configured/not-configured
  check of `LICHESS_TOKEN` from the process environment, without displaying or
  persisting the authentication token. Startup status shall use the same
  environment check and shall not claim that configuration is missing when a
  non-blank token is available.

**Apply** shall apply the selected values while leaving the dialog open.
**OK** shall apply them and close the dialog. **Cancel** shall close the dialog
without applying changes made since the most recent Apply.

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
9. Connection and in-progress hardware operation state does not persist.
10. Remembered unavailable camera and projector identities remain visible and
    selected in the Devices tab.

## 7. Validation Status

The implemented menu bar exposes File, Camera, Game, Projector, and
Calibration menus. The complete automated suite contains 246 passing tests,
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
without changing the remembered selection.
