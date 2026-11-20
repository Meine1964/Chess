# Copilot Working Preferences

## Implementation Consent

- Treat questions, investigations, reviews, brainstorming, and requests for recommendations as read-only unless the user explicitly asks to implement, edit, fix, create, delete, run, or apply something.
- Do not modify files merely because a possible improvement or fix was identified. Explain the finding and wait for an explicit implementation request.
- Running read-only searches and inspections is allowed when needed to answer accurately.
- Never commit or push unless the user explicitly requests that operation.

## Specification-First Development

- Before changing implementation code, create or update the relevant specification in `docs/` so the requested behavior, authoritative state, failure handling, and acceptance criteria are explicit.
- If no applicable specification exists, create one before implementation. Do not treat chat discussion, a todo list, tests, or code comments as a substitute for the repository specification.
- Keep implementation and tests consistent with the specification. When runtime evidence changes the solution, update the specification as part of the same task rather than leaving the disproved behavior documented.
- Before staging or pushing implementation changes, update the specification to describe the final validated behavior, implementation status, tests, and known remaining limitations.
- Do not push implementation changes when their specification is missing, outdated, or still describes the delivered behavior as merely proposed.
- Documentation-only, instruction-only, formatting-only, and repository-maintenance changes do not require a new product specification unless they intentionally change application behavior.

## Git Push Workflow

- When the user says `push`, treat it as permission to commit and push the current task's already-validated changes.
- Run the commit-and-push workflow directly in the terminal, not through an execution subagent, so Git paths, output, and exit codes remain exact.
- Run one concise Git status check, then stage only files changed for the current task using individually quoted, full repository-relative paths. Include known new untracked task files; do not use PowerShell arrays or bare filenames.
- Commit and run `git push` directly. Do not assign Git output to a PowerShell variable or merge stderr with `2>&1`, because normal remote progress can be misreported as `NativeCommandError`.
- Do not rerun tests, reread diffs, or repeat validations unless files changed since validation, file ownership is ambiguous, or Git reports an error.
- Exclude unrelated modified, generated, and untracked files without investigating them further.
- If a Git command returns an uncertain result, check the commit and upstream state once with `git rev-list --left-right --count <upstream>...HEAD` before retrying. Do not repeat a commit or push that already succeeded.

## Repository Boundaries

- Never modify, create, delete, format, migrate, commit, or otherwise alter files in the legacy `STPA_Tool` repository.
- Treat `STPA_Tool` as read-only reference material. It may be inspected when necessary to understand legacy behavior.
- Implement STPA functionality, bug fixes, and UI changes in `Assurance_Engine`, including requests that mention legacy feature names such as the Controllers tab or HCS editor.
- If a requested change appears to require editing `STPA_Tool`, stop and identify the appropriate `Assurance_Engine` implementation surface instead.

## Proportionate Validation

- Preserve quality with the smallest test that can meaningfully falsify the change.
- For a narrow change, run one focused test, targeted test module, compile check, or relevant static check first. Stop when that check passes unless the change's risk justifies more coverage.
- Run a broader regression suite only when explicitly requested, when shared behavior or cross-module contracts changed, or when focused validation reveals wider impact.
- Do not repeatedly run unchanged broad test suites after small follow-up edits. Re-run the directly affected check instead.
- For documentation-only or instruction-only changes, validate structure and syntax; do not run the application test suite.
- If no automated test covers the changed behavior, report the specific manual check needed instead of substituting an unrelated broad suite.

## CI And Pre-Push Validation

- Before pushing changes to Python source, tests, runtime configuration, dependencies, quality scripts, or workflows, run the exact local CI entry point: `.venv\Scripts\python.exe scripts/run_quality.py`. Focused tests remain the first check during implementation, but they do not replace this pre-push gate.
- Run the full quality gate after the final task-owned code change. Do not rerun it during the push workflow when it already passed and no staged file changed afterward.
- Do not push when the quality runner reports any mypy, Ruff, test, coverage, resource-warning, or compatibility failure. Fix task-caused failures and rerun the exact entry point before staging.
- When a GitHub Actions run fails, inspect the exact logs for every failed job and matrix version before editing. Group repeated matrix failures by root cause; do not infer the cause from the notification summary or annotations alone.
- Reproduce the failing workflow command as closely as the available environment permits. If the failure is platform-specific and cannot be reproduced locally, use the remote traceback as evidence and validate the workflow syntax plus the smallest relevant local command.
- Keep the `hosted-smoke` Ubuntu job's Qt runtime provisioning intact. PySide6 requires the system package `libegl1`; Python package installation alone is insufficient. When changing PySide6, HCS font rendering, `requirements-web.txt`, or hosted runner images, verify required Linux shared libraries explicitly.
- After a successful push, report the commit and remote result without waiting for the Quality workflow to complete. Monitor or inspect the workflow only when the user explicitly requests it or when a workflow failure is reported. If any job fails, retrieve its logs, fix the confirmed cause, rerun local validation, and push only after the correction is validated.

## Change Scope

- Keep edits minimal and confined to the requested behavior.
- Do not fix unrelated findings. Mention material issues separately and wait for direction.
- Before a potentially slow validation command, state what will run and why it is proportionate to the risk.

## Future Template Tag Numbering

- When creating a new template or deliberately redesigning one, zero-pad sequential numeric tag components to the width required by the expected range. For ten through ninety-nine records, use `XX01` through `XX99`, not `XX1` through `XX99`.
- Preserve the padded parent path in parent-scoped child tags, for example `PS01` to `PE01.1`.
- Do not migrate existing templates solely to adopt this convention. Existing project tags remain stable, and only an explicit Retag operation may renumber them.

## Failed Fix Cleanup

- When validation, runtime evidence, or user feedback disproves a fix, remove every change introduced solely for that unsuccessful approach before attempting or finalizing a replacement.
- Cleanup includes implementation code, tests, probes, flags, configuration, comments, imports, and documentation that no longer serve the confirmed solution.
- Retain a test from an unsuccessful attempt only when it still reproduces the real defect or protects the confirmed behavior; otherwise remove or rewrite it.
- Do not leave speculative fallback behavior or a superseded workaround in place merely because it is harmless or already written.
- After cleanup, rerun the narrow check for the confirmed fix and inspect the task-owned diff to verify that only intentional changes remain.

## Continuous Dead-Code Removal

- During inspections, reviews, and investigations, identify and report credible dead-code candidates even when cleanup is not the primary task. Treat this as read-only discovery unless the user has authorized implementation.
- Remove code, files, imports, dependencies, configuration, tests, and documentation made obsolete by the current change as part of that same change.
- When working in an existing area, remove nearby code only when repository-wide evidence shows it is no longer reachable or used and the cleanup stays within the task's ownership boundary.
- Check dynamic imports, plugin registration, callbacks, configuration references, command-line entry points, maintenance workflows, and plausible external consumers before declaring code dead. Static non-reference alone is insufficient for these surfaces.
- Do not preserve compatibility shims or obsolete implementations unless compatibility is explicitly requested or a verified consumer still requires them.
- Do not expand a task into speculative cleanup. Report uncertain candidates separately and leave them unchanged until their ownership and usage are resolved.
- Validate removals with the narrowest meaningful test or runtime check; broaden validation when deleting shared code or crossing module boundaries.

## PowerShell Commands

- Use PowerShell-native syntax. Use `Set-Location` instead of Command Prompt forms such as `cd /d`, and use semicolons rather than `&&` for sequential commands.
- Prefer workspace-relative paths and single-quoted PowerShell path literals to reduce escaping complexity.
- Keep `python -c` commands short and single-purpose. For multiline or quote-heavy Python, pass a PowerShell single-quoted here-string to `python -` instead.
- Do not nest multiple layers of shell and Python quoting when a simpler invocation is available.
- After a command fails because of parsing or quoting, change to a simpler command structure rather than retrying the same quoting pattern.

## Command Scope And Recovery

- Execute the known command directly. Never reconstruct a command from PowerShell or shell history when output is missing or incomplete.
- Rerun a known command with a narrow output filter when exact diagnostics are needed. Use the direct terminal rather than an execution subagent when command fidelity or complete output matters.
- Never launch an unrestricted recursive workspace scan, such as `Get-ChildItem -Recurse`, to recover a command or locate its output. Use an explicit path, a targeted glob, or `rg` with a narrow scope.
- Do not let an execution subagent substitute history searches, repository exploration, or broader commands for the command requested. Terminate the process immediately if it deviates or begins an unintended broad scan.
- When a tool reports that output was saved, read the reported output file directly. If no path is available, rerun the original command instead of searching the workspace for it.
