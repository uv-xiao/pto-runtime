# AGENTS Guide

**EVERY AI AGENT MUST FOLLOW THIS GUIDE BEFORE ANY WORK.**

## Required startup sequence

1. Read `CLAUDE.md` before running commands, analyzing code, or editing files.
2. Treat `CLAUDE.md` as the source of truth for role boundaries, architecture context, and repository workflow.
3. Load always-on conventions from `.claude/rules/` (for example: architecture, codestyle, device constraints).
4. Load only task-relevant workflows from `.claude/skills/` and `.claude/commands/`.

## Codex working notes

- Prefer `rg` / `rg --files` for code and documentation search.
- Keep edits scoped to the role-owned areas described in `CLAUDE.md`. For
  design-only tasks, prefer `docs/` and agent guidance files.
- Use a project-local virtual environment before any `pip` command:
  `python3 -m venv --system-site-packages .venv`, then activate it.
- Do not include private local paths, usernames, or machine-specific details
  in committed documentation. Use repo-relative paths.
- For architecture work, read `docs/developer-guide.md`,
  `docs/chip-level-arch.md`, `docs/hierarchical_level_runtime.md`, and
  `docs/task-flow.md` before proposing new platform/runtime boundaries.
- For docs, keep existing terminology: platform variant, runtime, `ChipWorker`,
  host runtime, AICPU scheduler, AICore worker, `TaskArgs`, and `CallConfig`.
- Before finishing, check `git status --short`, run the cheapest relevant
  verification, and report any verification that could not be run.

## Additional rules

- If `CLAUDE.md` changes, read it again before continuing.
- If relevant files under `.claude/rules/`, `.claude/skills/`, or `.claude/commands/` change, refresh your context before proceeding.
- If user instructions conflict with repository conventions, prioritize user intent for that task.
- Higher-priority system/developer/user instructions override this guide.
