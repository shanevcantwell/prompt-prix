# Git Safety

**Checkpoint: GIT-SAFETY-LOADED** — Confirm you've read this file before any branch operation.

---

## Before Any Branch Operation

Before checkout, delete, rebase, or any branch manipulation:

```bash
# 1. Where am I?
git branch --show-current

# 2. Any uncommitted work?
git status

# 3. What's the recent history?
git log --oneline -5
```

---

## Never Do These Things

### Workspace Destruction

```bash
# NEVER - destroys VSCode workspace and disconnects Claude Code session
git checkout <branch>  # if it would remove *.code-workspace

# NEVER - without explicit confirmation
git clean -fdx
```

### Orphan Branch Disasters

```bash
# NEVER - delete a branch containing the only copy of work
git branch -D <branch>  # without confirming another copy exists

# NEVER - checkout to orphan and delete original
git checkout --orphan <new>
git branch -D <old>  # You just lost everything
```

---

## Critical Files to Preserve

These files must survive all operations:
- `*.code-workspace`
- `.vscode/`
- `.claude/`
- `pyproject.toml`
- `.env`

If a git operation would remove or modify these, **stop and discuss**.

---

## Safe Patterns

### Switching Branches

```bash
# First, check for uncommitted work
git status

# If clean, safe to switch
git checkout <branch>

# If not clean, stash or commit first
git stash push -m "WIP: description"
git checkout <branch>
```

### Deleting Branches

```bash
# Confirm the branch is merged or backed up
git log --oneline <branch> -5  # What's on it?
git branch -a | grep <branch>  # Is there a remote copy?

# Only then delete
git branch -d <branch>  # lowercase -d fails if not merged (safer)
```

### Rebasing

```bash
# Create a backup branch first
git branch backup-before-rebase

# Then rebase
git rebase <target>

# If it goes wrong
git rebase --abort
# or
git checkout backup-before-rebase
```

---

## Recovery

If something goes wrong:

```bash
# Find lost commits
git reflog

# Recover to a previous state
git checkout <hash>

# Or reset current branch to previous state
git reset --hard <hash>
```

The reflog keeps ~90 days of history. But `.git/` deletion is unrecoverable.

---

## Why This Matters

Claude Code runs inside the VSCode workspace. Operations that modify the workspace file terminate the session. Operations that delete branches can lose context that only the now-terminated session understood.

The agent can destroy its own execution environment. These rules prevent that.
