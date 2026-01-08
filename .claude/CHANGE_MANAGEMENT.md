# Change Management Workflow

**Checkpoint: WORKFLOW-LOADED** — Confirm you've read this file before proceeding.

---

## The Sequence

```
1. IDENTIFY   →  2. FILE    →  3. TEST-DRIVEN LOOP  →  4. COMMIT  →  5. CLOSE
   (problem)      (issue)       (see below)             (atomic)      (with ID)
```

This is the primary operating model. Claude is responsible for enforcing this workflow, even when the user is in flow state.

---

## Step 1: Identify

When identifying a bug or feature request during conversation:
1. **Stop** before writing code
2. **Say**: "This looks like a bug/feature. Filing issue first."

---

## Step 2: File

**Before writing any code:**

```bash
# Check what's already open
gh issue list --state open

# File the issue
gh issue create --title "Bug: brief description" --body "## Summary
One-sentence description.

## Evidence
- Error message, failing test, or observed behavior
- Steps to reproduce (if bug)

## Root Cause (if known)
File and line number, architectural issue, etc.

## Proposed Fix
Brief description of the approach."
```

Even for "quick fixes"—file first. It takes 10 seconds.

---

## Step 3: Test-Driven Loop (Mandatory)

Do not skip or compress these sub-steps:

```
3a. Write targeted test(s) for the fix
         ↓
3b. Run tests → expect FAILURE (proves test catches the bug)
         ↓
3c. Implement the fix
         ↓
3d. Run tests → expect PASS
         ↓
    If FAIL → return to 3c
    If PASS → proceed to Step 4
```

**Why this matters:** If you write the fix before the test, you can't prove the test actually catches the bug. A test that passes before and after the fix proves nothing.

---

## Step 4: Commit

### One Issue = One Commit

Do not batch multiple issues into one commit:

```bash
# Good: Atomic commits
git commit -m "Fix #9: Only Loaded bypass - track loaded_models per server"
git commit -m "Fix #10: Replace dispatcher with BatchRunner"

# Bad: Batched commit
git commit -m "Fix scheduler bugs and add GPU prefix feature"
```

### Commit Message Format

```
{Fix|Implement|Add|Update} #{issue}: Brief description

- Bullet point of key change
- Another key change

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Step 5: Close

Always close with commit reference:

```bash
gh issue close {N} --comment "Fixed in {commit_hash}. Hotfix candidate for v{X.Y.Z}."
```

---

## When User Pushes to Skip

When the user says "just fix it" or pushes to skip the process:
- Acknowledge the request
- File the issue anyway (it takes 10 seconds)
- Explain: "This keeps the changelog clean and makes the work traceable"

---

## Changelog Generation

With atomic commits referencing issues:

```bash
gh issue list --state closed --json number,title,closedAt \
  --jq 'sort_by(.closedAt) | reverse | .[] | "- #\(.number): \(.title)"'
```

---

## Why This Matters

This isn't bureaucracy. It's how we:
- Generate accurate changelogs without archaeology
- Bisect regressions to specific commits
- Onboard collaborators without oral history
- Prove what was fixed and why, months later

The discipline costs 30 seconds per change. The absence costs hours of forensics.
