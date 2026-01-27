---
name: doc-fix
description: Document a bug fix or issue resolution. Use when the user fixes a bug, resolves an issue, or corrects a problem in the codebase.
---

Document a bug fix or issue resolution.

## Instructions

1. Get the current git commit hash (post-fix state)
2. Ask for or identify:
   - What was the bug/issue?
   - How was it discovered?
   - What was the root cause?
   - What was the fix?
   - How was it verified?
3. Determine the next fix number by listing existing files in `docs/code-reviews/`
4. Add an entry to the relevant code review document if one exists, OR
5. Create a new document in `docs/code-reviews/` titled `NNN-YYYY-MM-DD-{component}-fix.md`
6. Include before/after code snippets if relevant
7. Note any related issues that should be checked

This creates an audit trail for the technical report.

Bug/fix to document: $ARGUMENTS
