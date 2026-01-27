---
name: doc-status
description: Generate a status report of the project documentation and implementation state.
disable-model-invocation: true
---

Generate a status report of the project documentation and implementation state.

## Instructions

1. List all files in `docs/` and summarize what's documented
2. Check git status for uncommitted changes
3. Read `README.md` progress checklist and report what's done vs pending
4. List any open issues from code reviews (grep for "- [ ]" in docs/)
5. Summarize:
   - Decisions made (count and list)
   - Experiments completed
   - Code reviews done
   - Outstanding fixes needed
   - Next recommended action

Output a concise status suitable for daily standup or progress check.
