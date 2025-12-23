Perform a code review on specified file(s) against the V3 design spec.

Instructions:
1. Read the file(s) to review
2. Read `project-design/MOE-PROJECT-DESIGN-V3.md` as the reference spec
3. Get current git commit hash
4. Analyze the code critically for:
   - Correctness against V3 spec
   - Potential bugs (tensor shapes, gradient flow, edge cases)
   - Missing features specified in V3
   - Code quality issues
5. Categorize findings by severity (HIGH/MEDIUM/LOW)
6. Create review document in `docs/code-reviews/` with naming: `YYYY-MM-DD-{component}-review.md`
7. Include:
   - Commit hash reviewed
   - Findings with file:line references
   - What's verified as correct
   - Next steps to fix issues

Be critical and thorough. This is pre-production review.

Files to review: $ARGUMENTS