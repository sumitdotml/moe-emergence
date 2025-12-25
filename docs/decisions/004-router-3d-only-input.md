# Decision: Router Input Shape Is 3D Only

**Date:** 2025-12-26
**Status:** Accepted
**Context Commit:** `775bc0a`

---

## Context

The Router currently assumes inputs with shape `[batch, seq, hidden]`. A review noted that it could be extended to accept 2D `[tokens, hidden]` inputs. The current architecture (GPT-2 integration and MoE wrapper) only passes 3D tensors, so 2D support is not required.

---

## Options Considered

### Option A: Support 2D and 3D inputs

**Description:** Detect input rank and handle `[tokens, hidden]` by skipping reshape or by reinterpreting tokens as flattened sequences.

**Pros:**
- More flexible for standalone or flattened-token call sites
- Easier reuse in non-GPT-2 contexts

**Cons:**
- Adds branching and surface area to a core path
- Increases test burden and possible ambiguity in shape expectations

### Option B: Keep 3D-only contract

**Description:** Require callers to pass `[batch, seq, hidden]` and keep Router logic simple and aligned with GPT-2 integration.

**Pros:**
- Simpler code and fewer edge cases
- Matches current architecture and call sites
- Clear, explicit contract for the Router

**Cons:**
- Less flexible for future flattened-token use cases
- Would require a small refactor if 2D support becomes necessary

---

## Decision

Choose Option B. The Router will remain 3D-only because all current call sites already supply 3D inputs and no planned work requires 2D inputs. Keeping a tight contract avoids unnecessary complexity.

---

## Consequences

- **Positive:**
  - Simpler Router implementation and fewer code paths
  - Consistent behavior across GPT-2 integration and demos

- **Negative:**
  - Reduced flexibility for future experiments that use flattened tokens

- **Risks:**
  - If a future training pipeline flattens tokens, a change will be required

---

## References

- `docs/code-reviews/001-2025-12-23-moe-py-review.md`
- `docs/code-reviews/004-2025-12-23-loss-dedup-and-tests.md`
- `project-design/MOE-PROJECT-DESIGN-V3.md`
