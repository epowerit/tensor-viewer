# Architecture and next increments

## Trace first, presentation second

```text
Project draft → worker → real PyTorch execution
                           ↓
                       trace recorder
                           ↓
                  operation adapter registry
                           ↓
                  immutable versioned run
                           ↓
                 frontend presenter registry
                           ↓
                 reusable tensor renderers
```

A trace has ordered operations plus tensor references. The ordered list powers playback. References retain branching and multiple inputs/outputs; the UI can follow an operand back to its producer. The first frontend uses a timeline and direct dependency links rather than an architecture-wide graph editor.

Every captured tensor state has its own ID. Actual storage identity is recorded separately. Values are serialized at capture time, so a later in-place operation cannot rewrite history. Aliases share a storage ID; a changed alias encountered later gets a new state. In this first version, implicit updates through an alias do not yet receive synthetic dependency edges back to the mutating operation. That provenance refinement belongs before presenting a complete mutation-aware graph.

The recorder runs at the high-level `TorchFunctionMode` boundary to retain readable names such as `reshape`, `contiguous`, and `linear`. It also captures generic tensor-returning calls. This is eager execution, not a compiler or a replacement implementation of PyTorch. Fused operations are shown as a single call.

## Semantic units

An operation adapter supplies an explanation and supported interaction capability. Layout adapters can provide an exact output-to-input index map. Matrix multiplication instead supplies a contribution interaction: the frontend selects the row and column from recorded operands. The trace retains PyTorch's result; displayed arithmetic is rounded for readability.

Capabilities are deliberately separate from operation names. Several operations can reuse the same mapping presenter or tensor renderer without sharing an incorrect explanation. Unknown operations remain inspectable. Semantic axis labels come from explicit annotations or transformations that preserve known meanings.

## Keep execution and teaching honest

- Preserve actual source locations and original inputs, including parameter tensors.
- Treat memory copying independently from logical rearrangement.
- Never invent an element-to-element path through a computation.
- Do not infer an image grid, token meaning, or head axis merely from dimension sizes.
- Label partial executions and keep successful steps when later operations fail.
- Store the exact source and input specification with every run.

## Next useful increments

1. Patch embedding: use a small image and link spatial patch membership to token rows.
2. Collapsible conceptual stages and an overview of the dependency graph.
3. Configurable visible axes, richer linear-projection interactions, and selected-element transition animations.
4. Multi-input calls, reusable input fixtures, and saved model weights.
5. Multi-file projects and an explicit model entry point.
6. Read-only Git import pinned to a commit, followed by explicit dependency setup and execution.

Git import should feed a source-loading layer into the existing runner. It should record repository revision, selected entry point, environment/dependencies, and run configuration. Cloning a repository must not automatically run its setup or model code. The trace API, operation adapters, and tensor renderers can remain independent of the source origin.

Larger projects also need bounded, on-demand tensor slices and stronger resource management. Full tensor snapshots are intentionally appropriate only for the small examples supported today. Hosted untrusted execution requires a separate security boundary; the current local worker is not that boundary.

## Manual browser acceptance checks

- Create an Attention or Tensor basics project, edit its configuration, and execute it.
- Select an output after permutation and confirm that input slicing follows the mapped coordinate.
- Inspect a matrix product and softmax normalization group.
- Change code after a run and verify the saved-run notice and unchanged source excerpt.
- Reopen previous runs and reload the page to verify persistence.
- Validate invalid input forms and a failed forward pass.
- Verify desktop layout and mobile access to project navigation.
