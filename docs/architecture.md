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

A trace has ordered operations plus tensor references. The ordered list powers playback; references build the journey graph. The frontend links each operand to its latest recorded producer, lays out dependency levels from left to right, and separates parallel branches vertically. Multiple operand positions and multiple outputs are retained. Unproduced parameter roots stay in the inspector, while external captured tensors appear as explicit roots. Selecting a node highlights its recorded ancestors and source line; selecting an executable source line locates its operation. The graph is a view of one saved execution, not a graph editor or a symbolic model of every possible path.

Selecting a node also opens `TransformationFocus`, an unscaled panel over the canvas that reuses `OperationView` and `TensorCard` for the incoming tensors, operation, and output. It keeps the graph mounted and code independently collapsible. Container queries keep diagrams side by side when space allows and stack them on narrow screens. Dismissing focus restores the graph overview without losing the selection.

The shared tensor explorer separates the viewing plane from logical coordinates. Users can choose any two axes, navigate remaining axes as slices, and inspect recorded values without executing a tensor operation. Each tensor has a local cursor; presenter-provided highlights indicate supported relationships. A selected cell drives its slice and page, so keyboard navigation and mapped selections stay visible. Storage positions use the recorded offset plus the coordinate/stride dot product, and are labeled with their storage ID. Scalars have one selectable cell; any zero-sized axis means there are no cells to draw.

Rendering is bounded to an 8 × 8 window (optionally 16 × 16) per explorer. Visible-element counts and row/column ranges distinguish that window from the whole tensor. `TensorNavigation` uses constant-size numeric controls instead of one option per slice. Coordinate entry validates the original axis order, selects the exact logical element, and brings it into view. Cell labels use compact rounding while the selection readout retains the full recorded value. Shared display preferences keep the focused view and inspector consistent.

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
2. Collapsible conceptual stages and richer layout for larger dependency graphs.
3. Richer linear-projection interactions and selected-element transition animations.
4. Multi-input calls, reusable input fixtures, and saved model weights.
5. Multi-file projects and an explicit model entry point.
6. Read-only Git import pinned to a commit, followed by explicit dependency setup and execution.

Git import should feed a source-loading layer into the existing runner. It should record repository revision, selected entry point, environment/dependencies, and run configuration. Cloning a repository must not automatically run its setup or model code. The trace API, operation adapters, and tensor renderers can remain independent of the source origin.

Large value snapshots are written as immutable NumPy arrays by the worker before later mutation can change their contents. Small tensors remain inline for backwards compatibility. `TensorState.value_source` distinguishes inline values, paged values, and shape-only metadata. The run response excludes large arrays; the bounded values endpoint reads at most 256 logical positions using a memory map. The client cancels stale window requests and shows loading/error states rather than stale values.

Layout adapters emit compact `mapping_rule` descriptors for large tensors. Identity, permutation, and unfolding relationships are evaluated per coordinate, avoiding an array with one entry per element. Large inverse unfold memberships are capped at 256 highlights. Dot-product and normalization views also bound the highlighted contributors and do not claim numerical calculations from missing data.

The recorder supports 8,388,608 elements per value tensor and 32 million across captured states. Shapes mode constructs the module and inputs on PyTorch's meta device and records no numeric values, allowing up to 2^40 logical elements per tensor. Value-dependent code and unsupported meta kernels report failures explicitly. Neither mode removes the 20-second worker or 256-operation bounds. Hosted untrusted execution still needs a separate security boundary; the current local worker is not that boundary.

## Manual browser acceptance checks

- Create an empty canvas, add attention or a CNN → tokens → attention sequence, edit its configuration, and execute it.
- Select an output after permutation and confirm that input slicing follows the mapped coordinate.
- Inspect a matrix product and softmax normalization group.
- Change code after a run and verify the saved-run notice and unchanged source excerpt.
- Reopen previous runs and reload the page to verify persistence.
- Validate invalid input forms and a failed forward pass.
- Verify the attention Q/K/V fork and two matrix-product joins on the full canvas.
- Pan, zoom, fit, collapse the inspector, and follow source lines back to graph nodes.
- Check full-path playback and optional step-by-step reveal; opening the editor pauses playback.
- Verify desktop layout and mobile access to project navigation and drawers.
- Select input and operand cells, move across page boundaries with arrow keys, and swap viewing axes without changing the logical coordinate or value.
- Check overlapping `unfold` memberships, empty tensors, scalars, and storage positions on non-contiguous views.
- Jump to the final coordinate of a large synthetic tensor, confirm 64 or fewer rendered cells, reject out-of-bounds coordinates, and inspect the same controls on a narrow screen.


## Model composition

`Blueprint` stores the input-enabled flag and an ordered sequence of component IDs, kinds, and integer settings. `composer.py` owns the toolbox registry, bounded shape inference, validation, and Python generation. `/toolbox` exposes this catalog; `/compose` returns inferred stages and generated code. Project create/update canonicalizes code from a blueprint on the server. Invalid designs can be saved, but generate an explicit failure and are blocked from execution in the UI. Numerical mode checks model-weight and known intermediate sizes before constructing the model. Existing code projects have no blueprint and continue through the same runner.

`Toolbox.tsx` defines the compact category rail and the on-demand searchable library panel. Component definitions and parameter bounds come from the backend catalog. `App` controls which category is open; selecting a component closes that panel and opens its settings without changing the canvas scale. Project setup and input configuration use the same inspector styling. Custom add-on loading is deferred.

`BuilderCanvas` keeps design-time shapes separate from captured tensors: previews use `value_source=shape`, have no values, and are labeled inferred. Execution still produces an immutable run through the existing worker. Adding new components requires a registry entry, matching code/shape rule, and a numerical comparison against uninstrumented PyTorch.

## Indexed volume renderer

`volume.ts` provides pure sampling, logical coordinate mapping, rotation, and cube-face projection. A large axis retains indices 0, 1, and its final index, plus a selected interior index if necessary. Gaps describe all omitted intervals exactly. The last three axes form spatial cells, the preceding axis forms separate volumes, and earlier axes are explicit fixed slices. This bounds geometry to at most 4^4 = 256 indexed cells.

`TensorVolume` is reused for node thumbnails and the enlarged inspector. It projects the same unit-cell geometry for every layer; rotation changes the camera, not coordinates or values. Rear faces are culled and cells are painted in depth order. Cell text is clipped to its visible face. The volume is a compressed logical view, not a physical storage layout or a proportional image. `TensorVolumeDialog` adds per-axis navigation, gap drill-down, slice isolation, and exact-value readout. Numeric runs use the existing bounded snapshot endpoint; design and meta previews never invent numeric contents.
