# Architecture

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

Every captured tensor state has its own ID. Actual storage identity is recorded separately. Values are serialized at capture time, so a later in-place operation cannot rewrite history. `Operation.mutations` records before/after pairs as `write`, `alias`, or `metadata`, separately from actual return values. Version counters, storage identity, shape, strides, offset, and dtype are observed around intercepted calls. Known layout-only operations do not cause value updates to other aliases merely because their shared version counter increments.

A storage write immediately snapshots every already-captured tensor on that storage, including aliases with independent version counters. These are storage-level dependencies: disjoint slices may have unchanged values. Snapshotting is eager to preserve intermediate writes accurately; all additional states count toward the existing capture budget. Large snapshots stay paged, and Shapes-only mode retains metadata without values. No element-level write mask or synthetic PyTorch operation is invented.

`producedTensorIds` resolves explicit returns and mutation states. The graph links prior alias states into the writer with dashed storage edges and links their refreshed states to later consumers. Stage folding retains effects that cross a module boundary even if the module did not return them. `MutationView` compares one selected before/after pair using the existing bounded tensor explorer and 3D dialog. Original operands and return values remain separately inspectable.

This remains an eager `TorchFunctionMode` recorder. Native storage operations that bypass that boundary (including `set_` in the supported environment) cannot always be attributed. If a previously captured tensor changes outside an intercepted operation, a tracking notice accompanies the new captured root. External NumPy/raw-storage writes that do not change observable tensor metadata/version counters cannot be detected. Old traces default to empty mutation lists; rerun them to obtain the new provenance.

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

## Completed roadmap increments

The current milestone includes shifted-window/relative-bias lessons, branch/join authoring, multi-file entry points, committed Git source import, and explicit dependency environments. These build on the existing hierarchical model, trace recorder, immutable runs, and bounded tensor renderers. The scoped limits below remain intentional; Git import is a source snapshot importer rather than an arbitrary repository installer.

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
- Run patch embedding and follow a nonzero batch, patch, channel, pixel, and feature through the full progression. Verify source-image coordinates, corresponding kernel coordinates, token order, and selection preservation when switching to Tensor details and back.
- Run Hierarchical vision and expand fine-grid windows, patch merging, coarse-grid windows, and the classifier. Inspect partition/restoration and grouping lessons, select a later window and batch, and verify linked coordinates through 3D selection. Check the TL/BL/TR/BR feature order and confirm the five-step progression fits a narrow screen.

## Patch embedding lesson

`operations/spatial.py` qualifies complete Conv2D patch tilings using recorded kernel size, stride, padding, dilation, groups, and output shape. The lesson reports `patch_projection` and a rectangular `patch_size`. General or incomplete convolutions keep a generic inspector. The shared PatchEmbedding source records projection, flatten, and transpose as distinct real operations in both examples and composed models.

`patches.ts` links those steps through captured tensor IDs and validated identity/permutation rules, rather than matching shapes alone. Its pure helpers map patch positions, input pixels, kernel entries, projected features, and token coordinates. `PatchEmbeddingView` uses those mappings and the existing paged-value hook to display a synchronized spatial grid, selected input channel, projection kernel, one contribution, and the recorded output. Pixel windows are explicitly views, not synthetic execution nodes. The first lesson supports at most 8 × 8 visible patches/pixels and eight visible output features, independent of batch, image, kernel, or embedding size. Shape-only runs use the same coordinate interactions without values.

Tests compare captured results to PyTorch, verify exact rectangular patch contributions, reject unsupported convolution variants, follow real dependencies, and bound large shape-only traces and frontend windows.

## Selected-element layout replay

`layoutTransition.ts` validates nonempty, value-preserving bijections from the recorded layout capability, axis order, shapes, dtype, and selected output-to-input mapping. It uses logical flat indices for reshape and coordinate permutation for axis reordering. Storage positions are calculated independently from offsets and strides. Compatible older explicit maps are supported; one-to-many windows, dtype views, failed operations, and unverified layouts do not get this animation.

`LayoutTransitionView` animates one selected marker through source, mapping-rule, and destination stops. Both endpoints use bounded 4 × 8 last-two-axis windows with labeled fixed leading coordinates. The pure scene and interpolation helpers place stops exactly at the selected cells. Numeric values come from immutable snapshots through the existing paged hook; shape-only runs show a marker without fabricated values. Animation never executes model code or adds synthetic operation nodes.

Playback uses a cancellable animation frame, pauses when the document is hidden, and resets on selection changes. Manual steps and scrubbing remain available with reduced motion. The focused view adapts to a vertical diagram on narrow screens. Tensor details stay mounted to preserve viewing axes and selection. Tests cover reshape, permutation, noncontiguous copies, scalar/unit axes, legacy maps, rejected variants, billion-element windows, and exact playback stops.

## Linear projection lesson

The compute adapter marks standard, nonempty `linear` calls as `linear_projection` after validating input, matrix-weight, optional vector-bias, and output shapes. `linear.ts` validates those same recorded operands, including compatible older runs, and provides pure coordinate mappings. Selecting an output fixes its leading coordinates and output feature; selecting an input changes the input vector and contribution index; selecting a weight changes the output feature and contribution index. Weight rows are output features, and columns are input features. Noncontiguous storage does not change logical snapshot indexing.

`LinearProjectionView` reuses `TensorCard`, its 3D explorer, direct-index controls, and `useTensorValues`. It allocates an eight-term contribution window independently of tensor size. Complete sums fetch at most 256 input/weight pairs; larger dimensions display only the selected window sum. Missing, non-finite, and overflowing products never become fabricated finite results. Bias and actual output come from the immutable run. Browser arithmetic is explicitly approximate because floating-point accumulation may differ from PyTorch. Shape-only traces use the same mappings without numbers. The Tensor details fallback preserves the lesson selection when switching back.

Backend checks compare selected contributions and complete outputs with PyTorch across input ranks, optional bias, keyword operands, noncontiguous views, paged snapshots, and meta tensors. Frontend tests cover row orientation, leading-coordinate preservation, input/weight selection, bounded final windows, and unavailable numeric data. Manual acceptance checks cover keyboard selection, 3D selection, narrow screens, and numeric paging.

## Recorded module stages

`Trace.module_calls` is an additive, optional-on-old-runs collection of actual `nn.Module` invocations. Forward hooks record a unique call ID, parent call, canonical module path, class, operation interval (exclusive end), and entry/return tensor IDs. Snapshot work in hooks temporarily suspends operation recording, so metadata capture does not invent detach or reshape steps. Repeated invocations stay separate. Calls with no recorded tensor operations are discarded; failed calls still retain their executed interval and do not receive synthetic outputs.

`stages.ts` selects calls spanning at least two operations. Generated composition wrappers are omitted while real component and nested-module calls remain available. `collapseJourney` contracts selected call intervals on the original dependency graph, retains crossing tensor edges, and lays out the resulting DAG. It never rewrites the trace or infers a stage from equal tensor shapes. The `Stages` navigator expands calls one level at a time, while `All operations` restores the full graph. `StageFocus` reuses bounded `TensorCard` inspectors for actual call inputs and outputs and links to all enclosed operations. Direct operation navigation opens its ancestor groups. Reveal mode keeps all stages expanded, and the minimap follows the same visibility cutoff as the canvas.

This initial grouping follows module calls, not arbitrary Python regions. Functional operations between modules remain individual nodes. Older executions remain ungrouped until rerun. Recorded storage dependencies survive folding; boundaries still distinguish actual module returns from side effects. Unintercepted external writes retain the recorder limitations described above.

## Reusable generated inputs

`InputFixtureDraft` validates a named `InputSpec` and recording mode. The `input_fixtures` SQLite table stores immutable configuration snapshots through GET/POST `/api/v1/input-fixtures`. Saving and listing never execute model code or allocate tensors; shape-only entries retain the full logical shape within the existing 2^40 limit. Value entries use the same 8,388,608-element input limit as projects.

`InputLibrary` is shared by the builder inspector and custom-code editor. It starts collapsed, loads on open, previews all saved settings, and copies a selected configuration into the draft. It does not add a live fixture reference to the project. The composer then performs its existing shape checks, and a run captures the full copied specification. Invalid or unapplied input fields block saving to the library. Applying an input clears input-field errors while preserving unrelated constructor/code edits and validation errors.

`InputSpec.random_stream` defaults to `model` for backward compatibility. New UI random selections opt into `input`: a dedicated CPU `torch.Generator` seeded from `InputSpec.seed` generates the input independently of parameter initialization and does not consume the global RNG used by the forward pass. The worker retains the existing model seeding. Meta execution does not generate numeric values. The stream choice is part of fixtures, project signatures, custom-component checks, and run snapshots. Reproducibility is scoped to the current execution environment, shape, dtype, and seed; weights and PyTorch version are not pinned by a fixture.

Checks cover validation, persistence after restart, copy isolation, immutable run settings, large metadata-only saves, independent seeded inputs across different constructors, preserved legacy RNG behavior, and the unchanged forward RNG stream. Browser acceptance covers builder and custom-code loading, invalid-draft recovery, numeric/shape-only mode restoration, and narrow layouts.

## Uploaded tensor inputs

`POST /api/v1/input-fixtures/upload` streams an octet-stream body into a temporary file with a hard byte limit; name and original filename are bounded query fields. `input_files.py` checks the NumPy magic, format version, header length before reading the header, dtype, rank, element budget, and exact payload size before mapping data. Pickle is never enabled. Native-endian contiguous values are written under a fresh opaque ID in `.data/inputs/`; file metadata and the corresponding immutable fixture are committed together. Failed imports remove only their own temporary/partial files.

`InputSpec.uploaded` pins the asset ID, filename, canonical-file SHA-256, shape, dtype, and byte count. Its validator prevents mismatched dimensions, data types, or generator modes. API save/compose/check/run paths also require the exact stored asset metadata, preventing client-supplied paths or fabricated references. The worker receives the inputs directory separately from project JSON, checks metadata and checksum before user code runs, and makes a writable private array copy. Meta runs use `torch.empty` on the meta device. Individual custom-component checks use generated zero metadata at their actual incoming shape, while full-sequence checks retain the original uploaded input metadata.

`UploadInput` imports into the shared `InputLibrary` without replacing an in-progress draft. Applying remains an explicit action. Both editors display uploaded provenance and lock shape/dtype controls; switching to a generated source clears the file reference. Fixture copying clones nested asset metadata, and project signatures normalize absent references for older projects. Original logical values are retained; original Fortran strides and byte order are not portrayed as the imported tensor's memory layout. Large numeric inputs use the existing immutable paged snapshots after execution.

Tests cover all supported dtypes, Fortran and big-endian arrays, exact int64/non-finite readouts, paged values, meta execution, repeated in-place mutation, backend restart, reference tampering, corrupt/missing files, custom modules after shape changes, allocation/header bombs, partial/extra payloads, and rejected object arrays without unpickling. Browser acceptance checks import errors, successful application, fixed fields, source switching, recorded numeric values, and narrow layouts.

## Model composition

`Blueprint` stores the input-enabled flag and an ordered sequence of component IDs, kinds, and integer settings. `composer.py` owns the toolbox registry, bounded shape inference, validation, and Python generation. `/toolbox` exposes this catalog; `/compose` returns inferred stages and generated code. Project create/update canonicalizes code from a blueprint on the server. Invalid designs can be saved, but generate an explicit failure and are blocked from execution in the UI. Numerical mode checks model-weight and known intermediate sizes before constructing the model. Existing code projects have no blueprint and continue through the same runner.

`vision.py` supplies inspectable source for `TokenPreparation`, `ClassTokenReadout`, and `VisionTransformer`. The full ViT composes the existing patch embedding and pre-norm transformer block with these reusable sequence components. Shape inference counts one extra class token, learned class/position parameters, all block weights, final normalization, and classifier parameters. Fixed image resolution is validated at runtime; different batches are supported. This is an untrained educational model with logits output, no dropout, and no automatic positional interpolation. Tests compare the complete traced execution to an independent functional PyTorch reference, match modular assembly using identical weights, and exercise rectangular images and large shape-only batches.

`hierarchy.py` supplies reusable spatial embedding, window partition/restoration, window transformers, 2×2 patch merging, and spatial readout. `HierarchicalVisionTransformer` combines two fixed-window stages around one merge. Spatial stages use NCHW; partition packs row-major windows into `[B × windows_per_image, window², C]` for the shared `TransformerBlock`, and restoration reverses that layout. Patch grouping concatenates TL/BL/TR/BR neighbors into 4C features, then LayerNorm and a bias-free projection produce 2C. The full model validates both grids' window divisibility, supports variable batch sizes, and uses initialized parameters without shifted windows or relative-position bias. Pure shape helpers count parameters and bound scores by `B × windows_per_image × heads × window⁴`, along with expanded MLP intermediates, before numeric allocation.

`spatialGrouping.ts` recognizes a layout lesson only after validating the recorded module type, four connected operations, exact intermediate shapes, permutations, mapping rules, dtype, and successful non-mutating boundaries. It composes existing logical-index mappings in both directions; class names alone do not establish provenance. `SpatialGroupingView` displays all five recorded shapes, shares interactive tensor cards and 3D selection, and bounds neighborhood highlighting to 64 cells. It distinguishes value-preserving grouping from the later learned merge projection and reports recorded endpoint storage sharing separately. Tests compare every partition cell, inverse restoration, merge neighbor order, independent per-window functional attention, and the complete hierarchy against a separate PyTorch calculation. Large shape-only and non-contiguous cases remain bounded.

Tensor–tensor addition advertises the `broadcast_add` lesson only for compatible, non-empty shapes and finite numeric alpha. The frontend independently validates old/new traces, right-aligns axes, and computes both operand coordinates in O(rank), without an element-sized map. Reverse selection retains output coordinates on broadcast axes. `AdditionView` reuses tensor explorers and paged values; it never identifies arbitrary additions as residual/positional by shape alone. Captured outputs remain authoritative, and mutations keep the separate storage-change view. LayerNorm has its own group/affine lesson; GELU retains a semantic description with ordinary tensor inspection.

`Toolbox.tsx` defines the compact category rail and the on-demand searchable library panel. Component definitions and parameter bounds come from the backend catalog. `App` controls which category is open; selecting a component closes that panel and opens its settings without changing the canvas scale. Project setup and input configuration use the same inspector styling. The Custom category also exposes locally saved single-file PyTorch modules.

`BuilderCanvas` keeps design-time shapes separate from captured tensors: previews use `value_source=shape`, have no values, and are labeled inferred. Execution still produces an immutable run through the existing worker. Adding new components requires a registry entry, matching code/shape rule, and a numerical comparison against uninstrumented PyTorch.

## Custom component library

`CustomComponentDraft` validates syntax and constructor settings without executing code. SQLite stores immutable `CustomComponent` entries; toolbox responses include their metadata and source snapshots. A custom `ComponentSpec` pins that snapshot alongside optional per-node JSON arguments. Saving a new version changes only the selected instance.

`custom_components.py` owns explicit checks and namespace generation. Passive `/compose` requests only consult bounded in-memory check caches. `/compose/check` shares the execution lock with Run, executes custom modules on the meta device through the existing timeout-controlled worker, then checks the complete generated sequence to catch stride-sensitive failures. Any change to code, arguments, input configuration, recording mode, or sequence requires a matching check. Unsupported meta kernels and value-dependent behavior produce explicit errors. Results are not numeric tensor captures.

Generated code places each module's source in a separate namespace and aligns its compiled source line numbers with the displayed program. The recorder reads embedded source assignments to retain original variable names and source excerpts. Runtime guards also verify each custom output against its checked shape and dtype; differences fail the trace instead of silently using stale preview shapes. The initial sequence contract requires one non-empty rank 1–6 tensor in and out with a preserved dtype. Multi-input components in the sequence builder, per-component checkpoint remapping and multi-file reusable add-ons remain outside the builder contract. Dependency environments and multi-file entry points are available in custom-code projects. Model-level saved weights are supported for the complete generated composition. Custom-code projects support multiple forward inputs separately.

`CustomComponentDialog` provides source import, library metadata, and default JSON settings. The inspector's `ConstructorArguments` makes unapplied edits explicit and blocks Run until they are applied. The existing generated-code, tensor, operation, and run-history views are reused.

## Indexed volume renderer

`volume.ts` provides pure sampling, logical coordinate mapping, rotation, and cube-face projection. A large axis retains indices 0, 1, and its final index, plus a selected interior index if necessary. Gaps describe all omitted intervals exactly. The last three axes form spatial cells, the preceding axis forms separate volumes, and earlier axes are explicit fixed slices. Axes up to 16 expand fully within a total budget of 512 cells for thumbnails and 1,024 for enlarged views. If expansion would exceed that budget, only the largest necessary axes are condensed. Budgeting reserves room for a selected interior index, keeping compression stable during selection. Consecutive cells meet at their boundaries; only omitted intervals introduce geometric gaps. Isolating a slice frees budget for other axes. Paged numeric reads are split into concurrent requests of at most 256 indices, with a shared abort signal.

`TensorVolume` is reused for node thumbnails and the enlarged inspector. It projects the same unit-cell geometry for every layer; rotation changes the camera, not coordinates or values. Rear faces are culled and cells are painted in depth order. Cell text is clipped to its visible face. The volume is a compressed logical view, not a physical storage layout or a proportional image. `TensorVolumeDialog` adds per-axis navigation, gap drill-down, slice isolation, and exact-value readout. Numeric runs use the existing bounded snapshot endpoint; design and meta previews never invent numeric contents.


## Forward call configuration

`ProjectDraft` preserves the original `input` field and adds `input_name` (default `x`), `input_binding` (default positional), and up to seven `additional_inputs`. Each `ForwardInput` pins a name, positional/keyword binding, and full `InputSpec`. The computed `forward_inputs` property is the execution adapter; it is not a second serialized source of truth. Old projects and runs load without migration. Validation rejects duplicate/invalid names, positional arguments after keywords, per-input or aggregate numeric budget violations, and multi-input blueprints.

The API checks every uploaded reference on project create/update/run. Before executing source, the worker verifies and privately loads every numeric upload. It constructs the model with the legacy primary-input seed/dtype rules, binds placeholders against the actual bound `forward` signature, then builds every input with its own generator and dtype. Signature mismatch produces a named error before numeric generation. A single recorder captures all inputs as independent named roots with their own axis labels, and invokes `model(*args, **kwargs)` so module hooks still run. Keyword inputs participate in module boundaries and operation dependencies through the existing recorder. Shapes-only mode creates only meta tensors. Immutable project/run JSON includes the entire call configuration; changing an auxiliary input invalidates the displayed-run signature.

`ForwardInputs` owns collapsible input cards and the shared recording mode; `TensorInputFields` reuses the input library, upload provenance, and random-stream controls. Collapsed cards stay mounted so invalid unfinished fields cannot disappear from validation. The builder's `ComponentToolbar` displays icon categories horizontally with keyboard navigation; the left rail retains workspace navigation only. The component panel returns focus to its opening toolbar control on Escape.

Checks cover numeric cross-attention against PyTorch, keyword-only calls, mixed floating/integer inputs, module boundaries, partial failures, signature errors, independent RNG streams, old projects, saved histories, additional uploaded assets, and large shape-only inputs. Browser checks cover adding/removing inputs, invalid collapsed fields, a four-input run, toolbar navigation, and narrow layouts.


## Immutable model checkpoints

`SavedWeights` pins an opaque ID, display name, original filename, canonical-file SHA-256, byte count, timestamp, and bounded tensor manifest. `ProjectDraft.weights` defaults to null, preserving existing projects and seeded behavior. The `weights` SQLite table holds immutable library metadata; `.data/weights/<id>.pt` stores normalized CPU tensor values and standard state-dictionary version metadata. APIs list/import assets and explicitly check a draft's compatibility. Neither import nor selecting a checkpoint executes project code.

`weights.py` validates archive sizes/counts and the uncompressed PyTorch format, restricts deserialization with `weights_only=True`, checks metadata on the meta device before materializing values, and normalizes a fresh tensor-only state dictionary. Import uses a disposable worker with a 20-second deadline. There is no unrestricted loader fallback or custom allowlist. API project save/run/check paths require exact library metadata. The execution worker checks the canonical checksum before user source, constructs the model using the existing primary-input dtype/seed rule, compares exact keys/shapes/dtypes, and loads with strict matching. Weights attach to the complete model, including builder-generated module names. No remapping or partial loading is performed.

Compatibility checks share the execution lock and run in a separate worker, construct on meta, and never call forward. `Trace.weight_check` reports compatibility; numeric and Shapes-only runs retain it with the pinned project snapshot. Model parameter snapshots contain the actual loaded values. Standard buffer metadata is retained, and copying into a fresh model prevents forward mutations from altering the canonical asset. The checkpoint reference does not capture a full environment or optimizer/training state.

`WeightLibrary` is shared between Code and Build Setup. Import saves an asset for explicit application, searchable tensor manifests render at most 20 rows, and reports are shown only for the exact checked draft signature. The run signature includes checkpoint identity in both builder and code modes. Recorded run details expose the checkpoint reference rather than whichever weights the current draft selects.

Validation covers numerical outputs against known weights, parameter snapshots, module buffers/version metadata, mixed dtypes, constructor changes, strict mismatches before forward, metadata-only checking, shape-only runs, in-place mutation isolation, old projects, persistence, tampering/missing assets, wrapper formats, malformed/oversized files, and restricted-object rejection. Browser acceptance covers importing/applying, incompatible constructors, switching back to initialized state, historical provenance, and narrow layouts.


## Shifted-window components and lessons

`window_attention.py` extends the reusable attention code with `WindowScoreAdjustment`, `RelativeWindowAttention`, `ShiftedWindowTransformer`, and `AlternatingWindowPair`. Learned offsets index a `(2M−1)² × heads` table with query-minus-key coordinates. A Boolean mask distinguishes wrapped regions after a negative spatial roll; blocked scores use `-inf` before softmax. Per-axis shifts disable on single-window dimensions. Partition/restoration, residual paths, and MLPs remain inspectable operations. Composer checks head divisibility, spatial divisibility, parameter counts, and attention allocation before constructing numeric tensors.

The frontend proves the seven-operation adjustment chain (lookup, permutation, bias add, grouped reshape, mask, reshape, softmax), tensor shapes/dtypes, edges, and non-mutating successful boundaries before offering its lesson. `WindowScoresView` uses selected logical indices and bounded value requests for scores, bias, masks, lookup, and probabilities. It does not infer mask values in metadata mode. `roll` emits a compact mapping descriptor; modular inverse/forward coordinate transforms handle repeated/negative axes and flattened rolls in bounded memory. Tests use an independent coordinate-gathered shifted-attention calculation, not the implementation's partition or mask helper, and include rectangular windows, disabled shifts, exact blocked weights, variable batches, and large metadata runs.

## Explicit builder dependencies

`ComponentSpec.sources` optionally names one earlier tensor (ordinary components) or two (add/concatenate/stack joins). `input` is reserved. Unspecified ordinary sources retain previous-output semantics; joins default to previous plus model input. `compose` tracks shape, axes, and generated variable names per source ID, validates a forward-only DAG, and emits explicit Python references. It retains the final-component output contract and validates both sources before joining. `ComponentStage.source_shapes` powers both incoming shape summaries. Custom-component checks use the selected incoming shape and complete generated graph.

`connections.ts` mirrors only the source defaults, while backend inference remains authoritative. `BranchConnections` measures actual cards and draws their explicit edges at the current zoom. Source tags navigate to producers. Invalid explicit references remain visible for repair after reordering/removal. Run continues to record actual data dependencies independently of authored connections.

## Source snapshots, Git, and environments

`ProjectDraft` adds relative `entry_path`, `import_root`, bounded additional `files`, optional imported-repository provenance, and an optional opaque environment ID. The entry remains in `code` for backward compatibility. File trees are limited to 128 UTF-8 source/configuration files, 500 KB each, and 2 MB total; absolute paths, traversal, `.git`, control characters, and null bytes are rejected. Blueprints cannot supply additional sources or a separate execution environment.

`source_projects.import_git` reads a temporary bare fetch and resolves a commit; it never checks out or imports source. Global/system Git configuration, credential helpers, and hooks are disabled. Regular supported text blobs are read by object ID; symlinks, submodules, and unsupported/binary files are omitted. HTTPS public repositories and absolute local clones are supported. Commands have process-group cleanup and deadlines; import has a total 90-second deadline. The source snapshot pins URL, commit, subdirectory, and the imported tree hash. Subsequent project edits remain independently stored, not represented as unchanged commit contents.

For explicit execution, `project_namespace` writes a private temporary tree, sets the selected import root on Python's path, gives the entry normal package semantics, and runs from the source root. Recorder source resolution maps captured frames to relative filenames and source-local assignments/axis annotations. Import, syntax, and forward failures retain those filenames. All sources and configuration are embedded in immutable run JSON; inspecting history never reads the current draft or live repository. Single-file legacy projects retain the original compile path.

`environments.py` creates a local venv only on an explicit setup request, bridges the backend's site-packages, and installs user-supplied exact PyPI pins with `--only-binary=:all:`. No requirements file is executed implicitly. Setup shares the run lock, has child-process deadlines/cleanup, and removes failed partial environments. The manifest records requested pins and available versions. Environments inherit base dependencies, so workers independently report their actual Python and distribution versions on every run. These are reproducibility records, not a guarantee of a hermetic runtime. Both Run and checkpoint compatibility checks resolve the selected environment through its opaque ID.

`GitImport`, source-file helpers, `ProjectEditor`, and `EnvironmentSetup` separate read/import, edit/configure, dependency installation, and execution. Filenames follow operations into focus and code inspection. Tests cover committed versus working-tree content, relative imports, package roots, source names/annotations, import/syntax/forward errors, traversal/size limits, symlink omission, snapshot persistence, invalid dependency specifications, separate-environment execution, and signature compatibility for legacy projects.


## Joining and partitioning tensors

`operations/assembly.py` validates the recorded operands and outputs for `cat`/`concat`/`concatenate`, `stack`, `split`, `chunk`, and `unbind`, then supplies the `tensor_assembly` capability. The recorder captures positional and keyword dimensions and preserves list operand order, including repeated references. Axis names propagate to every output; stacking inserts a `stack` label, unbinding removes that label, and joins preserve labels only where all inputs agree. Explicit source annotations remain authoritative. Dtype-changing joins, unsupported rank mismatches, and wholly empty tensors use generic inspection; mutations keep their storage lesson.

`assembly.ts` independently validates shapes, dtype, axis, successful status, and non-mutating boundaries before offering an exact coordinate mapping. The actual output list is authoritative for split sizes and chunk counts. Cumulative intervals require O(parts) metadata; whole-to-part selection uses a binary search plus O(rank) coordinate conversion. Stacking/unbinding insert/remove an index without allocating an element map. Empty intervals are skipped when selecting a whole-tensor cell, and repeated input IDs retain distinct operand positions.

`AssemblyView` renders one part and the whole tensor using the shared paged `TensorCard`/3D explorer. At most six part buttons plus labeled gaps are visible, with a direct index control for every recorded part. Selecting either side synchronizes the other; empty parts have no fabricated coordinate. Readouts request only selected captured values and distinguish logical coordinates from recorded storage identity. Shape-only runs retain structural navigation. Stack branches is a two-input builder tool with insertion-axis controls, exact shape checks, and the existing rank/allocation limits.


## Convolution neighborhoods

`operations/convolution.py` validates batched/unbatched Conv1D/Conv2D operands, matching floating dtypes, bias size, grouped channels, and exact output geometry before supplying the `convolution` interaction. Argument capture now names Conv1D stride, padding, dilation, and groups and preserves keyword operand order. Complete ungrouped patch tilings keep `patch_projection`; the general neighborhood adapter handles other supported convolutions. String `same` padding splits effective kernel size minus one into floor/ceiling before/after padding, while `valid` uses zero.

`convolution.ts` independently validates recorded geometry and computes one selected product with O(rank) metadata. A padded coordinate has no logical input index. Output channel determines its input group, and kernel channels remain local to that group. Input-to-output selection solves the stride/dilation integer equation using the Euclidean algorithm and bounded candidate intervals; it does not scan either tensor. BigInt modular arithmetic avoids intermediate precision loss. Unreachable input cells produce a clear notice. Weight selection preserves the current batch and spatial output position.

`ConvolutionView` reuses `TensorCard`, the 3D explorer, direct-index controls, and paged snapshot reads. At most 256 products are fetched for a complete sum; larger kernels use an eight-term window, labeled as partial. Virtual padding contributes zero, while absent or non-finite stored values never become fabricated finite results. Shape-only traces show geometry without numeric calculations. Browser arithmetic remains approximate and the immutable recorded output stays authoritative. Explicit nonzero padding operations are separate predecessors, so the convolution maps into their recorded padded tensor rather than inventing a link to an earlier source.

Validation compares all output cells against both native PyTorch and independent grouped receptive-field summation. Frontend reference fixtures use exactly representable inputs/weights; a backend check verifies their recorded outputs directly with PyTorch. Coordinate tests exhaustively compare inverse selection with brute-force small geometries, and cover rectangular kernels, asymmetric same padding, batches, groups/depthwise multipliers, bias, non-contiguous tensors, skipped cells, old positional arguments, unsafe geometry, and billion-element metadata with bounded contribution windows.


## Pooling neighborhoods

`operations/pooling.py` validates batched/unbatched floating-point 2D max, average, and adaptive-average pooling before supplying the `pooling` interaction. Positional and keyword arguments are captured by name. Geometry checks include output-size rounding and the ceil-mode correction that excludes windows starting beyond the input. Returned maximum indices must have the output shape and int64 dtype. All validated pooling outputs preserve input axis names unless an explicit source annotation overrides them. Unsupported or inconsistent traces retain general inspection.

`pooling.ts` independently proves the recorded geometry. Regular windows use stride, padding, and max-pooling dilation. Average-pooling regions stop at the declared padded extent: ceil overhang beyond it does not contribute to the divisor. The divisor is either the explicit nonzero override, the declared padded region area, or the real sample count. Adaptive bounds use exact integer floor/ceiling ratios with BigInt intermediates; non-divisible and enlarged outputs can overlap. Reverse selection solves regular stride/dilation geometry through shared `tensors/windowGeometry.ts`, also used by convolution, or computes adaptive output intervals directly. Neither path scans the input or output tensor.

`PoolingView` links the existing 2D/3D tensor explorers and paged values. It fetches at most 256 samples for a complete calculation, otherwise an eight-position window, and distinguishes virtual padding from stored cells. A captured max-pooling index is interpreted within the selected input spatial plane and validated against the region before navigation. Without indices, only a complete finite window matching the recorded maximum can identify ties; partial or metadata-only windows never invent a winner. Average browser arithmetic is approximate; recorded outputs remain authoritative. PyTorch-generated fixtures are checked by both backend and frontend tests, alongside ceil/divisor, adaptive overlap, ties, non-finite, non-contiguous, and very large shape cases.

Pooling, convolution, linear projection, and patch embedding views load on demand through React Suspense; their validation helpers stay available for operation routing. This keeps lesson growth out of the initial application bundle.


## Layer normalization groups

`operations/normalization.py` validates exact trailing normalized dimensions, matching floating dtypes and output shape, nonempty operands, finite nonnegative epsilon, and affine tensor shapes. Argument capture records explicit weight/bias tensor-or-null roles, so bias-only calls cannot be mistaken for scale-only calls. Semantic keyword ordering remains handled by the recorder. Explicitly unambiguous older traces are accepted; a two-operand trace with no affine-role evidence uses general inspection. The `layer_normalization` capability is independently validated by `layerNormalization.ts`.

Group and feature indices use logical row-major coordinates, independent of physical strides. A group contains the product of the trailing normalized dimensions; leading coordinates select the group, and trailing coordinates select its shared scale/bias entry. The frontend allocates at most 256 indices for complete group statistics, or eight for a selected window. No map proportional to tensor size is created. Statistics use a two-pass population-variance calculation with a shifted mean to retain small differences and exact constant groups. Epsilon is inside the square root; explanatory arithmetic remains approximate relative to PyTorch. Incomplete, missing, non-finite, or overflowing values never supply a fabricated finite result.

`LayerNormalizationView` loads on demand and shares the tensor explorer, 3D dialog, index controls, and paged snapshot hook. Its mean/variance/normalize/affine progression is explanatory UI, not synthetic trace nodes. Parameter selection preserves the current leading group; selecting either main tensor updates the paired coordinate and group. Shapes-only traces keep structural exploration. Larger numeric groups show full-group statistics alongside a bounded cell window and the recorded output. Tests independently compare population formulas, module/functional calls, reordered keywords, bias-only calls, multiple axes, non-contiguous inputs, constants/zero epsilon, non-finite values, and billion-element metadata. Shared PyTorch fixtures verify frontend reconstruction of every small-case output.

Large groups use `GET /api/v1/runs/{run_id}/operations/{operation_id}/normalization?group=…`. The endpoint validates the recorded LayerNorm operation and group bounds, then reads only the immutable input snapshot. `statistics.py` scans it twice in 16,384-element float64 chunks, respecting the existing 8,388,608-element numeric tensor limit. It validates snapshot shape, dtype, and logical contiguous layout; non-finite or overflowing groups return explicit statuses with null statistics. Missing/corrupt snapshots return a recoverable 410, while shapes-only traces return 409. The constant-size response contains group identity, count, mean, population variance, and denominator. No user code is executed and no trace nodes are added.

The frontend validates response identity and completeness, cancels stale requests, and keeps at most 32 immutable summaries in an LRU cache. Changing features within a group reuses statistics; changing groups, operations, or runs cannot display a previous selection's result. Values-hidden and shapes-only views do not request statistics. Endpoint tests cover multiple axes, logical order after transpose, later in-place mutations, persistence after restart, corrupted snapshots, low-precision input dtypes, chunk bounds, constants, non-finite inputs outside the visible window, and invalid requests. Client tests cover complete-group arithmetic, cache bounds, cancellation, response mismatch, and retry.
