# TensorViewer

A local learning workspace for seeing what PyTorch operations do to tensors. Build a model from the component toolbox, paste an `nn.Module`, or upload a Python file, then inspect an actual forward pass. Git source snapshots are also available under More sources.

Two independent projects live here:

- **`backend/`** — Python, FastAPI, PyTorch, SQLite, and an execution worker.
- **`frontend/`** — React, TypeScript, Vite, Tailwind, and reusable SVG tensor renderers.

## Start locally

Requirements: Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), and Node.js 22 or newer with npm. The projects each have their own lockfile.

In one terminal:

```sh
cd backend
uv sync --locked --python 3.11
uv run uvicorn tensorviewer.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. The frontend proxies `/api` to the backend. Interactive API documentation is at http://127.0.0.1:8000/docs.

On an empty workspace, the application offers the same **New project** chooser used later: **Build visually**, **Paste code**, or **Upload code**. All three use the same **Inputs**, **Generate diagram**, and **Diagram / Code / Runs** workspace. Project creation stores the model; it does not run the code. Existing projects and saved executions remain available.

## First walkthrough

1. Choose **New project** and name it. **Build visually** opens an empty canvas. **Paste code** accepts a PyTorch module directly; **Upload code** reads a `.py` file up to 500 KB. Review the suggested class to run; constructor arguments are optional JSON. Class detection is only a suggestion, so aliases or inherited modules can be entered manually.
2. For a visual model, open a category in the horizontal component toolbar. Its searchable panel is collapsed by default. Use Left/Right or Home/End to navigate the toolbar, and Escape to close the panel. Adding a component closes the panel and opens its settings; a first component supplies a compatible starter input. **Setup** edits the project name.
3. Open **Inputs** to set the tensor shape and values. This is the same panel for every project, including the input node on a manual canvas. Type, seed, and axis labels live under **Advanced settings**; saved inputs and NumPy files have their own disclosure. **Shapes only** supports large tensors without allocating values. Source models can add multiple forward inputs and configure argument binding.
4. On a manual canvas, select a component to configure, reorder, or remove it. Connections display inferred shapes. If something is missing or invalid, the primary button becomes the next useful action: **Set up input**, **Add component**, **Set project name**, or **Review inputs / code / model**. Review opens the relevant settings, including collapsed fields. For example, insert **Image to tokens** between convolution and attention.
5. Click **Generate diagram** in the header or at the bottom of **Inputs** to save and run the model. Both buttons offer the same next action. For custom visual components, generation checks their shapes first; there is no separate required preview step. Every path displays the same recorded tensor diagram, playback, node inspection, and code links. Select a node to inspect its transformation, or a cell / **3D** control to enlarge one tensor.
6. Use **Inputs** and **Run again** to try a different tensor. **Edit model** returns to component editing for a manual model or source editing for a code model. **Code** edits the current project; model and environment settings stay collapsed until needed. **Executed code** opens a sidebar containing the source saved with that run; tensor exploration stays in the center. **Runs** holds previous executions. Reopening any project with a saved run returns to its diagram.

Opening **Inputs**, **Code**, or **Runs**, or editing a manual model, preserves your place in the recorded diagram. The current node retains its selected cell, viewing axes, slices, lesson settings, and scroll position. **Back to journey** and reopening that same node also preserve its inspection state. Selecting a different node or loading another run starts a new inspection; the app does not keep an inspection history for every node. Hidden mapping replays pause and require Play to resume, and an open 3D dialog closes when its inspection is hidden. If the current code or inputs differ from the displayed run, the relevant panel explains the difference; use **Run again** to generate an updated diagram.

**Diagram**, the panel close button, and **Escape** return to the model canvas or recorded diagram you were using. **Code** and **Inputs** also share a **Back to model / Back to diagram** button and the next-step action at the bottom. Closing a panel preserves unfinished fields; it does not save or run the model. Choose **View last run** on the model canvas to switch explicitly to its saved execution.

Unfinished fields show **Finish edits** instead of **Saved**, even when an invalid value has not yet been applied to the model. Choose it to return to the setting that needs attention. Once the edits are complete, saving and generation become available again.

If a run stops, its earlier steps remain available. Choose **Fix code** (or **Edit model** for a visual model) or **Edit inputs** beside the error. A source error opens its recorded line only when that file is unchanged, so the editor does not point at an unrelated line after edits.

Generated Python remains read-only while attached to a visual model. You can select and copy it; editing controls appear only after choosing **Use as custom code**. This secondary conversion is not required to paste or upload a new model. It is one-way, and earlier runs retain their blueprints. Model code executes only on explicit **Generate diagram / Run again**, **Preview custom shapes**, or **Check compatibility** actions. Editing settings does not execute custom code.

**All tools** opens a compact category browser. Pick a category or search across the entire library using names such as “CNN”, “linear layer”, or “multi-head attention”. Search accepts words in any order. Select a result to add it and open its settings; the **Custom** category holds saved modules and the option to create your own.

The library contains 43 built-in components across Models, Spatial, Sequence, Layers, Activations, and Shape adapters, plus your saved Custom components. These include add/concatenate/stack joins, shifted-window transformers, regular/shifted pairs, a complete Vision Transformer, a two-stage hierarchical vision model, window partition/restoration, window transformers, patch merging, spatial embedding/classification, class-token/position preparation, a class-token classifier, patch embedding, self-attention, transformer blocks, feed-forward networks, 1D/2D convolution, simple RNN, three pooling variants, linear projection, layer/batch normalization, five activations, image-to-token conversion, flatten, transpose, split/merge axes, insert/remove unit axes, unfold, contiguous, and mean reduction. Search always searches the complete library, even from a category panel. Batch normalization uses initialized running statistics in evaluation mode. The Custom category holds reusable modules you save locally.

## Follow a complete Vision Transformer

From an empty canvas, add **Models → Vision Transformer**. It supplies a `[2, 3, 8, 8]` image input and starts with 4×4 patches, 8 embedding features, 2 attention heads, 1 transformer block, and 10 output classes. Run and expand the recorded stages to follow image → patch tokens → class token + learned positions → transformer blocks → class-token classifier. The default sequence changes `[2, 3, 8, 8] → [2, 4, 8] → [2, 5, 8] → [2, 5, 8] → [2, 10]`.

The same model can be assembled from **Patch embedding → Class token + positions → Transformer blocks → Class-token classifier**. Token 0 is the learned class token; image patches follow in row-major spatial order. Positions are learned parameters added to every batch. The classifier normalizes the sequence, selects token 0, and produces logits without softmax. Parameters start untrained; this example teaches the computation, not meaningful image classification.

Patch size must divide both image dimensions. Positional parameters have a fixed sequence length for the configured image resolution; rebuilding for a different resolution does not interpolate saved positional weights. Shape checks include the extra class token in attention's quadratic allocation and count all learned parameters. Large examples can use **Shapes only**. The original explicit attention and transformer implementations remain shared with the individual toolbox components.

## Follow a spatial hierarchy

From an empty canvas, add **Models → Hierarchical vision**. The default `[2, 3, 8, 8]` image becomes a `[2, 8, 4, 4]` patch grid. A transformer processes independent 2×2 windows, patch merging produces `[2, 16, 2, 2]`, and a second window transformer processes this coarser grid. Final normalization, spatial averaging, and a classifier produce `[2, 10]` logits. Expand the recorded stages to follow the actual operations.

The same model can be assembled from **Spatial patch embedding → Window transformer → Patch merging → Window transformer → Spatial classifier**. Set both window transformers to 2 heads to match the full model's defaults. Partition and restoration are also independent tools: `[B, C, H, W] → [B × windows_per_image, window², C]` and back. The packed leading axis keeps each window and image independent while sharing transformer weights.

Select a partition, restoration, or patch-grouping stage or one of its layout operations. The grouping lesson shows all five recorded shapes through four reshape/permute steps. Selecting either tensor, including its enlarged 3D view, highlights the corresponding cell and neighborhood in the other. The coordinate explanation identifies the original batch, grid location, feature, and resulting window/token or feature slot. Highlights cover at most 8×8 cells, with the existing paged explorers available for large tensors. Shapes-only runs retain mappings without invented values.

Patch merging groups 2×2 neighbors in top-left, bottom-left, top-right, bottom-right order. Grouping preserves values; subsequent LayerNorm and a learned `4C → 2C` projection compute new values. Both grids must divide evenly into windows, and merging requires even spatial dimensions. The complete model therefore requires image height and width divisible by `patch × 2 × window`. Numerical checks account for local attention allocations and model parameters; use **Shapes only** for larger examples.

This is an untrained educational hierarchy with fixed, non-overlapping windows. It has no shifted windows, relative-position bias, dropout, automatic padding, or class token; it is not a complete Swin implementation. Shifted attention is available separately through the reusable components below.

## Follow an addition across tensors

Select a tensor–tensor `add` step, including positional embedding and residual additions. The addition lesson links both operands to the selected result cell. Coordinates align from the last axis; size-one dimensions use index 0, and missing leading dimensions broadcast. Selecting an operand keeps the current output batch and slice wherever that operand broadcasts. Same-shape residual additions pair matching coordinates directly.

The calculation uses captured values and honors PyTorch's `alpha` multiplier. The recorded output remains authoritative. Both tensor explorers support paging, coordinate jumps, and enlarged 3D selection; the calculation requests only the selected values. Shapes-only runs retain the coordinate relationships without invented numbers. Scalar Python operands, empty tensors, and unsupported layouts retain ordinary inspection; in-place writes keep their dedicated storage-change view.

The builder supports up to 16 components with forward-only source connections. Transformer blocks include pre-norm attention, residuals, and a feed-forward network; multiple blocks are configurable. The explicit RNN supports up to 32 time steps. Each node can read any earlier tensor; add, concatenate, and stack joins combine two paths. Git source projects use the same execution viewer.

## Follow a tensor through joins and splits

Select a recorded `cat` (including `concat`/`concatenate`), `stack`, `split`, `chunk`, or `unbind` operation. The lesson pairs one selected part with the whole tensor. Selecting a cell on either side—including in its enlarged 3D view—links the exact source and destination coordinates. Selecting the whole tensor also opens the matching part. Concatenation extends an existing axis; stacking inserts an axis that identifies the input tensor. Neither adds values together.

Part cards show actual recorded shapes and axis ranges. Small lists show every part; large lists show a bounded set with labeled gaps and a direct part-index control. Split outputs may have different sizes or be empty, and `chunk` can return fewer parts than requested. Empty parts are labeled without inventing a cell. Recorded values and storage identities remain authoritative; Shapes-only runs use the same coordinate navigation without numbers. Mappings use logical coordinates, including for non-contiguous inputs, and do not allocate a map per tensor element.

Use **Layers → Concatenate branches** or **Stack branches** in the builder. Split/chunk/unbind lessons work with custom code; the builder still expects one tensor output per component. Dtype-promoting joins, unsupported mixed-rank empty inputs, wholly empty tensors, and `out=` mutations retain their ordinary or storage-change inspector.

## Follow a convolution neighborhood

Add **Spatial → 2D convolution** or **Sequence → 1D convolution**, run, and select its convolution node. The lesson links the actual input, kernel, and output. Select an output to see the input cells and weights that contribute to it, or select a kernel entry to inspect one product. Input selection finds a nearby output that uses that cell, retaining the current output when possible; cells that stride or dilation never samples are explicitly identified. All three explorers support direct coordinates, paging, and 3D selection.

The calculation displays `input position = output position × stride − padding before + kernel position × dilation`. PyTorch uses cross-correlation, so the kernel is not flipped. Out-of-bounds coordinates are virtual zero padding, not invented input cells. Channel groups restrict each output channel to its own input channels; kernel channel indices are local to that group. Recorded shapes and arguments validate every mapping. The lesson supports batched/unbatched 1D and 2D floating-point convolutions, rectangular kernels, stride, dilation, groups/depthwise convolutions, numeric zero padding, and `valid`/`same` padding. These advanced settings can be supplied in custom code. Explicit reflection/circular padding remains its own recorded operation; the convolution uses that operation's actual output.

Small kernels show all products plus bias; more than 256 products use an eight-term window with an explicitly partial sum. The recorded output remains authoritative because browser accumulation and rounding can differ from PyTorch. Large shapes do not allocate a neighborhood map; Shapes-only runs retain coordinate exploration without numerical calculations. Empty tensors, unsupported dtypes, transposed/3D convolutions, or unverified geometry keep general inspection. Complete patch projections retain the existing patch-to-token lesson.

## Follow scores through softmax

Select a recorded `softmax` operation, including one inside Attention, to follow **maximum → shift → exponentiate → normalize**. Select either the score or weight tensor in 2D or 3D; the lesson pairs the same coordinate and identifies its complete normalization group. **Group**, **Score**, and the eight-row window reach any position, including groups along a non-last axis. Shape and coordinates stay unchanged.

The calculation shows the whole group's maximum and exponential total, with the recorded PyTorch weight displayed separately. The table links scores to shifted values, exponentials, and actual recorded weights; bars use a fixed 0–1 scale. Small groups calculate from complete captured values in the browser. Groups larger than 256 scores use a bounded backend scan of the saved input and a cached summary, so a visible window never substitutes for the full denominator. The existing numeric tensor limit still applies.

Negative-infinity scores contribute zero when a group contains a finite score. Entirely negative-infinity groups, NaN, and positive infinity have explicit unavailable reference states, while their recorded outputs remain visible. Finite scalar softmax is supported. Shapes-only runs retain group and coordinate navigation without numbers. The lesson requires an explicit axis and matching floating input/output dtypes; implicit axes, dtype conversions, empty tensors, and mutations retain general inspection. The existing **Window bias and mask** lesson keeps priority for its recognized shifted-window stages. See the [PyTorch softmax definition](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html).

## Follow mean and sum reductions

Add **Shape → Mean reduction**, or call `mean` / `sum` in custom code. Select the recorded operation to see which axes are reduced and which stay fixed. For example, `[B, tokens, features].mean(1)` combines all tokens separately for every batch and feature, producing `[B, features]`. `keepdim=True` retains the reduced axes with size one.

Select an input or output cell to follow its exact reduction group, including in the enlarged 3D explorer. **Output cell** and **Contributor** reach any group and member. The lesson shows **gather → add → divide for mean**, with the real recorded output alongside the explanation. The contributor strip displays up to eight cells with counted gaps; its table provides all coordinates and a window control. Nonadjacent, negative, multiple, and all-axis reductions work with the same navigation, including scalar results and non-contiguous input views.

Reference arithmetic covers the complete group. Groups of up to 256 cells calculate in the browser; larger groups calculate from the saved snapshot on the backend in bounded chunks, including groups with nonadjacent reduced axes. Only the eight-cell contributor window and a compact summary reach the browser. Moving within a group reuses its summary; selecting another output loads that output's group. The existing limit of 8,388,608 numeric elements per tensor still applies.

Integer references preserve large captured integers exactly. Reference math can differ from PyTorch's recorded result because of dtype rounding, accumulation order, or integer overflow; both results remain clearly labeled. Dtype conversions other than exact widening to int64, non-finite inputs, overflow, and shapes-only runs have explicit unavailable states. Empty tensors and unsupported dtypes use general inspection; `out=` writes retain mutation inspection. Semantics follow the recorded PyTorch calls ([mean](https://docs.pytorch.org/docs/stable/generated/torch.mean.html), [sum](https://docs.pytorch.org/docs/stable/generated/torch.sum.html)).

## Follow activation functions

Add **Activations → ReLU**, **GELU**, **Sigmoid**, or **Tanh**, run, and select the recorded activation. The lesson links the same coordinate in both tensors and their enlarged 3D views. It shows **captured input → function → recorded output**, alongside a reference curve with selectable points from an eight-element window. Select a point with a click or keyboard Enter/Space, or choose a row in **Inspect captured pairs**. The **Element** control reaches any logical index without loading the complete tensor.

The curve initially shows inputs from −4 to 4, where the nonlinear behavior is easiest to see. **Fit visible cells** includes captured pairs outside this range. Overlapping points remain individually selectable in the table. The line is a mathematical reference, not extra execution data; dots use the actual captured input and output. Shapes-only and values-hidden views retain the labeled reference without fabricated tensor values.

GELU respects the recorded `approximate="none"` or `"tanh"` mode; its Gaussian-CDF curve uses a browser approximation and the captured output remains authoritative. ReLU clips negative values to zero; GELU can retain small negative values. Sigmoid and tanh saturate independently per cell; they do not normalize an axis. See the PyTorch definitions for [ReLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ReLU.html), [GELU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html), [Sigmoid](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Sigmoid.html), and [Tanh](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Tanh.html).

The lesson supports scalar and nonempty floating tensors, non-contiguous inputs, and integer ReLU. Non-finite values and integers beyond browser numeric precision stay visible as captured text and are excluded from the finite plot. In-place/out variants retain the shared-storage mutation inspector. Empty tensors, unsupported dtypes, and unknown GELU modes use general inspection.

## Follow layer normalization

Add **Layers → Layer normalization**, run, and select its `layer_norm` operation. The lesson links the input and output at the same coordinate, highlights the normalization group, and shows **mean → population variance → normalize → scale and shift**. These calculations explain the captured operation; they are not additional execution nodes. Each output depends on the entire group.

The normalized shape selects the final one or more dimensions. For `[B, tokens, features]` with normalized shape `[features]`, each token in each batch is an independent group. A normalized shape `[height, width]` groups both spatial axes. The **Group** and **Feature in group** controls reach any coordinate, and the 3D explorers stay linked. Expand **Inspect scale and bias** to select a parameter while preserving the current group.

Variance divides by the group size, and epsilon is inside the square root. Scale and bias are per normalized coordinate and shared across leading groups. LayerNorm uses the current input's statistics even in evaluation mode. Epsilon and affine parameters mean the final output need not have exactly zero mean or unit variance. See the [PyTorch LayerNorm definition](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html).

The lesson supports nonempty floating-point inputs, multiple trailing axes, optional scale/bias (including bias-only functional calls), and non-contiguous tensors. Groups of up to 256 elements calculate statistics in the browser. Larger groups, including common widths such as 768 and 1,024, calculate full-group statistics from the saved snapshot on the backend in bounded chunks. The browser still loads an eight-element window and the selected output/parameters; it never substitutes a partial-group mean or variance. The existing numeric capture limit of 8,388,608 elements per tensor still applies. Shapes-only runs show group boundaries and parameter coordinates without numeric statistics. Non-finite values, zero denominators, unsupported mixed dtypes, and ambiguous older affine operands retain explicit unavailable states or general inspection. Multi-axis normalization and advanced parameters are configured in custom code.

## Explore pooling windows

Add **Spatial → Max pooling**, **Average pooling**, or **Adaptive average pooling**, run, and select its pooling operation. Each output links to its exact input window; input selection finds a nearby output that samples it. Both tensor explorers support direct coordinates and 3D selection. Batch and channel remain independent.

Max pooling uses virtual negative-infinity padding. If custom code returns indices, **Follow recorded maximum** selects the actual recorded source cell. Without indices, a complete finite window can identify matching maxima and ties, but does not invent a winner index. Average pooling shows its sum and exact divisor, including `count_include_pad`, `divisor_override`, and truncated `ceil_mode` edge windows. Adaptive averaging shows size-ratio regions, including overlaps and unequal region sizes.

The first lesson supports batched/unbatched floating-point 2D pooling, rectangular windows, stride, max-pooling dilation, padding, and adaptive output sizes. Advanced parameters and returned maximum indices are available through custom code. At most 256 samples are loaded for a complete calculation; larger regions use an explicitly partial eight-position window. Shapes-only runs retain geometry and navigation without numeric winners or values. Unsupported variants keep general tensor inspection.

## Reusable custom components

1. Open the **Custom** puzzle icon in the component toolbar and choose **New custom component**.
2. Paste a module or use **Import .py**, select its class, name it, and supply default constructor arguments as a JSON object. **Save & add component** saves it to the library and inserts a copy in the current sequence. Saving checks Python syntax without executing it.
3. Edit **Constructor arguments** on a node. Valid JSON applies automatically to that node; library defaults stay unchanged. Unfinished invalid edits survive switching panels or components. Finish them or choose **Revert to applied arguments** before saving or running.
4. Choose **Generate diagram / Run again**. Before recording values, the backend checks the custom modules and assembled sequence on PyTorch meta tensors in separate worker processes. This checks connections, non-contiguous inputs, and recording limits without allocating the requested numeric tensors. A failed check opens the model settings; a successful check continues into the recorded run. Source lines and variable names inside custom modules remain linked to the generated code.
5. Optionally choose **Preview custom shapes** in the builder to inspect connections before generating a diagram. This is an explicit preview action; changes to settings alone never execute the custom module.

The first custom-component contract is `nn.Module.forward(x) → tensor`, with one to six non-empty dimensions and an unchanged dtype. Use installed dependencies; modules that require numeric values to determine their output cannot be shape-checked in the builder. They can still be explored as ordinary custom-code projects with small value inputs.

Library entries are immutable. **Edit source as new version** saves a new entry and updates only the selected node. Other projects and past runs retain their original source and arguments. Multiple components can define the same Python class name: each has its own namespace. Check results are bounded, session-local caches tied to the exact code, arguments, input configuration, and sequence. Changing the sequence or restarting the backend may require another check. Every check and run uses the same local trusted-code execution model described below.

## Follow a patch into a token

Add **Models → Patch embedding**, choose a patch size, and run the sequence. Select its convolution, flatten, or transpose node to open the patch lesson. The progression follows the recorded image → projected patch grid → flattened grid → token rows, with exact shapes and selected coordinates at each stage.

Select a spatial patch to highlight its source image region, then choose a batch, input channel, pixel, and output feature. The paired pixel and kernel grids show matching recorded operands. The calculation identifies one contribution to the feature; the full projection sums every channel and patch position, plus bias. Flatten and transpose preserve the projected values. **Tensor details** returns to the regular tensor inspector and 3D viewer without discarding the lesson selection.

The lesson handles complete, non-overlapping, unpadded Conv2D patch projections, including rectangular kernels in custom code. Other supported convolutions use the convolution neighborhood lesson; unverified variants retain general inspection. Large images and kernels use windows of at most 8 × 8 cells, with direct index controls; feature lists show at most eight entries. Shapes-only runs show coordinates and transformations without numeric values.

## Explore recorded stages

New runs record module-call boundaries alongside individual operations. Runs with more than 24 operations start with their outer stages collapsed. Use **Stages** to browse nested calls, **Stage overview** to fold the model, or **All operations** to see the complete graph. Expanding a stage opens one level at a time, so transformer blocks can reveal their attention and feed-forward stages before exposing every operation. Calls with fewer than two recorded operations stay as ordinary operation nodes.

Each stage card displays its actual returned tensor. Select its header to inspect recorded entry/return tensors, choose among multiple operands or outputs, open the existing 3D viewer, or jump to any operation inside the call. Repeated calls to the same module remain distinct. Connections crossing collapsed stages retain their tensor dependencies, including branches and residual paths. Failed calls remain visible without an invented return value.

Playback and code navigation open the containing stages as needed. **Reveal steps** expands the graph and disables folding until it is turned off. Grouping is a viewing preference and never changes model code or saved execution. Older runs without module-call metadata retain their original operation view; execute again to record stages.

## Indexed 3D tensors

The same orthographic cell geometry is used in node previews and the enlarged viewer. `[B, N, D]` is one volume with X = D, Y = N, Z = B. `[B, C, H, W]` shows separate batch volumes with X = W, Y = H, Z = C. Each batch is built from the same cell geometry. For higher ranks, earlier axes are held at explicitly selected coordinates.

Small tensors display all cells as continuous grids. Dimensions up to 16 expand fully when they fit the view’s cell budget; larger or denser previews condense only the necessary axes. Condensed axes retain the first two indices, a labeled gap, the final index, and the selected coordinate. Each gap records its exact omitted interval. Click **…** to open an omitted slice; isolation can expand previously condensed dimensions. Geometry is compressed only where a gap is shown.

Drag to rotate, use rotation buttons, or select Front/Reset. Every displayed cube has real logical coordinates; recorded runs fetch those exact values. Isolate a batch, channel, row, or column to inspect hidden interior cells. Coordinate controls and direct jumps reach any element. Arrow keys navigate width/height; Page Up/Down navigate depth. Node previews display at most 512 cells; enlarged views display at most 1,024. Paged values are loaded in batches of up to 256 indices per request. Before execution, builder nodes explicitly show inferred shapes with no fabricated values.

Drag or scroll to pan, pinch or Ctrl/Command + scroll to zoom, and use **Fit entire journey** for the overview. An expanded node stays readable regardless of canvas zoom; **Back to journey** or Escape closes it and restores your previous pan and zoom. **Fit entire journey**, Home while the canvas is focused, and the minimap explicitly reset the overview. The **Stages** menu lets you group calls or show every operation. Expanding or collapsing stages changes the graph layout, so it establishes a new overview instead of restoring coordinates from the old layout. Playback highlights successive operations; its **Playback options** menu contains **Reveal steps** and **Start from the first step**. The left rail opens **Code** and **Run history** beside the canvas; **Inputs** stays in the header once initial input setup is complete.

**Executed code** is a source-only sidebar: select an executable line to open its operation in the center. Tensor values and explanations have one central inspection view. Specialized lessons still offer **Tensor details** when you want the general tensor inspection controls.

Open **Follow connections** in any enlarged input, operation, or stage to see where its tensors come from and which recorded steps use them next. Select a connection to follow a branch directly, including joins and shared-storage changes. Tensor names, shapes, and input positions identify each connection; long lists show six at a time. This follows actual tensor dependencies, while playback remains in execution order. Returned model tensors are identified separately from results with no later recorded use.

Every tensor is interactive, including inputs and matrix operands. Click a cell for its recorded value, coordinates, logical flat index, and storage position; use arrow keys to move, or Home/End to move to the first/last column. Slice controls keep the selection visible. Large row and column axes show page arrows and direct position fields beside the visible range. **Go to cell**, below the selected value, opens exact coordinate entry in one click. **View** opens viewing axes and 8×8 or 16×16 window settings when applicable. Viewing axes choose which two dimensions to display without changing the recorded tensor. Selecting an input in an overlapping `unfold` highlights matching outputs, including matches on other slices or pages. Large inverse relationships show up to 256 matches.

The Attention example uses randomly initialized parameters, not trained weights. It demonstrates mechanics; its attention patterns have no learned semantic meaning. Drawings show at most 8×8 cells by default, or 16×16 in the denser view, with zero-based index labels and previous/next controls. The focused 2D plane remains available alongside the indexed 3D viewer. Cell text and calculations are rounded; the element readout shows the full recorded value. Color intensity represents value magnitude; paged tensors scale shading to the visible window. Full values are retained independently of the compact grid labels.

Projects and immutable run metadata are saved in `backend/.data/tensorviewer.sqlite3`. Large value snapshots live in `backend/.data/snapshots/`, imported input tensors in `backend/.data/inputs/`, and saved model weights in `backend/.data/weights/`; keep these together when backing up the workspace. Set `TENSORVIEWER_DATA_DIR` to use a different directory. Run history lists the latest 20 runs; older runs remain in the local database.

## Follow a linear projection

Add **Layers → Linear projection**, or select a recorded `linear` step inside attention or a transformer. The lesson links an input vector, the corresponding weight row, and one output feature. Selecting a cell in any of the three tensors, including their enlarged 3D views, updates the others while preserving the appropriate batch/token coordinates. Direct feature controls and the existing coordinate jump reach any index.

The calculation shows the selected product, bias when present, and PyTorch's recorded output. **Inspect matching products** opens an eight-feature contribution window. Projections with at most 256 input features also show a sum over all products; larger projections show an explicitly partial window sum. Neither partial sums nor rounded browser arithmetic replace the recorded result. Shape-only runs retain the same coordinate relationships without numeric values. Standard matrix weights and optional vector bias are supported; other operand layouts retain the general inspector.

## Replay an element's layout change

Select an element in a reshape, view, flatten, permute, transpose, squeeze/unsqueeze, contiguous, or clone step, then choose **Follow element**. Play, pause, scrub, or step through its original coordinate, index rule, and result coordinate. A moving marker follows the actual recorded mapping. The replay explains logical coordinates; storage sharing and recorded storage offsets are shown separately so an axis permutation is not mistaken for a physical memory move.

The input and output windows each show at most 4 × 8 cells from the last two axes, with leading coordinates fixed and labeled. Select a window cell or use **Go to coordinate** to reach another element, including in large paged or shape-only runs. Selecting a different element resets playback. **Tensor details** returns to the full explorers and their 3D views with the selection preserved. Reduced-motion preferences replace automatic travel with manual steps. One-to-many unfold mappings, dtype reinterpretations, empty tensors, and unverified mappings retain the existing inspection view.

## Follow an in-place write

Run a module that creates a view, then modifies the original tensor or the view:

```python
view = x.view(2, 3)
frozen = x.clone()
view[0, 1] = 99
x.add_(10)
return view, frozen, x
```

With an input shape of `[1, 2, 3]`, select either write in the journey. **Inspect affected tensor** switches between the written tensor and its recorded shared views. Compare the same coordinate before and after, open either snapshot in 3D, or follow its earlier producer. The cloned tensor remains independent. **Operation operands** shows the actual arguments and return values; indexed assignment has side effects but no returned tensor.

Dashed edges connect earlier shared views to the operation that writes their storage. Later operations use the refreshed snapshots. These connections track shared storage, not a per-cell write mask: disjoint slices or cells assigned the same value may be unchanged. Layout-only operations such as `transpose_` retain a separate explanation and do not refresh unchanged views.

Every observed alias update counts toward the snapshot budget. Large tensors remain paged; Shapes-only runs show no fabricated values. Native calls that bypass tracing and writes through NumPy/raw storage are not fully tracked. An observable change outside a recorded operation adds a **Run details → tracking notice**, without guessing its producer. Existing saved runs remain unchanged; rerun for mutation dependencies.

## Reuse an input configuration

Open **Inputs → Saved inputs & NumPy files** in any project. **Save current input** stores a named copy of its shape, axis names, generator, dtype, seed, random-stream setting, and recording mode. **Use this input** copies those settings into the current project and rechecks its component sequence. Later edits affect only that project. Each saved entry and each recorded run keeps its original settings.

The library starts collapsed and supports numeric and large shape-only configurations. Saving generated settings does not allocate values. In a project with multiple forward inputs, each input can use a separate library entry. Model weights have their own library.

New selections of **Seeded random** use an input generator independent of model initialization. The same shape, dtype, and seed produce the same random input across models in the current PyTorch environment. Existing random projects retain **Shared with model · legacy**, so their earlier execution behavior remains available. You can explicitly switch to **Independent of model** in input settings. The seed still initializes a fresh model for each run; the input library does not store weights, and exact reproducibility across different PyTorch releases is not guaranteed.

## Import a NumPy tensor

Open **Input tensor → Saved inputs → Import NumPy file**. Choose a `.npy` file, name it, and select **Import input**. Review its detected shape and dtype, then choose **Use this input** to copy it into your project. The same library is available in the builder and custom-code editor. Importing alone does not replace the active input or execute model code.

The first version accepts one non-empty rank 1–6 array with `float32`, `float64`, or `int64` values, up to 8,388,608 elements (64 MiB of values plus a bounded header). NumPy format versions 1 and 2 are supported. `.npz` archives, object/pickled arrays, structured arrays, and other dtypes are rejected without conversion. Fortran storage and non-native byte order are normalized into a contiguous CPU tensor while preserving logical values and axis order. Axis meanings are left unnamed rather than guessed; custom-code input settings can add axis names.

```python
import numpy as np

x = np.arange(64, dtype=np.float32).reshape(2, 4, 8) / 8
np.save("token-samples.npy", x, allow_pickle=False)
```

An uploaded input's shape and dtype stay fixed in settings; reshape, permute, or convert within your model to see those transformations in the trace. **Use generated values** switches back to editable synthetic inputs. The model seed remains configurable. Each value run checks the stored file's checksum and loads a private copy, so in-place model operations cannot alter the library. A missing or changed file produces an explicit error. Earlier recorded tensor snapshots remain independent of the input file.

**Shapes only** uses the uploaded tensor's metadata without loading numeric values into the model. Numeric runs use the same bounded value windows and 3D explorers as generated inputs. NaN, infinity, and integers beyond JavaScript's exact numeric range retain their explicit string readouts rather than being rounded into misleading values.

## Saved model weights

Open **Inputs → Model weights** in any project. Choose **Import checkpoint**, select a `.pt` or `.pth` file, name it, and save it to the local library. Importing does not change the active project. **Use checkpoint** pins that exact saved version to the project; **Use initialized weights** returns to the module's seeded initialization.

Export a standard tensor state dictionary from PyTorch:

```python
torch.save(model.state_dict(), "weights.pt")
```

Plain state dictionaries and the common `state_dict` / `model_state_dict` wrappers are accepted. Import uses `torch.load(weights_only=True)` with CPU mapping in a separate process, and never falls back to unrestricted pickle. Full model objects, TorchScript, legacy serialization, sparse/quantized/complex tensors, custom objects, and non-tensor extra state are outside this first version. Limits are 64 MiB per file, 63 MiB of logical tensor values, 2,048 tensors, eight dimensions, and 8,388,608 elements per tensor.

**Check compatibility** explicitly constructs the selected module on PyTorch's meta device and verifies exact parameter/buffer names, shapes, and dtypes, then loads the state without running `forward`. A value-dependent constructor may not support this metadata check. Run always checks again against the actual constructed model. Missing/unexpected keys and mismatches stop execution; no prefixes are stripped and no partial state is silently accepted. Floating weights must match the first input's floating dtype, which also sets the model dtype. Integer buffers retain their declared types. Builder checkpoints must use the parameter names in its generated `ComposedModel`.

Each run records the checkpoint name, immutable ID, SHA-256, and tensor manifest. **Run details** shows the reference, while operation inspectors show the recorded parameter values actually used. Checksums are verified before user code runs, including Shapes-only runs. In-place changes affect that run's model, not the saved checkpoint. Previous runs retain their reference even after a project chooses different weights. Back up `.data/weights/` with the database, inputs, and snapshots. This does not pin the PyTorch environment or guarantee reproducibility across versions.

## Multiple forward inputs

In a source-code project, open **Inputs**. Expand an input card to edit its name, shape, values, dtype, seed, and axis labels; **Add tensor input** creates another independent tensor. Every card can reuse a saved input or import a NumPy file. The call preview shows the exact argument order and keyword names, for example `model(query, key, value, mask=mask)`.

For `forward(query, key, value, *, mask)`, configure query/key/value as positional inputs and mask as a keyword input. Names must be unique Python identifiers; keyword names must match the function. Positional inputs must precede keyword inputs. Run validates the Python signature before generating the input tensors, then records each tensor as a separate named graph root. Unused inputs remain visible, and each saved run retains every input configuration.

The first input keeps the existing model initialization behavior: its seed initializes the model and its floating dtype sets model parameter dtype. Other inputs keep their own dtypes. Independent random inputs use their own seeds, so changing one does not consume another's random stream. The recording mode applies to the whole call. There are at most eight tensor inputs, 8,388,608 elements per numeric input, and 32 million total input elements within the existing snapshot budget. Shapes-only runs retain the per-tensor logical limit. Non-tensor forward options and nested argument containers are not yet configurable.

The visual builder has one model input and supports branches through explicit earlier-output connections and two-input joins. Choose **Use as custom code** to configure a model with multiple external inputs.

## Large tensors

Set the builder input to `[1024, 32, 32]` to explore a real million-element input, then add compatible components. The browser requests only the visible cells from immutable snapshots, including after permutation and in-place mutation. Each request contains at most 256 logical indices. Slice controls handle large batch counts without creating huge menus.

**Inputs → Shapes only** uses PyTorch meta tensors to inspect shape, strides, and layout without allocating numeric values. Shapes such as `[1024, 1024, 1024]` are supported. This mode is explicitly labeled and never supplies invented numbers. Data-dependent branches or operations without meta support may stop the trace; use a smaller value run for those.

Value runs allow up to 8,388,608 elements per tensor and 32 million elements across captured states. Shape runs allow up to 2^40 logical elements per tensor. Both retain the existing 256-operation and 20-second limits. These are bounded local execution modes, not arbitrary model-size support.

## Supported scope

Source projects run one selected `torch.nn.Module` from a bounded Python source tree, with JSON constructor arguments and one to eight generated or uploaded tensor inputs with positional or keyword binding. You can paste code or upload a `.py` file. Evaluation mode and `torch.no_grad()` are used. Every run starts a fresh model with the supplied seed and then loads the selected checkpoint, if present.

- CPU, dense real tensors; input generators: sequential, random normal, zeros, and ones, plus `.npy` tensor imports.
- Rich lessons: reshape/view/flatten, permute/transpose, contiguous/clone, squeeze/unsqueeze, cat/stack/split/chunk/unbind, Tensor.unfold, matrix multiplication, linear projection, 1D/2D convolution neighborhoods, 2D max/average/adaptive-average pooling, layer normalization, ReLU/GELU/sigmoid/tanh activations, patch embedding, softmax, basic arithmetic, and reductions. Interactive element mappings are available for the supported layout operations; dot-product interaction supports operands of rank two and above.
- Intercepted in-place writes include recorded shared-storage effects. Other intercepted tensor-returning calls get a generic inspection view. Not every Python statement is a tensor operation. Fused calls remain fused; custom extensions, compilation, tensor subclasses, training, and GPU execution are outside this milestone.
- A trace records only the path taken by that input. It is not a symbolic graph of all possible branches.
- Limits: 8,388,608 elements per tensor in value mode; 32 million elements across snapshots; 256 recorded operations; a 20-second worker deadline. These are trace/execution budgets, **not a peak-memory guarantee**.

**Run trusted code only.** Code executes with your local user's access in a separate process and temporary working directory. A subprocess is not a security sandbox. Keep both services bound to loopback. Untrusted uploads on a shared server need a separately designed sandbox before deployment.

## Checks

```sh
cd backend
uv run pytest -q
uv run ruff check tensorviewer tests
uv run ruff format --check tensorviewer tests
```

```sh
cd frontend
npm test
npm run build
npm run format:check
```

Backend tests compare the recorded Attention output with uninstrumented PyTorch, verify exact index mappings and storage semantics, preserve snapshots through mutation, retain failed steps, and exercise persistence and worker timeouts. Frontend unit tests verify coordinate conversion, broadcasted dot-product contributors, normalization groups, and graph connections for branches, joins, repeated operands, multiple outputs, and failed operations.

See [the architecture guide](docs/architecture.md) for the source, composition, tracing, and presentation contracts.

## Shifted windows and relative positions

Add **Spatial → Regular + shifted windows** for two independent blocks: a regular-window block followed by a cyclically shifted block. **Shifted-window transformer** exposes an individual block. The input is NCHW and must divide into square windows; choose a shift from zero through window size minus one. A spatial axis containing only one window has its shift disabled.

The actual trace includes negative roll → window partition → Q/K/V → scaled scores → learned relative-position bias → boundary mask → softmax → weighted values → residual/MLP → restoration → positive roll. A mask prevents artificial wraparound neighbors from attending, while genuine neighbors across the original window boundary can communicate. Blocked logits use negative infinity and receive exactly zero softmax weight.

Select the **Window bias and mask** stage, its masked-fill operation, or its softmax. The lesson links a query/key pair across the learned bias, recorded Boolean mask, and recorded attention weights. It identifies batch, window, head, local coordinates, query-minus-key offset, and the captured lookup index. Values come from the trace; Shapes-only mode retains structural navigation without numerical claims. Cyclic roll also supports exact forward/inverse cell mapping for large tensors without allocating a full index map.

These are educational fixed-resolution blocks with initialized parameters, not a complete Swin model. There is no automatic padding, pretrained checkpoint conversion, or dropout. They compose with the existing embedding, merging, and classifier tools.

## Author branches and joins

Each component's **Connections** disclosure selects a prior output or the model input. Ordinary sequences show a compact source summary; joins and branches initially show their source selectors. Connect two projections to the original input, then add **Layers → Add branches**, **Concatenate branches**, or **Stack branches** and select both outputs. Addition requires identical shapes; concatenation requires equal dimensions except along its configured axis. Stacking requires identical shapes and inserts a new axis of size two at any position. The builder displays both incoming shapes, explicit connection paths, and source buttons that navigate to the producer.

Components stay in execution order, and the final component is the model output. Forward references and cycles are rejected. Reordering/removing a referenced node leaves an explicit connection error for repair; it does not silently substitute another tensor. Unconfigured sequence nodes continue to take the immediately preceding output, and a newly added join defaults to the previous output plus the original input. The recorded journey uses real PyTorch dependencies.

## Import and run a source project

1. Choose **New project → More sources → Git repository**. Enter a public HTTPS Git URL or an absolute path to a local repository, a revision, and optionally a subdirectory. Git must be installed locally.
2. **Read source files** fetches a committed snapshot and resolves the revision to a commit SHA. It ignores uncommitted local edits, symlinks, submodules, binaries, and unsupported file types. This action does not import Python, install dependencies, execute hooks, or run setup scripts.
3. Select the entry `.py` file, the `nn.Module` class, and the import root (`.` or commonly `src`). **Set up inputs** creates the project and opens Inputs. Configure tensor inputs there and constructor arguments under **Code → Model settings** before generating the diagram.
4. The source-file selector edits the entry file or any helper/configuration file. **Add source file** creates a relative path; additional files can be removed. Changing **Entry file** preserves the whole tree. Package-relative imports and `src` layouts are supported. The worker's working directory is the source root.
5. **Python environment** defaults to the backend's packages. To add dependencies, enter explicit `name==version` requirements, one per line, and choose **Create and select environment**. Setup accepts published wheels only, with no repository setup, editable installs, source builds, or automatic requirements-file installation. Run is separate from setup.
6. Run saves the exact source tree and configuration. Recorded operations and errors identify helper filenames and lines. Code inspection opens the appropriate captured file. Run details and history retain the imported revision, entry point, environment reference, and actual Python/package versions observed by that worker.

Source snapshots contain up to 128 Python/text/configuration files, 500 KB per file, and 2 MB total. Import a focused subdirectory for larger repositories. HTTP authentication, Git LFS assets, checkpoint downloads, repository installation, and arbitrary external data files are outside this importer. Local repositories provide a path for already authenticated clones. Missing dependencies or unsupported execution paths produce normal trace errors.

Dependency environments inherit the backend's base packages and add their own pinned packages; they are not hermetic lockfiles or a security sandbox. Base-package upgrades can affect them, so each run records versions again. Setup is serialized with execution and has bounded subprocess deadlines. Environments live under `.data/environments/`; recreate them after moving installations. Source files and Git provenance are stored in project/run JSON in SQLite. Imported source edits are independent of the original commit and are preserved in each run.
