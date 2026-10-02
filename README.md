# TensorViewer

A local workbench for tensors: write a few lines of PyTorch, run them, and step through what each operation did to every cell. Start in the console, build a model from the component toolbox, paste or upload an `nn.Module`, or import a committed Git source tree, then inspect the actual execution.

Two independent projects live here:

- **`backend/`** — Python, FastAPI, PyTorch, SQLite, and an execution worker.
- **`frontend/`** — React, TypeScript, Vite, Tailwind, and reusable SVG tensor renderers.

## Contents

- **Getting started**: [Start locally](#start-locally) · [The project library](#the-project-library) · [Declare inputs in the code](#declare-inputs-in-the-code) · [The workbench](#the-workbench) · [The console](#the-console) · [The editor shows the run](#the-editor-shows-the-run) · [Value distributions](#value-distributions) · [Breakpoints](#breakpoints) · [Axis ink](#axis-ink) · [Check shapes before running](#check-shapes-before-running) · [Shape contracts](#shape-contracts) · [Tensor insights](#tensor-insights) · [When an operation fails](#when-an-operation-fails)
- **Seeing operations**: [Loops play once](#loops-play-once) · [Light follows execution](#light-follows-execution) · [Watch a tensor move](#watch-a-tensor-move) · [Follow selection, broadcasting, and reductions](#follow-selection-broadcasting-and-reductions) · [Follow scores through softmax](#follow-scores-through-softmax) · [Follow mean and sum reductions](#follow-mean-and-sum-reductions) · [Follow activation functions](#follow-activation-functions) · [Follow an addition across tensors](#follow-an-addition-across-tensors) · [Follow a tensor through joins and splits](#follow-a-tensor-through-joins-and-splits) · [Follow a linear projection](#follow-a-linear-projection) · [Follow a convolution neighborhood](#follow-a-convolution-neighborhood) · [Explore pooling windows](#explore-pooling-windows) · [Follow layer normalization](#follow-layer-normalization) · [Follow a patch into a token](#follow-a-patch-into-a-token) · [Replay an element's layout change](#replay-an-elements-layout-change) · [Follow an in-place write](#follow-an-in-place-write) · [Indexed 3D tensors](#indexed-3d-tensors) · [Explore recorded stages](#explore-recorded-stages) · [Large tensors](#large-tensors)
- **Inputs and weights**: [Inputs that mean something](#inputs-that-mean-something) · [Reuse an input configuration](#reuse-an-input-configuration) · [Import a NumPy tensor](#import-a-numpy-tensor) · [Multiple forward inputs](#multiple-forward-inputs) · [Saved model weights](#saved-model-weights)
- **Building models**: [First walkthrough](#first-walkthrough) · [Follow a complete Vision Transformer](#follow-a-complete-vision-transformer) · [Follow a spatial hierarchy](#follow-a-spatial-hierarchy) · [Shifted windows and relative positions](#shifted-windows-and-relative-positions) · [Author branches and joins](#author-branches-and-joins) · [Reusable custom components](#reusable-custom-components) · [Import and run a source project](#import-and-run-a-source-project)
- **Reference**: [Supported scope](#supported-scope) · [Checks](#checks)

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

On an empty workspace, the application installs the [project library](#the-project-library) and opens its first project. **New project → Library** adds any library project again; **New project → Code** accepts pasted code or an uploaded `.py` file. Run tensor statements using `x`, or choose **Model class** to keep an entire `nn.Module` source file intact; its [declarations](#declare-inputs-in-the-code) set the class and inputs, so it is ready to run. **Build a diagram** opens the model builder; **Git repository** imports a source snapshot for configuration. All three paths use the same recorded tensor journey. Model code executes only when Run, Check shapes, Check custom shapes, or Check compatibility is selected, or while live shape checking is switched on. Existing custom-code projects and saved executions remain available.

## The project library

Twenty-four projects, each a self-contained PyTorch file in `backend/tensorviewer/library/`, run from the basics to complete transformers:

| Track | Projects |
| --- | --- |
| Foundations | 01 Tensor shapes · 02 Broadcasting · 03 Reductions and softmax · 04 Multilayer perceptron · 05 Convolutional image classifier · 06 Word embeddings · 07 Recurrent language model |
| Attention | 08 Scaled dot-product attention · 09 Multi-head attention · 10 Causal self-attention · 11 Positional encodings |
| Text transformers | 12 Transformer encoder (BERT-style) · 13 GPT language model · 14 LLaMA-style decoder (RMSNorm, RoPE, SwiGLU) · 15 Encoder–decoder translator · 21 Mixture of experts (top-1 routing) |
| Vision transformers | 16 Patch embedding · 17 Vision Transformer · 18 Swin Transformer · 19 Masked autoencoder · 22 Object detection transformer (DETR) · 23 Perceiver (latent cross-attention) |
| Multimodal | 20 CLIP dual encoder · 24 Image captioner (ViT encoder, cross-attending text decoder) |

Projects 21–24 are the advanced end. The mixture of experts routes each token to one feed-forward expert, and its expert loop folds. DETR's learned object queries each predict a class and a box. The Perceiver's 8 latents read 256 pixels once, then attend among themselves. The captioner's decoder cross-attends from words to image patches.

An empty workspace starts with all of them. The explorer lists your own projects first, most recent at the top, then the open project's files and steps, then the library in numbered order under its tracks; a renamed library project counts as your own. Its step list follows playback ("Steps · 36 of 61"). A folded loop's later passes sit behind one `↻` row that opens its repeats, steps not yet run are unlit, and the current step's row stays in view. **New project → Library** marks the entries already in your workspace and opens those instead of making a duplicate (type a different name to make a copy). It creates any missing one, and the command palette's **Restore library projects** adds back the ones that are missing. Each project is ordinary code: its docstring names it and explains it, and `# axes:` comments name the result of the line they follow. Steps inside that line take names only where they follow: in `q = q.reshape(b, t, h, d).transpose(1, 2)  # axes: batch, heads, tokens, head_features` the reshape reads batch, tokens, heads, head_features, and a step that changes the axes otherwise keeps plain `axis N` names.

## Declare inputs in the code

A pasted or uploaded module, a library file, and a file read from disk all go through the same reader (`POST /api/v1/sources/read`). Pasting a library file's code gives the same project the library creates. The reader looks at comments and the docstring only; it never runs the code.

```python
"""Pair attention.

The rest of the docstring becomes the project summary.
"""
# input query: batch=2, tokens=8, features=16
# input image: batch=1, channels=3, height=16, width=16 | image
# input words: text "the cat sat on the mat"
# input mask (keyword): batch=2, tokens=8 | ones | float32 | seed=3
# model: Pair
# constructor: {"heads": 2}
# capture: shapes
```

- **`# input NAME: axis=size, …`** declares the forward parameter `NAME`. After the shape, `|` separates a generator (`random`, `arange`, `ones`, `zeros`, `image`), a dtype, and `seed=N`. Floating inputs default to `random`, and `int64` inputs to `arange`. `text "…"` makes token ids over the sentence's own vocabulary. `(keyword)` or `(positional)` sets how the value is passed. Keyword-only parameters are passed by keyword.
- **`# model:`** chooses the class. Without it, the last `nn.Module` subclass in the file runs (the **Model class** field also chooses).
- **`# constructor:`** is a JSON object of constructor arguments.
- **`# capture:`** is `values` or `shapes`.

The **Input** bar above the canvas edits the starting tensors. When the model takes several inputs, each one gets a chip with its name and shape, and the chosen one's values, shape, or sentence are edited in place.

After editing the `# input` lines, **Read from code** in Inputs & settings (Input tensors) applies them again. `# constructor:` and `# capture:` are applied only when present, so values set in the panel are not replaced by defaults. A parameter with a default is left out unless it is declared. An undeclared required parameter gets a `[2, 4, 8]` random input you can change under Inputs. A declaration the reader cannot use becomes a note shown after creation (settings open on it), not an error. Code without an `nn.Module` is refused.

## The workbench

TensorViewer opens around the tensor journey. A compact input summary sits above the canvas; code and recorded tensors open only when needed.

- **Input**: expand the summary to change values and dimensions without opening code. Setup collapses after a run; invalid shapes keep it open. **Examples** supplies a matching script and input; **＋ Operation** adds a transformation to a console experiment.
- **Journey**: the main canvas shows the recorded model. Play follows each operation with its real input tensors and result; the camera moves between these groups without opening inspection panels. Scrub the timeline or use its previous/next controls to move through recorded steps. Pan or zoom to take control of the camera; **Resume following tensors** returns to the active step.
- **Operation scenes**: the active operation displays each participating tensor separately. Splits expose every output; joins show each part, including tensors from the same producer. Weights and biases appear beside their layer. Short role labels and a recorded-operation explanation identify how the tensors participate. Repeated uses of one tensor retain separate operand connections. A dense operation uses a shared junction to keep connections manageable.
- **Trace a cell**: select a result cell to highlight the recorded input cells it reads. Its coordinate, snapshot value, and a short explanation replace the scene caption. Arrow keys move between result coordinates; Escape clears tracing. **Enlarge selected cell** opens the full 3D inspector. Large contributor groups show a bounded subset with an explicit count, and unsupported mappings are labeled rather than guessed.
- **Inspect**: select a node or use the transport's inspect action to open its transformation, follow individual cells, or enlarge its 3D view. The information button reveals the explanation and source line. **Stages** groups recorded module calls; stage boundaries and individual operations remain inspectable.
- **A step fits its frame**: an inspected step is laid out like a slide instead of a long page.
  - **Lenses**: views of the same step take turns on one stage instead of stacking. **Motion**, **Cells**, and **Follow** for layout steps; **Cells** and **Working** for dedicated lessons such as softmax, reductions, and activations. The selected cell carries across lenses.
  - **Title row**: **Follow connections**, the value toggle, the lens switch, and **Predict the shape** sit in the step's title row. Connections open as a dropdown.
  - **Side columns**: notes (the selected element's explanation, how to select, how the operation works) sit beside the stage. On wide frames, a lesson's formula, controls, and axis choice form a column beside the active lens.
  - **Fitting**: drawings stretch or shrink to the height that is left. If a lesson is still slightly too tall, it scales down to fit, never below 80%. Past that, only the part that does not fit scrolls, and the rest of the stage stays in place.
- **Playback options**: speed, camera following, gradual reveal, direct step selection, and run details share one popover. Tracking notices mark its button. Camera and connection animation respect reduced-motion preferences. The scene uses actual recorded dependencies; it does not invent cell motion for unsupported operations.
- **Code**: opens a resizable source panel on the right, with recorded shape inlays and tensor previews. On narrow screens it overlays the canvas and can be closed independently.
- **Tensors**: opens a shelf of clickable tensor cards with dimensions, data types, value ranges, and shared storage. **Run notes** and **Printed output** are available in the same shelf. Like a debugger's locals, the shelf follows the playback position. Each name holds the state it had at the current step, the step's own result is marked *just written*, and names assigned later wait unlit as *not computed yet*, with the step that will compute them. While a loop's repeats play, the shelf shows the pass passing through. With no step selected, it shows the end of the run. A name written more than once, like the residual stream `x`, carries a watch strip: one bar per recorded state, spanning its value range on a shared scale. The state at the current step burns and later states wait unlit, with `state 3 of 6` beside it. Hovering or focusing a card follows the name on the canvas: every node that wrote it is ringed, and so is its spot on the minimap. A state from a later pass of a folded loop rings the drawn pass, and one inside a collapsed stage rings the stage.
- **Projects**: access projects and source files from the collapsed left rail. **New** creates an experiment. Canvas projects keep the component builder alongside their recorded journey.
- **Run history**: reopen or compare saved runs. Each run says what it changed since the one before (`model.py · 2 lines`, `input tokens`, `values → shapes`), or **re-run** when the code and inputs were the same. A comparison opens on the steps that differ, under a line naming what the later run changed; **Show the N matching steps** adds the rest. All output tensors are checked; unavailable or paged values are explicitly marked as not compared. When nothing is wrong, **Run notes** lists what the checks looked for.
- **Run** (Ctrl/⌘ + Enter): saves and records the experiment. Repeated shortcuts while recording do not start additional runs.
- **Keyboard shortcuts** (?): every shortcut, grouped by where it works: anywhere, stepping, canvas, step lesson, and editor. Also in the command palette.
- **Project search** (Ctrl/⌘ + K): jump to steps, module calls, loops, tensors, projects, or actions. Steps are also found by their module and loop pass: `cross_attention` lists the calls `blocks.0.cross_attention` and `blocks.1.cross_attention` first (each opens at its first step), then their steps. `↻` lists the recorded loops; a folded one opens on its repeats.
- **Settings**: inputs, saved inputs, weights, environment, and module configuration. The status line shows the current recording and selected cell.

The address bar names the current project, run, step, and selected cell. **Copy a link** reopens that location for anyone using the same workspace; a link to a missing run falls back to the latest one. Multi-output selection is currently local to the view: shared links open the operation's first output.

During canvas playback, small supported transformations animate the actual indexed 3D cells between their recorded positions. Layout changes, selections, joins, splits, and reductions use verified mappings, capped at 128 paths. Input and result tensors remain visible as reference states; a reduction's paths identify contributors rather than simulate its arithmetic. A caption counts the paths currently visible across all output actors; zoomed or offscreen views may show only a subset. Large or unsupported operations retain their tensor view and inspection tools.

Pause and speed changes preserve the position within a transformation. With the canvas focused, **Space** toggles playback. Click a moving cell—or focus a path, use arrow keys to choose another, then press Enter—to trace its result coordinate and input contributors on the canvas. Reduced-motion preferences replace cell travel with static end markers.

## The editor shows the run

Every line that recorded an operation carries an inlay with the tensor it produced and its shape:

```python
grouped = x.reshape(2, 3, 2, 2)        ▢ grouped [2, 3, 2, 2]
swapped = grouped.permute(0, 2, 1, 3)  ▢ swapped [2, 2, 3, 2]
```

- Select an inlay or a highlighted line number to open that step. A line with several operations cycles through them.
- Rest the pointer on an inlay, or on a tensor's name in the code, for a preview: axis names and sizes, value range, the first window of values, or a picture when the axes are named height and width.
- When a tensor's axes were made from other axes, the preview also says where each one came from. After `heads = x.reshape(2, 3, 2, 4).permute(0, 2, 1, 3)` on an input with axes `batch, tokens, features`, it reads `x.batch`, `x.features (piece 1 of 2×4)`, `x.tokens`, `x.features (piece 2 of 2×4)`. Reshaping the pieces back together gives `x.features` again. The trace stops, and says so, at an operation with no axis-level story, such as a convolution's spatial output. The **Tensors** shelf shows the same lineage on each card, and in a step view each axis badge carries a short origin such as `← x.features[1/2]` (full text on hover).
- After an edit, inlays from the changed line onward are dimmed until the next run. A failing line is underlined.
- Type `name.` after a tensor whose shape is known to get suggestions that show the shape each would produce, for example `flatten  [2, 3, 4] → [2, 12]`. `torch.` and `F.` suggest common functions applied to the most recent variable. Arrow keys choose, Enter or Tab accepts, Escape dismisses. Suggestions come from a fixed list of common operations and simple shape rules; the recorded run remains the authority.
- The breadcrumb under the tabs names the project, the file, and the tensor produced on the caret's line.

## Value distributions

Every recorded tensor with values carries a histogram of its finite values, computed from the real tensor even when it is too large to send inline. Under each tensor card's readout, in lessons, the inspector, and the enlarged 3D view, a small chart shows how its values are spread. A dashed line marks zero, the chosen or hovered cell's value is marked in fire, and the line beneath gives the range, mean, standard deviation, and the share of exact zeros. NaN and infinite values are counted in red. A ReLU output shows its dead half as a spike at zero (54% zeros after the convolution in project 05), and saturation or outliers show at a glance. Large paged tensors now also report their range in the Tensors shelf. When two runs are compared, a large tensor whose values were not sent is still reported as changed if its distribution differs, with its mean shift ("mean 0.120 → 0.415"). Equal distributions prove nothing, so such a tensor stays "not compared". Runs recorded before this have no histograms; run again to get them.

## Breakpoints

Click the margin left of a line number, or press **F9** on a line in the editor, to set a breakpoint. Play then runs the journey until it reaches a step recorded on a breakpoint line and pauses there: the step's scene is on the canvas, its line is highlighted, and a red dot marks the transport. Play continues to the next breakpoint. When a breakpoint line runs in a later pass of a folded loop, playback pauses on the loop's repeats. Breakpoints are kept per project and file in this browser; **Clear breakpoints** in the command palette removes them.

**Pause at steps with warnings**, in the playback options, works like exception breakpoints. Playback also stops at any step a run note points at, such as a NaN source, a scrambled reshape, a loop drift, or a failed step. An amber dot marks the pause, and the choice is remembered.

The transport names the current step and, when the result has a name of its own, the result too (`rsqrt → scale`). The transport steps like a debugger. **Next** (F11) steps into the next operation. **Step over** (F10) moves to the next step at the same depth, past the module calls the current step makes: over an attention call, over a whole block, or over a loop. **Step out** (Shift+F11) moves to the first step after the call around the current one, and Shift+F10 steps back. Depth counts the recorded calls of two or more operations around a step (the canvas's stages), so a single layer such as a `Linear` is not a level of its own. The keys work anywhere in the workspace, the editor included. **Run to cursor** (Ctrl/Cmd+F10 in the editor) jumps to the caret line's next execution after the current step, shown at that execution's result. Selecting a line's inlay or number does the same, and selecting it again cycles through the line's steps.

Once playback starts, the code lights up with the canvas. Inlays of lines whose steps have not run yet are unlit, the current line is highlighted, and lines already run keep their ink. The minimap shows the same light: unlit, lit, and the active step in fire. A fresh run keeps every inlay readable until you start stepping.

## Axis ink

Every axis of a model input has its own color, and the color stays with that axis wherever it goes. In `x [2 × 4 × 8]` with axes `batch, tokens, features`, the input summary is the legend. The same colors mark the sizes in code inlays, canvas nodes, the Tensors shelf, the hover preview, the axis badges and motion titles of a step view:

```python
heads  = x.reshape(2, 4, 2, 4).transpose(1, 2)  ▢ heads  [2, 2, 4, 4]   batch · features piece · tokens · features piece
scores = heads @ heads.transpose(-2, -1)        ▢ scores [2, 2, 4, 4]   batch · features piece · tokens · tokens
out    = mixed.transpose(1, 2).reshape(2, 4, 8) ▢ out    [2, 4, 8]      batch · tokens · features
```

- A piece of a split axis keeps its parent's color with a broken underline, so both halves of `features` stay recognizable after the heads move.
- A merged axis carries the colors of everything it was made from.
- Axes that come from weights, or from tensors created in the code, share one neutral color.
- An axis whose origin was not traced, such as a convolution's spatial output, is not colored. Its size is shown plainly rather than guessed.

Animated tensors use the same ink. In a step's whole-tensor motion, **Color by** shades every cell by its position along one input axis, dark at index 0 and light at the last index, in that axis's color. Choose `tokens` and watch the amber stripes move during a transpose, or `features` to see a split axis come apart. The default is the axis the step moves (a transpose) or changes (a split, merge, or reduction); **order** restores the plain memory-order gradient. Cells flying across the canvas during playback take the same shades, and the 3D viewer colors each axis's index ticks and legend in its ink.

The cubes of 3D tensors and the squares of 2D grids are tinted glass: each cell takes the shade of the input column its value came from (the first input's last axis), so one value keeps its color on every node it reaches, including after a transpose, a split into heads, or a reshape. A tensor that no longer has that axis is shaded along its own traced axis, and one without a traced axis stays clear. Shapes are inked wherever a recorded tensor is named: code inlays and breadcrumbs, canvas nodes, the Explorer's step list, connections, stage outputs, the 3D viewer's title, lesson stages, and Run compare (both runs). An in-place write keeps its tensor's axes, so a written state stays inked like the state it overwrote. The same glass and ember carry into the Follow lens (its traveling marker burns too), a reduction's contributors, patch-embedding grids, and the editor's hover preview. The selected cell burns like an ember, in 3D cubes, 2D grids, and the motion view alike; reduced-motion preferences keep it still.

Hover a colored size for its full origin. Color is never the only signal: titles, the hover preview, and the badges' `←` origins say the same thing in words.

## The console

The console is the fastest way to see an operation. Type statements that use the input tensor `x`, then run:

```python
grouped = x.reshape(2, 3, 2, 2)
swapped = grouped.permute(0, 2, 1, 3)
flat = swapped.flatten(1)
```

`torch`, `nn`, `F` (`torch.nn.functional`), and `math` are already imported. The final assignment or expression is the output; write `return a, b` to return several tensors. Only the last operation of a statement takes the assigned name; intermediate results are named after their operation. An unassigned subscript or prefix operator is named as written, such as `x[:, 0]` or `~mask`, and steps show Python operator methods by what they do (`__getitem__` reads as **index**). Tensors made from arguments alone (`torch.arange(1, 9, 2)`, `torch.ones(4, 4)`, `zeros_like(x)`) get a lesson that states what was made, dtype conversions and `~` get the elementwise lesson (`~True = False`), and `F.normalize` explains dividing each vector by its length.

Expand the **Input** summary to see the input name, value source, shape, and dtype. Change the value source or the shape there; the settings button opens every input option. **Examples** loads a script together with a matching input, and **＋ Operation** adds a common operation applied to the most recent variable, before an explicit return when one exists.

A console project stores your statements and wraps them in a generated module for execution, so tracing, limits, and the trusted-code model are the same as for any other project. **Settings → Use as custom code** converts it to an ordinary module project; this is one-way. Layers created inside a script, such as `nn.Linear(4, 5)`, are constructed without recording their initialization.

Module and repository projects edit their source files in the same editor, one tab per open file. Add, remove, or replace files from **Projects → Files**. Canvas projects show their generated code read-only; use **Code** to reveal it.

## Check shapes before running

**check shapes** in the status bar (Ctrl/⌘ + Shift + Enter, or "Check shapes" in the command center) dry-runs the current console or module code on PyTorch metadata tensors. It computes no values, saves no run, and takes about a second. The result appears in the code as dashed `≈` inlays on every line the displayed run no longer covers, so after an edit you see the shapes the new code would produce:

```python
keys = x.mT                                    ≈ keys [2, 4, 3]
scores = torch.einsum("bij,bjk->bik", x, keys) ≈ scores [2, 3, 3]
```

A shape mismatch found this way is listed under **Run notes** as a warning with the same explanation a failed run gets, and its line is underlined. Predicted inlays have no step to open and no values; press Run for those.

A shape check has no values, so it stops at the first step whose result shape depends on them, such as a Boolean-mask index or a data-dependent branch. That line is marked "needs values: run to continue" and the status reads "shapes ✓ until values are needed". This is a limit of the check, not a problem in the code.

**live** beside it re-checks shortly after each edit. It is off by default because it executes your code as you type; turn it on only for code you are content to have run half-written. The choice is remembered in this browser. Canvas projects check shapes in the builder instead.

## Shape contracts

State the shape a line must produce with a `# shape:` comment, and the editor checks it against the recorded run, or against a shape check of code you have not run yet:

```python
q = x.reshape(2, 3, 4)   # shape: B, T, D
k = q.transpose(1, 2)    # shape: B, D, T
scores = q @ k           # shape: B, T, T
flat = scores.flatten(1) # shape: B, D      ✗ D was 4 on line 1, but axis 1 of flat is 9
```

- A number is an exact size. A name such as `B` or `tokens` binds to the size it first meets and must match on every later line. `_` accepts any one axis, and `...` accepts any number of axes, so `# shape: ..., D` checks only the last one.
- A met contract adds ✓ to the line's inlay. A broken one adds ✗, underlines the line, marks the margin, and is listed in **Run notes** with what was expected and what was found.
- Contracts are checked in line order within the open file. A line whose tensor is out of date or failed is skipped rather than counted as broken.
- Like `# axes:`, a contract is a comment: Python ignores it, and nothing is enforced at run time.

## Tensor insights

After every run, and after every shape check of edited code, TensorViewer reads the recorded tensors for patterns that are legal but often unintended. Each insight points at its step and line in **Run notes**; warnings also underline the line and mark the margin.

| Insight | What it notices |
| --- | --- |
| Both operands were stretched | Elementwise operands that each broadcast along an axis the other has, such as `[3, 1] − [3]` giving `[3, 3]`: an outer combination where one result per position was probably meant. Size-1 axes written out on both sides (`rows[:, None] * columns[None, :]`) are a deliberate table and are not flagged. |
| *x* grows (or shrinks) every pass of a loop | The value a loop carries into its next pass, its last result, changes its largest magnitude at least fourfold from the first pass to the last, steadily, such as a residual stream that keeps growing through the blocks. Located on the loop header; opening it shows the last pass. Needs values. |
| NaN or infinity first appears here | The first step whose result is not finite although its inputs were, with the usual cause for division, log, square root, and power. An infinity the step was asked to write, such as `masked_fill(mask, -inf)` before a softmax, is not flagged unless a NaN appears with it. Needs inline values. |
| reshape mixes axes | Using axis lineage, a `reshape`, `view`, or `flatten` that interleaves elements of different source axes, for example reshaping attention heads `[B, H, T, d]` straight back to `[B, T, H·d]` without permuting them first, or merging the pieces of a split axis out of order. Merging whole axes in order (`batch × tokens`) and batching heads (`batch × heads`) are not flagged. |
| matmul sums unrelated axes | Using axis lineage, a matrix product whose summed axes trace back to different input axes, for example `q @ k` without `.transpose(-2, -1)` when tokens and head size happen to agree. The note names both axes and suggests the transpose when it would pair them. Products against weights or tensors created in the code (`x @ W`) are not judged. |
| softmax across the batch | A softmax whose axis is an input's `batch` axis, found through lineage even after a transpose, so each example's weights depend on the rest of the batch. |
| reduction combines examples | `mean`, `sum`, and similar over the `batch` axis while other axes are kept: a hint, since batch statistics are sometimes the goal. A whole-batch scalar such as a mean loss is not flagged. |
| softmax over a size-1 axis | Every weight is 1 (or 0 for `log_softmax`) whatever the scores are. |
| reduction over a size-1 axis | `sum`, `mean`, and similar over axes that have one position change nothing. |
| squeeze() removed the batch axis | `squeeze()` without `dim` also drops a batch axis of size one. |
| An activation produced only zeros | `relu` and similar left nothing alive, or zeroed at least 90% of values. |
| Promoted to float64 | One float64 operand widened the result. |
| This write changes the input | An in-place write reached the storage of a model input. |
| Computed but never used | No later tensor operation reads a result and it is not returned. Uses outside tensor operations, such as an `if` condition or `print`, are not recorded, so treat this as a hint. |

Insights describe the run that happened; none of them is an error, and none changes the code.

## When an operation fails

A failed operation keeps its operands. The error card on the canvas leads with the same diagnosis, for example "The feature counts do not match: … It expects 32 input features, but the input's last axis has 24.", with PyTorch's own message beneath it. **Show the failing step** opens that step, and the canvas keeps the diagram clear of the card. The **Run notes** shelf and the step view show the operand shapes side by side with the conflicting axes marked, explain the rule that was broken, and suggest a fix when one is certain, for example:

```python
scores = view @ keys   # [6, 4] @ [6, 4]
```

reports that the left tensor's last size (4) must equal the right tensor's second-to-last size (6), and that `keys.transpose(-2, -1)` has shape `[4, 6]`. Diagnoses cover:

- **Shapes**: matrix products, linear layers, broadcasting, reshape/view element counts, cat/stack, convolution channels, permute, `expand` on an axis that is not size one (with the `repeat` that would work), and `unflatten` sizes.
- **Memory order**: `view` after a permute or transpose, with the `reshape` or `contiguous().view(…)` that fixes it.
- **Data types**: float tensors used as indices, class targets, or conditions; operands of different types; integer tensors given to `mean` and similar. Each names the operand to convert and the conversion.
- **Positions**: an index past the end of an axis, an embedding index past the table, a class number past the number of classes, and axes that do not exist.

Other failures show PyTorch's message with the operand shapes.

The steps before the failure stay on the canvas. Beside the error, **Fix code** opens the code at the line that failed (**Edit model** for a canvas project), and **Edit inputs** opens the starting tensors. The line is selected only while that file is unchanged since the run, so the editor never points at an unrelated line after edits.

## Loops play once

When a `for` or `while` loop repeats the same work (`for block in self.blocks: x = block(x)`), the journey draws its body once, inside a dashed loop frame. The frame's header names the loop's code and line, and a return arc over the frame shows the body's end feeding its start again. Two iterations match when they record the same operations, from the same lines, with the same result shapes. A loop whose passes differ, such as a Swin stage alternating regular and shifted windows, stays unfolded.

- **Playback** plays the first iteration step by step, then a single loop step. During it the arc flows, the iteration pips advance, and each iteration's values pass through the same nodes, which ignite once per pass. After the loop the body shows the last iteration, the one whose result flows onward. Stepping onto the loop step with Next plays the repeats too. **Inspect** on the loop step opens the result of the pass showing, and the loop keeps showing that pass.
- **The pips** in the frame's header choose which iteration the body shows; with the canvas focused, **[** and **]** step through the passes of the loop around the selection. Selecting or inspecting a body node then opens that iteration's step, and the step's lesson says which pass it is (`↻ 2 of 4`). In a folded loop that label has ‹ › buttons, and **[** and **]** work in the lesson too. They flip the lesson to the same step in another pass without closing it, keeping the chosen cell, so one score or activation can be compared pass by pass. A collapsed stage inside a folded loop, such as one decoder block, flips between its calls the same way (`blocks.0` ⇄ `blocks.1`). Selecting a later iteration's line in the editor opens the body with that iteration shown.
- **Loops whose passes differ** are shown in full, with a light outline and a number on each pass (`↻ pass 2 of 4`), so the repetition still reads.
- **The status bar** says how many loops are drawn once (`↻ 1 loop drawn once`), which is why playback has fewer steps than were recorded.
- **Stages** fold the same way: a loop over blocks shows one block stage in the frame. A collapsed stage that hides a loop carries a `↻×2` badge.
- **The pass strip** beside the pips charts the value each pass hands to the next (the range of the body's last result) on one scale, so values that grow, shrink, or drift across passes show up at a glance. Select a bar to show that pass.
- **In the editor**, each loop header that ran carries a chip: `↻ ×2 drawn once` for a folded loop, or `↻ ×4 passes differ` for one shown in full. Selecting the chip plays a folded loop's repeats, or opens the first step of a loop shown in full. While the repeats play, the header line is highlighted and the chip burns. During playback the editor scrolls the line in focus into view, unless you are typing in it.

The recorder learns iterations from the loop headers themselves, not from repeated shapes. Each operation records its enclosing loops and the iteration of each, including loops in the code that called the module. Loops in console statements are recorded too.

## Light follows execution

A tensor glows only once it holds values. When a run opens, only its inputs glow; every computed tensor is clear, uninked glass, like a tensor that has no real values yet. Its name and shape stay readable, and a note above the transport says what lights it up until the first step. Playing or stepping activates them in order. The active step's result ignites in fire as its values arrive, and from then on it keeps a steady glow in its axis ink. Activation is permanent for the run: stepping back, replaying, or jumping elsewhere never turns a tensor dark again, and only tensors that were still unlit play the arrival. In lessons, only the result's chosen cell burns. The cells it reads from in the inputs are ringed and lit, not on fire. A tensor you inspect on its own burns at the cell you choose.

## Watch a tensor move

Select a reshape, view, flatten, permute, transpose, squeeze/unsqueeze, contiguous, clone, or roll step, or one of their aliases (`swapaxes`, `movedim`, `.T`, `.mT`, `ravel`, `unflatten`, `view_as`, `reshape_as`). Every cell travels from its input position to its output position; colors follow input order, so a reshape keeps the color sequence while a permutation visibly reorders it. Play, pause, scrub, or jump to either end, and select a cell to follow it in the tensor explorers below.

The same view animates selection and reduction: indexing, masks, sorting, and flips leave the unselected cells behind or carry each value to its new place, `expand`/`repeat` and lookups copy cells from the input, and `sum`/`mean`/`max` and other reductions bring each group together into one output cell. Motion is drawn only when both tensors have at most 256 elements and at most 40 cells along a side, and only from a verified cell mapping. Larger tensors keep **Follow element** and the paged explorers. Reduced-motion preferences show the result and keep the slider.

## Follow selection, broadcasting, and reductions

These operations link output cells to their sources in the step view:

- **Indexing and slicing** with integers, slices, `None`, and `...`, for tensors of any size.
- **Selection by value**: index tensors and lists, Boolean masks, `masked_select`, `take`, `take_along_dim`, `diagonal`, `narrow`, `select`, `flip`, `rot90`, `repeat_interleave`, `tile`, `pixel_shuffle`, nearest-neighbor `interpolate`, and reflect/replicate/circular padding. The exact source of each output cell is found by applying the same call to a tensor of positions, in value runs of up to 4,096 elements, and is kept only if it reproduces every recorded output value. A mode that blends values, such as linear resizing, therefore gets no cell mapping.
- **`sort` and `topk`**: each value's original position, taken from the positions PyTorch returns.
- **Axis aliases**: `swapaxes`, `movedim`, `.T`, `.mT`, `ravel`, `unflatten`, `view_as`, and `reshape_as` share the reshape and permute views, including motion.
- **`cumsum` and `cumprod`**: every input cell accumulated into an output.
- **`einsum`** with explicit letters (no ellipsis): the output cell fixes the letters it keeps, and each combination of the summed letters is one highlighted product, with the arithmetic when values are available.
- **Constant padding**: whether a cell is an original or part of the border.
- **`dot`, `outer`, `mv`**: shown as the einsum they are, with every contributing product.
- **`one_hot`**: the index each indicator cell is compared with.
- **`cross_entropy` and `nll_loss`** with class-index targets: the target-class score of every sample, each sample's −log p(target), and the mean or sum that gives the result. Class weights, label smoothing, probability targets, and spatial targets keep general inspection.
- **`log_softmax`**: the normalization group and the log-probability arithmetic.
- **`expand`, `expand_as`, `broadcast_to`, `repeat`**: the input cell each output reads.
- **`gather`, `index_select`, `embedding`**: the exact source cell, for results of up to 4,096 elements in value runs.
- **Reductions** (`sum`, `mean`, `prod`, `amax`/`amin`, `max`/`min`, `argmax`/`argmin`, `any`/`all`, `logsumexp`, `std`, `var`): the whole group behind an output, with the arithmetic when the group's values are available.
- **Elementwise work**: arithmetic and comparisons with a tensor or a number, common activations and math functions, `where`, `masked_fill`, `tril`, and `triu`. Every operand's matching cell is highlighted, including broadcast operands, with the calculation or the branch that was taken. When operands have different shapes, a strip above the tensors writes the shapes right-aligned and marks each axis that is reused, such as `1→4`; `expand` shows the same strip.
- **`batch_norm` in evaluation mode**: the channel's stored mean, variance, scale, and shift behind each cell, with the arithmetic. Training mode, which uses statistics of the batch, keeps general inspection.

Tensor–tensor addition keeps its dedicated lesson below.

## Inputs that mean something

**Values → Sample image** fills the input with a deterministic picture, a bright disc on a gradient in the range 0–1. The last two axes are height and width and the axis before them is the color channel; other leading axes move the disc so batch items differ. **Values → Sentence** turns a sentence into lower-cased word and symbol tokens and feeds their integer ids as an `int64` tensor of shape `[1, tokens]`; the ids index the sentence's own sorted vocabulary, which is shown beside the field.

Any tensor whose last two axes are named `height` and `width` has a **Pixels** view that draws the selected plane, in color when a three-entry `channels` axis precedes it and in gray otherwise, scaled to that plane's own range. Convolutions and pooling keep those axis names. Planes above 1,024 pixels and tensors without those names are never drawn as pictures.

## First walkthrough

1. Choose **New project** in the title bar, then **Build a diagram**; name the project and open the empty canvas.
2. Use **Set up input**, or open a category in the horizontal toolbar above the canvas. Builder and Journey are tabs above the canvas. The component panel is collapsed by default; toolbar icons open its searchable categories. Use Left/Right or Home/End to navigate the toolbar, and Escape to close its panel. Adding a component closes the panel and opens its settings. A first component supplies a compatible starter input. **Setup** configures the project name and recording mode; input settings include shape, generator, data type, and seed.
3. Select a component header or its step in the bottom navigator to change settings, reorder it, or remove it. Selection restores a readable zoom; the percentage button resets to 100%. Dimension cards pair exact sizes with axis names. The inspector provides previous/next navigation, named axis choices, and a keep-dimension switch for reductions. Connections display inferred output shapes. Invalid connections disable Run and explain the mismatch. For example, insert **Image to tokens** between convolution and attention.
4. Click **Run** to save the composition and record real PyTorch execution. The **Journey** tab displays every recorded operation, including branches inside attention. The **Builder** tab returns to your editable sequence.
5. Select a node header to compare the incoming tensor, operation, and output. Select a cell on a node, or **3D** in its focused view, to enlarge that individual tensor.
6. **Code** in the title bar displays the generated Python, read-only. **Settings → Use as custom code** detaches the composition and enables editing/uploading your own module. This is a one-way conversion; the existing execution history retains earlier blueprints.

If anything is missing or invalid, **Run** becomes the next useful action instead: **Set up input**, **Add component**, **Set project name**, or **Review inputs / code / model**. Choosing it opens the setting that needs attention, including collapsed fields. For custom canvas components, Run checks their shapes first; there is no separate required preview step.

**All tools** open a compact category browser. Pick a category or search across the entire library using names such as “CNN”, “linear layer”, or “multi-head attention”. Search accepts words in any order. Select a result to add it and open its settings; the **Custom** category holds saved modules and the option to create your own.

Opening **Inputs & settings**, the code, or the run history, or editing a canvas model, preserves your place in the recorded journey. The open node keeps its selected cell, viewing axes, slices, lesson settings, and scroll position. Selecting a different node or loading another run starts a new inspection; the app does not keep an inspection history for every node. Hidden mapping replays pause and require Play to resume, and an open 3D dialog closes when its inspection is hidden.

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

The calculation shows the whole group's maximum and exponential total, with the recorded PyTorch weight displayed separately. The table links scores to shifted values, exponential, and actual recorded weights; bars use a fixed 0–1 scale. Small groups calculate from complete captured values in the browser. Groups larger than 256 scores use a bounded backend scan of the saved input and a cached summary, so a visible window never substitutes for the full denominator. The existing numeric tensor limit still applies.

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

The lesson handles complete, non-overlapping, unpadded Conv2D patch projections, including rectangular kernels in custom code. Other supported convolutions use the convolution neighborhood lesson; unverified variants retain general inspection. Large images and kernels use Windows of at most 8 × 8 cells, with direct index controls; feature lists show at most eight entries. Shapes-only runs show coordinates and transformations without numeric values.

## Explore recorded stages

New runs record module-call boundaries alongside individual operations. Runs with more than 24 operations start with their outer stages collapsed. Use **Stages** to browse nested calls, **Stage overview** to fold the model, or **All operations** to see the complete graph. Expanding a stage opens one level at a time, so transformer blocks can reveal their attention and feed-forward stages before exposing every operation. Calls with fewer than two recorded operations stay as ordinary operation nodes.

Each stage card displays its actual returned tensor. Select its header to inspect recorded entry/return tensors, choose among multiple operands or outputs, open the existing 3D viewer, or jump to any operation inside the call. Repeated calls to the same module remain distinct. Connections crossing collapsed stages retain their tensor dependencies, including branches and residual paths. Failed calls remain visible without an invented return value.

Playback and code navigation open the containing stages as needed. **Playback options → Reveal operations as they play** expands the graph and disables folding until it is turned off. Stage folding also pauses while playback is active. Grouping is a viewing preference and never changes model code or saved execution. Older runs without module-call metadata retain their original operation view; execute again to record stages.

## Indexed 3D tensors

The same orthographic cell geometry is used in node previews and the enlarged viewer. `[B, N, D]` is one volume with X = D, Y = N, Z = B. `[B, C, H, W]` shows separate batch volumes with X = W, Y = H, Z = C. Each batch is built from the same cell geometry. For higher ranks, earlier axes are held at explicitly selected coordinates.

Small tensors display every cell with rounded edges and narrow visual gutters that keep neighboring cells distinct. These gutters do not omit any indices. Dimensions up to 16 expand fully when they fit the view’s cell budget; larger or denser previews condense only the necessary axes. Condensed axes retain the first two indices, an ellipsis marking the exact omitted range, the final index, and the selected coordinate. Click **…** to open an omitted slice; isolation can expand previously condensed dimensions. Index spacing is compressed only across an explicitly marked omitted range.

Drag to rotate, use rotation buttons, or select Front/Reset. Every displayed cube has real logical coordinates; recorded runs fetch those exact values. Isolate a batch, channel, row, or column to inspect hidden interior cells. Coordinate controls and direct jumps reach any element. Arrow keys navigate width/height; Page Up/Down navigate depth. Node previews display at most 512 cells; enlarged views display at most 1,024. Paged values are loaded in batches of up to 256 indices per request. Before execution, builder nodes explicitly show inferred shapes with no fabricated values.

Choose **Slice** in the enlarged viewer to expose a flat plane through the selected cell. Choose any two axes as rows and columns; all remaining coordinates stay fixed and are listed above the plane. Choosing the other displayed axis swaps the view without permuting the tensor. Arrow keys follow those displayed axes, and coordinate controls move through batches, heads, channels, or other fixed dimensions. **Volume** restores the previous rotation with the same selected coordinate. When inspecting the active operation's result, selecting cells here also updates its contributor highlights on the model canvas. Volume and slice face lighting indicates orientation and selection, not numerical magnitude.

Drag or scroll to pan, pinch or Ctrl/Command + scroll to zoom, and use **Fit entire journey** for the overview. An expanded node stays readable regardless of canvas zoom; **Back to journey** or Escape closes it and restores your previous pan and zoom. **Fit entire journey**, Home while the canvas is focused, and the minimap explicitly reset the overview. The **Stages** menu lets you **Group into stages** or **Show every operation**; changing stages changes the layout, so it establishes a new overview instead of restoring coordinates from the old layout. Opening a side panel preserves the view's scale and center. Tiny or offscreen previews use explicit shape and element summaries; zooming in restores indexed cell geometry. Playback highlights successive operations; **Reveal steps** optionally reveals them gradually. The gear opens **Settings**, and the side bar holds **Run history**.

Operations that return multiple tensors have a selector on their node. Choose an output to see its actual name, shape, connected paths, and enlarged 3D tensor. Hover over a connection for its tensor shape and operand position. Separate return tensors and shared-storage dependencies retain their own connections, including when stages are collapsed. Inside nested library modules, intermediate results keep their operation names; only actual return tensors receive the caller's variable names.

**Executed code** is a source-only sidebar: select an executable line to open its operation in the center. Tensor values and explanations have one central inspection view. Specialized lessons still offer **Tensor details** when you want the general tensor inspection controls.

Open **Follow connections** in any enlarged input, operation, or stage to see where its tensors come from and which recorded steps use them next. Select a connection to follow a branch directly, including joins and shared-storage changes. Tensor names, shapes, and input positions identify each connection; long lists show six at a time. This follows actual tensor dependencies, while playback remains in execution order. Returned model tensors are identified separately from results with no later recorded use.

Every tensor is interactive, including inputs and matrix operands. Click a cell for its recorded value, coordinates, logical flat index, and storage position; use arrow keys to move, or Home/End to move to the first/last column. Slice controls keep the selection visible. Large row and column axes show page arrows and direct position fields beside the visible range. **Go to cell**, below the selected value, opens exact coordinate entry in one click. **View** opens viewing axes and 8×8 or 16×16 window settings when applicable. Viewing axes choose which two dimensions to display without changing the recorded tensor. Selecting an input in an overlapping `unfold` highlights matching outputs, including matches on other slices or pages. Large inverse relationships show up to 256 matches.

The Attention example uses randomly initialized parameters, not trained weights. It demonstrates mechanics; its attention patterns have no learned semantic meaning. Drawings show at most 8×8 cells by default, or 16×16 in the denser view, with zero-based index labels and previous/next controls. The focused 2D plane remains available alongside the indexed 3D viewer. Cell text and calculations are rounded; the element readout shows the value as its dtype stores it: a float32 with the fewest digits that read back as the same float32 (`1.041068`, not `1.0410679578781128`, which stays in the tooltip), and booleans as True and False, in grid cells too. Color intensity represents value magnitude; paged tensors scale shading to the visible window. Full values are retained independently of the compact grid labels.

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
def modify():
    view = x.view(2, 3)
    frozen = x.clone()
    view[0, 1] = 99
    x.add_(10)
    return view, frozen, x
```

With an input shape of `[1, 2, 3]`, select either write in the journey. **Inspect affected tensor** switches between the written tensor and its recorded shared views. Compare the same coordinate before and after, open either snapshot in 3D, or follow its earlier producer. The cloned tensor remains independent. **Operation operands** shows the actual arguments and return values; indexed assignment has side effects but no returned tensor.

When both snapshots hold their values (up to 4,096 elements), the cells whose values differ are highlighted in Before and After and counted, for the written tensor and for each shared view. A view that shares the storage but shows "no cell changed" was not touched by that write.

Dashed edges connect earlier shared views to the operation that writes their storage. Later operations use the refreshed snapshots. These connections track shared storage, not a per-cell write mask: disjoint slices or cells assigned the same value may be unchanged. Layout-only operations such as `transpose_` retain a separate explanation and do not refresh unchanged views.

Every observed alias update counts toward the snapshot budget. Large tensors remain paged; Shapes-only runs show no fabricated values. Native calls that bypass tracing and writes through NumPy/raw storage are not fully tracked. An observable change outside a recorded operation adds a **Run details → tracking notice**, without guessing its producer. Existing saved runs remain unchanged; rerun for mutation dependencies.

## Reuse an input configuration

Open **Inputs & settings** (the gear) in any project and expand **Saved inputs & NumPy files** on an input card. **Save current input** stores a named copy of its shape, axis names, generator, dtype, seed, random-stream setting, and recording mode. **Use this input** copies those settings into the current project and rechecks its component sequence. Later edits affect only that project. Each saved entry and each recorded run keeps its original settings.

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

Open **Inputs & settings → Model weights** in any project. Choose **Import checkpoint**, select a `.pt` or `.pth` file, name it, and save it to the local library. Importing does not change the active project. **Use checkpoint** pins that exact saved version to the project; **Use initialized weights** returns to the module's seeded initialization.

Export a standard tensor state dictionary from PyTorch:

```python
torch.save(model.state_dict(), "weights.pt")
```

Plain state dictionaries and the common `state_dict` / `model_state_dict` wrappers are accepted. Import uses `torch.load(weights_only=True)` with CPU mapping in a separate process, and never falls back to unrestricted pickle. Full model objects, TorchScript, legacy serialization, sparse/quantized/complex tensors, custom objects, and non-tensor extra state are outside this first version. Limits are 64 MiB per file, 63 MiB of logical tensor values, 2,048 tensors, eight dimensions, and 8,388,608 elements per tensor.

**Check compatibility** explicitly constructs the selected module on PyTorch's meta device and verifies exact parameter/buffer names, shapes, and dtypes, then loads the state without running `forward`. A value-dependent constructor may not support this metadata check. Run always checks again against the actual constructed model. Missing/unexpected keys and mismatches stop execution; no prefixes are stripped and no partial state is silently accepted. Floating weights must match the first input's floating dtype, which also sets the model dtype. Integer buffers retain their declared types. Builder checkpoints must use the parameter names in its generated `ComposedModel`.

Each run records the checkpoint name, immutable ID, SHA-256, and tensor manifest. **Run details** shows the reference, while operation inspectors show the recorded parameter values actually used. Checksums are verified before user code runs, including Shapes-only runs. In-place changes affect that run's model, not the saved checkpoint. Previous runs retain their reference even after a project chooses different weights. Back up `.data/weights/` with the database, inputs, and snapshots. This does not pin the PyTorch environment or guarantee reproducibility across versions.

## Multiple forward inputs

In a console or custom-code project, open **Settings → Forward inputs**. Expand an input card to edit its name, shape, values, dtype, seed, and axis labels; **Add tensor input** creates another independent tensor. Every card can reuse a saved input or import a NumPy file. The call preview shows the exact argument order and keyword names, for example `model(query, key, value, mask=mask)`.

For `forward(query, key, value, *, mask)`, configure query/key/value as positional inputs and mask as a keyword input. Names must be unique Python identifiers; keyword names must match the function. Positional inputs must precede keyword inputs. Run validates the Python signature before generating the input tensors, then records each tensor as a separate named graph root. Unused inputs remain visible, and each saved run retains every input configuration.

The first input keeps the existing model initialization behavior: its seed initializes the model and its floating dtype sets model parameter dtype. Other inputs keep their own dtypes. Independent random inputs use their own seeds, so changing one does not consume another's random stream. The recording mode applies to the whole call. There are at most eight tensor inputs, 8,388,608 elements per numeric input, and 32 million total input elements within the existing snapshot budget. Shapes-only runs retain the per-tensor logical limit. Non-tensor forward options and nested argument containers are not yet configurable.

The visual builder has one model input and supports branches through explicit earlier-output connections and two-input joins. Choose **Use as custom code** to configure a model with multiple external inputs.

## Large tensors

Set the builder input to `[1024, 32, 32]` to explore a real million-element input, then add compatible components. The browser requests only the visible cells from immutable snapshots, including after permutation and in-place mutation. Each request contains at most 256 logical indices. Slice controls handle large batch counts without creating huge menus.

**Input settings → Shapes only** (or **Settings**, or the values/shapes switch in the status bar) uses PyTorch meta tensors to inspect shape, strides, and layout without allocating numeric values. Shapes such as `[1024, 1024, 1024]` are supported. This mode is explicitly labeled and never supplies invented numbers. Data-dependent branches or operations without meta support may stop the trace; use a smaller value run for those.

Value runs allow up to 8,388,608 elements per tensor and 32 million elements across captured states. Shape runs allow up to 2^40 logical elements per tensor. Both retain the existing 256-operation and 20-second limits. These are bounded local execution modes, not arbitrary model-size support.

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
3. Select the entry `.py` file, the `nn.Module` class, and the import root (`.` or commonly `src`). **Import and configure** creates the project and opens Settings. Configure the constructor and tensor inputs before Run.
4. The source-file selector edits the entry file or any helper/configuration file. **Add source file** creates a relative path; additional files can be removed. Changing **Entry file** preserves the whole tree. Package-relative imports and `src` layouts are supported. The worker's working directory is the source root.
5. **Python environment** defaults to the backend's packages. To add dependencies, enter explicit `name==version` requirements, one per line, and choose **Create and select environment**. Setup accepts published wheels only, with no repository setup, editable installs, source builds, or automatic requirements-file installation. Run is separate from setup.
6. Run saves the exact source tree and configuration. Recorded operations and errors identify helper filenames and lines. Code inspection opens the appropriate captured file. Run details and history retain the imported revision, entry point, environment reference, and actual Python/package versions observed by that worker.

Source snapshots contain up to 128 Python/text/configuration files, 500 KB per file, and 2 MB total. Import a focused subdirectory for larger repositories. HTTP authentication, Git LFS assets, checkpoint downloads, repository installation, and arbitrary external data files are outside this importer. Local repositories provide a path for already authenticated clones. Missing dependencies or unsupported execution paths produce normal trace errors.

Dependency environments inherit the backend's base packages and add their own pinned packages; they are not hermetic lockfiles or a security sandbox. Base-package upgrades can affect them, so each run records versions again. Setup is serialized with execution and has bounded subprocess deadlines. Environments live under `.data/environments/`; recreate them after moving installations. Source files and Git provenance are stored in project/run JSON in SQLite. Imported source edits are independent of the original commit and are preserved in each run.

## Supported scope

Console scripts and source projects run on the same recorder. A source project runs one selected `torch.nn.Module` from a bounded Python source tree, with JSON constructor arguments and one to eight tensor inputs with positional or keyword binding. Evaluation mode and `torch.no_grad()` are used. Every run starts a fresh model with the supplied seed and then loads the selected checkpoint, if present.

- **Tensors**: CPU, dense, real. Inputs are sequential, random normal, zeros, ones, the sample image, a tokenized sentence, or an imported `.npy` file.
- **Operations with a linked, cell-level view**:
  - layout: reshape/view/flatten and their aliases, permute/transpose and their aliases, contiguous/clone, squeeze/unsqueeze, roll, `Tensor.unfold`
  - selection: indexing and slicing, index tensors and masks, gather/index_select/embedding, sort/topk, flip and other pure selections, expand/repeat, constant and reflect padding
  - joining: cat/stack/split/chunk/unbind
  - computation: matrix multiplication, einsum, dot/outer/mv, linear projection, elementwise arithmetic and comparisons with broadcasting, activations, where/masked_fill/tril/triu, reductions, cumsum/cumprod
  - networks: 1D/2D convolution, 2D max/average/adaptive-average pooling, patch embedding, layer normalization, evaluation-mode batch normalization, softmax/log_softmax, one_hot, cross_entropy/nll_loss, window attention
- **Everything else** that returns a tensor is recorded with its operands, arguments, source line, and values, and opens in the general inspector. Value-dependent cell mappings exist only in value runs of up to 4,096 elements.
- In-place writes record their shared-storage effects and the cells that changed.
- Not every Python statement is a tensor operation. Fused calls remain fused. Gradients and training, GPU execution, compilation, tensor subclasses, sparse and complex tensors, and custom C++ extensions are outside this milestone.
- A trace records only the path taken by that input. It is not a symbolic graph of all possible branches.
- **Limits**: 8,388,608 elements per tensor in value mode; 32 million elements across snapshots; 2^40 logical elements per tensor in shapes mode; 256 recorded operations; a 20-second worker deadline (10 seconds for a shape check). These are trace and execution budgets, **not a peak-memory guarantee**.

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

Set `PORT` and `TENSORVIEWER_API_URL` before `npm run dev` to run a second checkout beside the default ports.

Backend tests compare recorded results with uninstrumented PyTorch and with independent calculations: every cell of each mapping rule, reduction groups, einsum sums, class losses, convolution and pooling neighborhoods. They also cover storage semantics, snapshots preserved through mutation, retained failed steps, console wrapping and variable naming, shape checks, persistence, and worker timeouts. Frontend unit tests cover the pure logic behind each view: coordinate conversion, mapping rules and their inverses, motion layouts, diagnoses, line results and shape-check merging, completion, highlighting, links, run comparison, and graph connections. There are no browser-level tests; interface behavior is checked by hand against the list in the architecture guide.

See [the architecture guide](docs/architecture.md) for the source, composition, tracing, and presentation contracts.
