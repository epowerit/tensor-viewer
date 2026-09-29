# TensorViewer

A local learning workspace for seeing what PyTorch operations do to tensors. Build a sequence from the component toolbox, or paste a small `nn.Module`, then inspect an actual forward pass.

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

On an empty workspace, the application opens a blank model canvas. New projects also start empty. Model code executes only when Run or Check custom shapes is selected. Existing custom-code projects and saved executions remain available.

## First walkthrough

1. Choose **New project**, name it, and open the empty canvas.
2. Use **Set up input**, or open a category in the horizontal toolbar above the canvas. Build / Explore / Code / Runs stay on the left rail. The component panel is collapsed by default; toolbar icons open its searchable categories. Use Left/Right or Home/End to navigate the toolbar, and Escape to close its panel. Adding a component closes the panel and opens its settings. A first component supplies a compatible starter input. **Setup** configures the project name and recording mode; input settings include shape, generator, data type, and seed.
3. Select a component header or its step in the bottom navigator to change settings, reorder it, or remove it. Selection restores a readable zoom; the percentage button resets to 100%. Dimension cards pair exact sizes with axis names. The inspector provides previous/next navigation, named axis choices, and a keep-dimension switch for reductions. Connections display inferred output shapes. Invalid connections disable Run and explain the mismatch. For example, insert **Image to tokens** between convolution and attention.
4. Click **Run** to save the composition and record real PyTorch execution. **Explore** displays every recorded operation, including branches inside attention. **Build** returns to your editable sequence.
5. Select a node header to compare the incoming tensor, operation, and output. Select a cell on a node, or **3D** in its focused view, to enlarge that individual tensor.
6. **Code** displays generated Python. **Use as custom code** detaches the composition and enables editing/uploading your own module. This is a one-way conversion; the existing execution history retains earlier blueprints.

The library contains 28 built-in components across Models, Spatial, Sequence, Layers, Activations, and Shape adapters, plus your saved Custom components. These include patch embedding, self-attention, transformer blocks, feed-forward networks, 1D/2D convolution, simple RNN, three pooling variants, linear projection, layer/batch normalization, five activations, image-to-token conversion, flatten, transpose, split/merge axes, insert/remove unit axes, unfold, contiguous, and mean reduction. Search always searches the complete library, even from a category panel. Batch normalization uses initialized running statistics in evaluation mode. The Custom category holds reusable modules you save locally.

The first builder supports up to 16 components in a connected sequence. Transformer blocks include pre-norm attention, residuals, and a feed-forward network; multiple blocks are configurable. The explicit RNN supports up to 32 time steps. Arbitrary branches/joins in the builder and Git import remain later increments; the execution viewer already renders internal model branches.

## Reusable custom components

1. Open the **Custom** puzzle icon in the component toolbar and choose **New custom component**.
2. Paste a module or use **Import .py**, select its class, name it, and supply default constructor arguments as a JSON object. **Save & add component** saves it to the library and inserts a copy in the current sequence. Saving checks Python syntax without executing it.
3. Edit **Constructor arguments** on a node and choose **Apply arguments**. Settings belong to that node; library defaults stay unchanged.
4. Choose **Check custom shapes**. The backend runs the module and the assembled sequence on PyTorch meta tensors in separate worker processes. This checks connections, non-contiguous inputs, and recording limits without allocating the requested numeric tensors. Run becomes available when the sequence passes.
5. **Run** records the actual operations and values in Explore. Source lines and variable names inside custom modules remain linked to the generated code.

The first custom-component contract is `nn.Module.forward(x) → tensor`, with one to six non-empty dimensions and an unchanged dtype. Use installed dependencies; modules that require numeric values to determine their output cannot be shape-checked in the builder. They can still be explored as ordinary custom-code projects with small value inputs.

Library entries are immutable. **Edit source as new version** saves a new entry and updates only the selected node. Other projects and past runs retain their original source and arguments. Multiple components can define the same Python class name: each has its own namespace. Check results are bounded, session-local caches tied to the exact code, arguments, input configuration, and sequence. Changing the sequence or restarting the backend may require another check. Every check and run uses the same local trusted-code execution model described below.

## Follow a patch into a token

Add **Models → Patch embedding**, choose a patch size, and run the sequence. Select its convolution, flatten, or transpose node to open the patch lesson. The progression follows the recorded image → projected patch grid → flattened grid → token rows, with exact shapes and selected coordinates at each stage.

Select a spatial patch to highlight its source image region, then choose a batch, input channel, pixel, and output feature. The paired pixel and kernel grids show matching recorded operands. The calculation identifies one contribution to the feature; the full projection sums every channel and patch position, plus bias. Flatten and transpose preserve the projected values. **Tensor details** returns to the regular tensor inspector and 3D viewer without discarding the lesson selection.

The lesson handles complete, non-overlapping, unpadded Conv2D patch projections, including rectangular kernels in custom code. General convolutions retain the standard inspector. Large images and kernels use windows of at most 8 × 8 cells, with direct index controls; feature lists show at most eight entries. Shapes-only runs show coordinates and transformations without numeric values.

## Explore recorded stages

New runs record module-call boundaries alongside individual operations. Runs with more than 24 operations start with their outer stages collapsed. Use **Stages** to browse nested calls, **Stage overview** to fold the model, or **All operations** to see the complete graph. Expanding a stage opens one level at a time, so transformer blocks can reveal their attention and feed-forward stages before exposing every operation. Calls with fewer than two recorded operations stay as ordinary operation nodes.

Each stage card displays its actual returned tensor. Select its header to inspect recorded entry/return tensors, choose among multiple operands or outputs, open the existing 3D viewer, or jump to any operation inside the call. Repeated calls to the same module remain distinct. Connections crossing collapsed stages retain their tensor dependencies, including branches and residual paths. Failed calls remain visible without an invented return value.

Playback and code navigation open the containing stages as needed. **Reveal steps** expands the graph and disables folding until it is turned off. Grouping is a viewing preference and never changes model code or saved execution. Older runs without module-call metadata retain their original operation view; execute again to record stages.

## Indexed 3D tensors

The same orthographic cell geometry is used in node previews and the enlarged viewer. `[B, N, D]` is one volume with X = D, Y = N, Z = B. `[B, C, H, W]` shows separate batch volumes with X = W, Y = H, Z = C. Each batch is built from the same cell geometry. For higher ranks, earlier axes are held at explicitly selected coordinates.

Small tensors display all cells as continuous grids. Dimensions up to 16 expand fully when they fit the view’s cell budget; larger or denser previews condense only the necessary axes. Condensed axes retain the first two indices, a labeled gap, the final index, and the selected coordinate. Each gap records its exact omitted interval. Click **…** to open an omitted slice; isolation can expand previously condensed dimensions. Geometry is compressed only where a gap is shown.

Drag to rotate, use rotation buttons, or select Front/Reset. Every displayed cube has real logical coordinates; recorded runs fetch those exact values. Isolate a batch, channel, row, or column to inspect hidden interior cells. Coordinate controls and direct jumps reach any element. Arrow keys navigate width/height; Page Up/Down navigate depth. Node previews display at most 512 cells; enlarged views display at most 1,024. Paged values are loaded in batches of up to 256 indices per request. Before execution, builder nodes explicitly show inferred shapes with no fabricated values.

Drag or scroll to pan, pinch or Ctrl/Command + scroll to zoom, and use **Fit entire journey** for the overview. An expanded node stays readable regardless of canvas zoom; **Back to journey** or Escape closes it and restores the overview. Playback highlights successive operations; **Reveal steps** optionally reveals them gradually. The left rail opens **Code & inputs** and **Run history** beside the canvas.

Every tensor is interactive, including inputs and matrix operands. Click a cell for its recorded value, coordinates, logical flat index, and storage position; use arrow keys to move, or Home/End to move to the first/last column. Slice controls keep the selection visible. **Explore** opens viewing axes, 8×8 or 16×16 windows, page controls, and direct coordinate entry. Viewing axes choose which two dimensions to display without changing the recorded tensor. Selecting an input in an overlapping `unfold` highlights matching outputs, including matches on other slices or pages. Large inverse relationships show up to 256 matches.

The Attention example uses randomly initialized parameters, not trained weights. It demonstrates mechanics; its attention patterns have no learned semantic meaning. Drawings show at most 8×8 cells by default, or 16×16 in the denser view, with zero-based index labels and previous/next controls. The focused 2D plane remains available alongside the indexed 3D viewer. Cell text and calculations are rounded; the element readout shows the full recorded value. Color intensity represents value magnitude; paged tensors scale shading to the visible window. Full values are retained independently of the compact grid labels.

Projects and immutable run metadata are saved in `backend/.data/tensorviewer.sqlite3`. Large value snapshots live in `backend/.data/snapshots/`, and imported input tensors in `backend/.data/inputs/`; keep all three when backing up the workspace. Set `TENSORVIEWER_DATA_DIR` to use a different directory. Run history lists the latest 20 runs; older runs remain in the local database.

## Follow a linear projection

Add **Layers → Linear projection**, or select a recorded `linear` step inside attention or a transformer. The lesson links an input vector, the corresponding weight row, and one output feature. Selecting a cell in any of the three tensors, including their enlarged 3D views, updates the others while preserving the appropriate batch/token coordinates. Direct feature controls and the existing coordinate jump reach any index.

The calculation shows the selected product, bias when present, and PyTorch's recorded output. **Inspect matching products** opens an eight-feature contribution window. Projections with at most 256 input features also show a sum over all products; larger projections show an explicitly partial window sum. Neither partial sums nor rounded browser arithmetic replace the recorded result. Shape-only runs retain the same coordinate relationships without numeric values. Standard matrix weights and optional vector bias are supported; other operand layouts retain the general inspector.

## Replay an element's layout change

Select an element in a reshape, view, flatten, permute, transpose, squeeze/unsqueeze, contiguous, or clone step, then choose **Follow element**. Play, pause, scrub, or step through its original coordinate, index rule, and result coordinate. A moving marker follows the actual recorded mapping. The replay explains logical coordinates; storage sharing and recorded storage offsets are shown separately so an axis permutation is not mistaken for a physical memory move.

The input and output windows each show at most 4 × 8 cells from the last two axes, with leading coordinates fixed and labeled. Select a window cell or use **Go to coordinate** to reach another element, including in large paged or shape-only runs. Selecting a different element resets playback. **Tensor details** returns to the full explorers and their 3D views with the selection preserved. Reduced-motion preferences replace automatic travel with manual steps. One-to-many unfold mappings, dtype reinterpretations, empty tensors, and unverified mappings retain the existing inspection view.

## Reuse an input configuration

Open **Build → Input tensor → Saved inputs** (or the Input tensor section in **Code & inputs** for a custom-code project). **Save current input** stores a named copy of its shape, axis names, generator, dtype, seed, random-stream setting, and recording mode. **Use this input** copies those settings into the current project and rechecks its component sequence. Later edits affect only that project. Each saved entry and each recorded run keeps its original settings.

The library starts collapsed and supports numeric and large shape-only configurations. Saving generated settings does not allocate values. Multiple forward inputs and trained weights are not included yet.

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

## Multiple forward inputs

In a custom-code project, open **Code → Forward inputs**. Expand an input card to edit its name, shape, values, dtype, seed, and axis labels; **Add tensor input** creates another independent tensor. Every card can reuse a saved input or import a NumPy file. The call preview shows the exact argument order and keyword names, for example `model(query, key, value, mask=mask)`.

For `forward(query, key, value, *, mask)`, configure query/key/value as positional inputs and mask as a keyword input. Names must be unique Python identifiers; keyword names must match the function. Positional inputs must precede keyword inputs. Run validates the Python signature before generating the input tensors, then records each tensor as a separate named graph root. Unused inputs remain visible, and each saved run retains every input configuration.

The first input keeps the existing model initialization behavior: its seed initializes the model and its floating dtype sets model parameter dtype. Other inputs keep their own dtypes. Independent random inputs use their own seeds, so changing one does not consume another's random stream. The recording mode applies to the whole call. There are at most eight tensor inputs, 8,388,608 elements per numeric input, and 32 million total input elements within the existing snapshot budget. Shapes-only runs retain the per-tensor logical limit. Non-tensor forward options and nested argument containers are not yet configurable.

The visual sequence builder and reusable sequence components still use one tensor in/out. Choose **Use as custom code** to configure a multi-input call; arbitrary branch/join authoring comes later.

## Large tensors

Set the builder input to `[1024, 32, 32]` to explore a real million-element input, then add compatible components. The browser requests only the visible cells from immutable snapshots, including after permutation and in-place mutation. Each request contains at most 256 logical indices. Slice controls handle large batch counts without creating huge menus.

**Input settings → Shapes only** (or **Code & inputs** for custom projects) uses PyTorch meta tensors to inspect shape, strides, and layout without allocating numeric values. Shapes such as `[1024, 1024, 1024]` are supported. This mode is explicitly labeled and never supplies invented numbers. Data-dependent branches or operations without meta support may stop the trace; use a smaller value run for those.

Value runs allow up to 8,388,608 elements per tensor and 32 million elements across captured states. Shape runs allow up to 2^40 logical elements per tensor. Both retain the existing 256-operation and 20-second limits. These are bounded local execution modes, not arbitrary model-size support.

## Supported scope

This first release runs one Python file, one selected `torch.nn.Module` class, JSON constructor arguments, and one to eight generated or uploaded tensor inputs with positional or keyword binding. You can paste code or upload a `.py` file. Evaluation mode and `torch.no_grad()` are used. Every run starts a fresh model with the supplied seed.

- CPU, dense real tensors; input generators: sequential, random normal, zeros, and ones, plus `.npy` tensor imports.
- Rich lessons: reshape/view/flatten, permute/transpose, contiguous/clone, squeeze/unsqueeze, Tensor.unfold, matrix multiplication, linear projection, patch embedding, softmax, basic arithmetic, and reductions. Interactive element mappings are available for the supported layout operations; dot-product interaction supports operands of rank two and above.
- Other intercepted tensor-returning calls get a generic inspection view. Not every Python statement is a tensor operation. Fused calls remain fused; custom extensions, compilation, tensor subclasses, training, and GPU execution are outside this milestone.
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

See [the architecture guide](docs/architecture.md) for extension points and the path toward multi-file/Git projects.
