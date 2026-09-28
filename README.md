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

On an empty workspace, the application opens a blank model canvas. New projects also start empty. Nothing executes until Run is selected. Existing custom-code projects and saved executions remain available.

## First walkthrough

1. Choose **New project**, name it, and open the empty canvas.
2. Use **Set up input**, or open a tool category beneath Build / Explore / Code / Runs on the left rail. The toolbox is collapsed by default; category icons open a searchable component panel. Adding a component closes the panel and opens its settings. A first component supplies a compatible starter input. **Setup** configures the project name and recording mode; input settings include shape, generator, data type, and seed.
3. Select a component header or its step in the bottom navigator to change settings, reorder it, or remove it. Selection restores a readable zoom; the percentage button resets to 100%. Dimension cards pair exact sizes with axis names. The inspector provides previous/next navigation, named axis choices, and a keep-dimension switch for reductions. Connections display inferred output shapes. Invalid connections disable Run and explain the mismatch. For example, insert **Image to tokens** between convolution and attention.
4. Click **Run** to save the composition and record real PyTorch execution. **Explore** displays every recorded operation, including branches inside attention. **Build** returns to your editable sequence.
5. Select a node header to compare the incoming tensor, operation, and output. Select a cell on a node, or **3D** in its focused view, to enlarge that individual tensor.
6. **Code** displays generated Python. **Use as custom code** detaches the composition and enables editing/uploading your own module. This is a one-way conversion; the existing execution history retains earlier blueprints.

The library contains 28 executable components across Models, Spatial, Sequence, Layers, Activations, and Shape adapters. These include patch embedding, self-attention, transformer blocks, feed-forward networks, 1D/2D convolution, simple RNN, three pooling variants, linear projection, layer/batch normalization, five activations, image-to-token conversion, flatten, transpose, split/merge axes, insert/remove unit axes, unfold, contiguous, and mean reduction. Search always searches the complete library, even from a category panel. Batch normalization uses initialized running statistics in evaluation mode. Custom add-ons remain a future extension.

The first builder supports up to 16 components in a connected sequence. Transformer blocks include pre-norm attention, residuals, and a feed-forward network; multiple blocks are configurable. The explicit RNN supports up to 32 time steps. Arbitrary branches/joins in the builder and Git import remain later increments; the execution viewer already renders internal model branches.

## Indexed 3D tensors

The same orthographic cell geometry is used in node previews and the enlarged viewer. `[B, N, D]` is one volume with X = D, Y = N, Z = B. `[B, C, H, W]` shows separate batch volumes with X = W, Y = H, Z = C. Each batch is built from the same cell geometry. For higher ranks, earlier axes are held at explicitly selected coordinates.

Dimensions up to three display all indices. Larger dimensions display the first two, a labeled gap, and the final index; selecting an interior coordinate adds it to the preview. Each gap records its exact omitted interval. Click **…** to open an omitted slice. These are compressed indexed views, not proportional spatial renderings of the entire tensor.

Drag to rotate, use rotation buttons, or select Front/Reset. Every displayed cube has real logical coordinates; recorded runs fetch those exact values. Isolate a batch, channel, row, or column to inspect hidden interior cells. Coordinate controls and direct jumps reach any element. Arrow keys navigate width/height; Page Up/Down navigate depth. The renderer displays at most 256 cells, independent of tensor size. Before execution, builder nodes explicitly show inferred shapes with no fabricated values.

Drag or scroll to pan, pinch or Ctrl/Command + scroll to zoom, and use **Fit entire journey** for the overview. An expanded node stays readable regardless of canvas zoom; **Back to journey** or Escape closes it and restores the overview. Playback highlights successive operations; **Reveal steps** optionally reveals them gradually. The left rail opens **Code & inputs** and **Run history** beside the canvas.

Every tensor is interactive, including inputs and matrix operands. Click a cell for its recorded value, coordinates, logical flat index, and storage position; use arrow keys to move, or Home/End to move to the first/last column. Slice controls keep the selection visible. **Explore** opens viewing axes, 8×8 or 16×16 windows, page controls, and direct coordinate entry. Viewing axes choose which two dimensions to display without changing the recorded tensor. Selecting an input in an overlapping `unfold` highlights matching outputs, including matches on other slices or pages. Large inverse relationships show up to 256 matches.

The Attention example uses randomly initialized parameters, not trained weights. It demonstrates mechanics; its attention patterns have no learned semantic meaning. Drawings show at most 8×8 cells by default, or 16×16 in the denser view, with zero-based index labels and previous/next controls. The focused 2D plane remains available alongside the indexed 3D viewer. Cell text and calculations are rounded; the element readout shows the full recorded value. Color intensity represents value magnitude; paged tensors scale shading to the visible window. Full values are retained independently of the compact grid labels.

Projects and immutable run metadata are saved in `backend/.data/tensorviewer.sqlite3`. Large value snapshots are stored alongside it in `backend/.data/snapshots/`; keep both when backing up the workspace. Set `TENSORVIEWER_DATA_DIR` to use a different directory. Run history lists the latest 20 runs; older runs remain in the local database.

## Large tensors

Set the builder input to `[1024, 32, 32]` to explore a real million-element input, then add compatible components. The browser requests only the visible cells from immutable snapshots, including after permutation and in-place mutation. Each request contains at most 256 logical indices. Slice controls handle large batch counts without creating huge menus.

**Input settings → Shapes only** (or **Code & inputs** for custom projects) uses PyTorch meta tensors to inspect shape, strides, and layout without allocating numeric values. Shapes such as `[1024, 1024, 1024]` are supported. This mode is explicitly labeled and never supplies invented numbers. Data-dependent branches or operations without meta support may stop the trace; use a smaller value run for those.

Value runs allow up to 8,388,608 elements per tensor and 32 million elements across captured states. Shape runs allow up to 2^40 logical elements per tensor. Both retain the existing 256-operation and 20-second limits. These are bounded local execution modes, not arbitrary model-size support.

## Supported scope

This first release runs one Python file, one selected `torch.nn.Module` class, JSON constructor arguments, and one generated positional tensor input `forward(x)`. You can paste code or upload a `.py` file. Evaluation mode and `torch.no_grad()` are used. Every run starts a fresh model with the supplied seed.

- CPU, dense real tensors; input generators: sequential, random normal, zeros, and ones.
- Rich lessons: reshape/view/flatten, permute/transpose, contiguous/clone, squeeze/unsqueeze, Tensor.unfold, matrix multiplication, linear projection, softmax, basic arithmetic, and reductions. Interactive element mappings are available for the supported layout operations; dot-product interaction supports operands of rank two and above.
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
