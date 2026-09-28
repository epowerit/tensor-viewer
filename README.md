# TensorViewer

A local learning workspace for seeing what PyTorch operations do to tensors. Paste a small `nn.Module`, configure its input, and step through an actual forward pass.

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

On an empty workspace, the application creates and executes the bundled Attention example. Reopening an existing project loads its most recent saved run without executing its code.

## First walkthrough

1. Open **Code & inputs** to inspect the Attention class and change constructor arguments, input dimensions, dtype, generator, or seed.
2. Click **Run forward pass**. This saves the current project and records a new execution.
3. Select a `reshape` or `permute` step. Click an output cell to locate the same element in the input; the input slice follows the selected coordinate.
4. Select a `matmul` step. Click an output cell to highlight its contributing row and column and see the dot-product terms.
5. Select `softmax` to inspect the normalization group, or expand **Storage & strides** to see when tensors share storage.
6. Use **New project → Tensor basics** for a smaller example with numbered elements.

The Attention example uses randomly initialized parameters, not trained weights. It demonstrates mechanics; its attention patterns have no learned semantic meaning. Matrices larger than eight rows or columns are paged. Stacked sheets are schematic and the controls select a two-dimensional slice.

Projects and immutable run snapshots are saved in `backend/.data/tensorviewer.sqlite3`. Set `TENSORVIEWER_DATA_DIR` to use a different directory. Run history lists the latest 20 runs; older runs remain in the local database.

## Supported scope

This first release runs one Python file, one selected `torch.nn.Module` class, JSON constructor arguments, and one generated positional tensor input `forward(x)`. You can paste code or upload a `.py` file. Evaluation mode and `torch.no_grad()` are used. Every run starts a fresh model with the supplied seed.

- CPU, dense real tensors; input generators: sequential, random normal, zeros, and ones.
- Rich lessons: reshape/view/flatten, permute/transpose, contiguous/clone, squeeze/unsqueeze, Tensor.unfold, matrix multiplication, linear projection, softmax, basic arithmetic, and reductions. Interactive element mappings are available for the supported layout operations; dot-product interaction supports operands of rank two and above.
- Other intercepted tensor-returning calls get a generic inspection view. Not every Python statement is a tensor operation. Fused calls remain fused; custom extensions, compilation, tensor subclasses, training, and GPU execution are outside this milestone.
- A trace records only the path taken by that input. It is not a symbolic graph of all possible branches.
- Limits: 4,096 input elements; 16,384 elements per captured tensor; 250,000 elements across snapshots; 256 recorded operations; a 20-second worker deadline. These are trace/execution budgets, **not a peak-memory guarantee**.

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

Backend tests compare the recorded Attention output with uninstrumented PyTorch, verify exact index mappings and storage semantics, preserve snapshots through mutation, retain failed steps, and exercise persistence and worker timeouts. Frontend unit tests verify coordinate conversion, broadcasted dot-product contributors, and normalization groups.

See [the architecture guide](docs/architecture.md) for extension points and the path toward multi-file/Git projects.
