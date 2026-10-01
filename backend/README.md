# TensorViewer backend

```sh
uv sync --locked --python 3.11
uv run uvicorn tensorviewer.app:app --host 127.0.0.1 --port 8000
```

The server is one process with one execution at a time. User code never runs in it: every run, shape check, and compatibility check starts a fresh worker subprocess with a deadline.

## Modules

Execution and recording:

- `runner.py`: worker subprocess lifecycle, timeout, and process-group cleanup.
- `worker.py`: trusted code loading, seeded model and input construction, weight loading, and the forward call.
- `tracing.py`: the `TorchFunctionMode` recorder. It captures operands and results as immutable tensor states, source locations, module calls, argument names, variable names from source spans, and axis names.
- `mutations.py`: storage, version, and layout observation around each call, for in-place writes and shared views.
- `snapshots.py`: paged value snapshots for tensors above the inline limit.
- `statistics.py`: chunked full-group statistics read from snapshots.

Explaining operations (`operations/`). Adapters describe what was recorded; none computes a substitute result:

- `registry.py`: maps PyTorch function names to adapters.
- `layout.py`: reshape, permute, and their aliases, as verified index mappings.
- `relations.py`: compact cell rules for indexing, repetition, lookups, reductions, running totals, einsum, padding, elementwise work, one-hot, and class losses, plus replayed maps for pure selections.
- `compute.py`, `assembly.py`, `convolution.py`, `pooling.py`, `normalization.py`, `spatial.py`: matrix products and arithmetic, joins and splits, convolution and pooling neighborhoods, layer normalization, and patch projection.

Projects and inputs:

- `models.py`: versioned project, input, trace, tensor, and lesson contracts.
- `console.py`: wraps a console script in a module without rewriting its text.
- `composer.py`: the builder's component registry, shape inference, validation, and generated Python.
- `custom_components.py`: checks and namespaces for saved custom components.
- `vision.py`, `hierarchy.py`, `window_attention.py`: inspectable source for the built-in vision models and window attention.
- `source_projects.py`: multi-file source trees and Git snapshot import.
- `environments.py`: optional per-project dependency environments.
- `input_files.py`, `samples.py`: imported `.npy` inputs, the sample image, and sentence tokens.
- `weights.py`: restricted checkpoint import and compatibility checks.
- `templates.py`: bundled example projects.

Service:

- `app.py`: thin versioned API routes.
- `storage.py`: SQLite projects, immutable runs, and asset metadata.

## API

All application endpoints are under `/api/v1`. Interactive documentation is served at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service status and schema version |
| GET | `/templates` | Bundled example projects |
| GET, POST | `/projects` | List or create projects |
| GET, PUT | `/projects/{id}` | Read or replace a project draft |
| GET, POST | `/projects/{id}/runs` | List recent runs, or execute the saved draft and store the run |
| GET | `/runs/{id}` | Read the complete immutable run |
| GET | `/runs/{id}/tensors/{tensor_id}/values?indices=…` | Read up to 256 exact indexed values |
| GET | `/runs/{id}/operations/{operation_id}/normalization?group=…` | Full-group statistics for a layer normalization step |
| POST | `/shape-check` | Dry-run a draft on metadata tensors; nothing is stored |
| GET | `/toolbox` | Builder component catalog, including saved custom components |
| POST | `/compose` | Validate a blueprint and infer shapes without executing the model |
| POST | `/compose/check` | Execute custom components on metadata tensors to check a blueprint |
| GET, POST | `/components` | List or save custom components |
| GET, POST | `/input-fixtures` | List or save input configurations |
| POST | `/input-fixtures/upload` | Import a `.npy` tensor as a saved input |
| GET | `/weights` | List saved checkpoint metadata |
| POST | `/weights/upload` | Import a `.pt`/`.pth` tensor state dictionary |
| POST | `/weights/check` | Metadata-only compatibility check for a project draft |
| POST | `/sources/git` | Read a committed source snapshot without executing it |
| GET, POST | `/environments` | List dependency environments, or create one from pinned requirements |

A run request returns HTTP 201 even when the user's code fails: the run exists and its `trace.error` explains the failure. Earlier operations remain inspectable. A shape check likewise returns 200 with the failure in its trace. Request validation errors return 422 and missing resources return 404. Runs, shape checks, custom-component checks, compatibility checks, and environment setup share one lock; a request made while another holds it returns 409.

Endpoints that execute user code are `POST /projects/{id}/runs`, `/shape-check`, `/compose/check`, and `/weights/check` (construction only). Every other endpoint reads or stores data without importing project code.

## Axis annotations

Shapes do not prove semantic axis names. Supply optional input labels and annotate output assignments explicitly:

```python
q = q.permute(0, 2, 1, 3)  # axes: batch, heads, tokens, head_features
```

Names with a mismatched rank are discarded in favor of neutral labels. Without an annotation, names follow an operation only where its rule preserves their meaning: a permutation reorders them, a slice keeps the sliced axes, a reduction drops the reduced ones, and an elementwise result takes the names of an operand with the same shape. No square spatial grid is inferred from a token count.

## Variable names

A tensor is named after a variable only when its expression is the whole assigned value. In `scores = q @ k.transpose(-2, -1)` the product is `scores` and the transpose is `transpose`. `a, b = x.chunk(2)` names both results. The recorder decides this from the source span of the running instruction, so several statements on one line and multi-line values are handled.

## Extend an operation

Pick the narrowest mechanism that is exact:

1. **A closed-form cell rule**: add a validated rule to `operations/relations.py` and its evaluator to `frontend/src/operations/relations.ts`. Rules are constant size and work for tensors of any size and in shapes mode.
2. **A pure selection whose source depends on values**: add the function name to `REPLAYED`. The recorder applies the same call to a tensor of positions and keeps the result only if it reproduces every recorded output value.
3. **A dedicated view**: add an adapter module, register its function names in `operations/registry.py`, and add a presenter in the frontend.

Return a generic lesson when exact correspondence is unsupported; an operation that cannot be explained exactly must stay inspectable, not approximated. Record argument names in `tracing.arguments_for` and semantic operand order in `operands_for`. Write a test that compares every output cell with PyTorch or an independent calculation.

The API is the source of truth for TypeScript types. After editing models:

```sh
uv run python -c 'import json; from tensorviewer.app import app; open("../frontend/openapi.json", "w").write(json.dumps(app.openapi(), indent=2) + "\n")'
cd ../frontend
npm run generate:api
```

A running server keeps the models it started with. Restart it after changing `models.py` or `app.py`; workers always use the current code.

See the root README for the trusted-code boundary and current limitations.
