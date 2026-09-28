# TensorViewer backend

```sh
uv sync --locked --python 3.11
uv run uvicorn tensorviewer.app:app --host 127.0.0.1 --port 8000
```

## Modules

- `models.py`: versioned trace, project, tensor, and lesson contracts.
- `composer.py`: component registry, sequential shape inference, size validation, and canonical generated Python.
- `runner.py`: per-run subprocess lifecycle and timeout.
- `worker.py`: trusted code loading, seeded model/input construction, and forward execution.
- `tracing.py`: `TorchFunctionMode` call recording, source locations, module context, tensor snapshots, and storage identity.
- `operations/`: reusable semantic adapters. These explain actual execution rather than calculating substitute results.
- `storage.py`: SQLite project persistence and immutable run records.
- `app.py`: thin versioned API routes.
- `templates.py`: editable teaching examples.

## API

All application endpoints are under `/api/v1`:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service status and schema version |
| GET | `/templates` | Bundled example projects |
| GET | `/toolbox` | Component catalog and editable parameter definitions |
| POST | `/compose` | Validate a blueprint and infer shapes without executing the model |
| GET, POST | `/projects` | List or create projects |
| GET, PUT | `/projects/{id}` | Read or replace a project draft |
| GET, POST | `/projects/{id}/runs` | List recent runs or execute the saved draft |
| GET | `/runs/{id}` | Read the complete immutable run |
| GET | `/runs/{id}/tensors/{tensor_id}/values?indices=…` | Read up to 256 exact indexed values |

A run request returns HTTP 201 even when the user's code fails: the run exists and its `trace.error` explains the failure. Earlier operations remain inspectable. Request validation errors return 422; missing resources return 404. Only one run is accepted at a time in this local single-worker server; a concurrent run returns 409.

## Axis annotations

Shapes do not prove semantic axis names. Supply optional input labels and annotate output assignments explicitly:

```python
q = q.permute(0, 2, 1, 3)  # axes: batch, heads, tokens, head_features
```

Use one tensor operation per annotated line for clear source mapping. Names with a mismatched rank are discarded in favor of neutral labels. An unannotated permutation can propagate labels from its input. No square spatial grid is inferred from a token count.

## Extend an operation

Add an adapter to `operations/`, register its PyTorch function name in `operations/registry.py`, and write a semantic test. Return a generic lesson when exact correspondence is unsupported. Input references preserve argument positions, including repeated operands. Matrix multiplication keyword operands are normalized into semantic order.

The API is the source of truth for TypeScript types. After editing models:

```sh
uv run python -c 'import json; from tensorviewer.app import app; open("../frontend/openapi.json", "w").write(json.dumps(app.openapi(), indent=2) + "\n")'
cd ../frontend
npm run generate:api
```

See the root README for the trusted-code boundary and current limitations.
