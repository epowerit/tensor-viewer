# TensorViewer frontend

```sh
npm ci
npm run dev
```

Run the Python backend on port 8000. Vite runs on port 5173 and proxies `/api` to the backend. `npm run build` produces the frontend bundle; `npm run preview` previews that bundle (configure an API reverse proxy for a standalone deployment).

## Modules

- `api/`: request client and generated OpenAPI types. Responses include Pydantic defaults; UI aliases make those defaults required.
- `tensors/`: reusable tensor renderer, coordinate transforms, and semantic coordinate tests.
- `operations/`: operation scene and a registry of interaction presenters. Presenters consume recorded tensors; they do not run uploaded algorithms.
- `components/`: project editor, new-project dialog, and walkthrough navigation.
- `App.tsx`: workspace state, saving, execution, and history.

The same tensor card is used for layout changes, attention operands, and generic inspection. The presenter registry independently defines index tracking, dot products, normalization, and generic inspection. The SVG shows a two-dimensional slice and schematic additional sheets; it is not a drawing of physical memory.

Checks:

```sh
npm test
npm run build
npm run format:check
```

Regenerate types from the committed contract with `npm run generate:api`. Export an updated backend contract first when Python models change; instructions are in the backend README.
