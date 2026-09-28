# TensorViewer frontend

```sh
npm ci
npm run dev
```

Run the Python backend on port 8000. Vite runs on port 5173 and proxies `/api` to the backend. `npm run build` produces the frontend bundle; `npm run preview` previews that bundle (configure an API reverse proxy for a standalone deployment).

## Modules

- `api/`: request client and generated OpenAPI types. Responses include Pydantic defaults; UI aliases make those defaults required.
- `builder/`: blank canvas, collapsed tool categories in the main navigation rail, searchable icon-based component flyout (`Toolbox.tsx`), sequential composition, inferred shape previews, and project/input/component settings.
- `journey/`: recorded-dependency graph, branch layout, pan/zoom canvas, tensor thumbnails, expanded transformation view, and linked source inspector.
- `tensors/`: reusable tensor renderer, coordinate transforms, and semantic coordinate tests.
- `operations/`: operation scene and a registry of interaction presenters. Presenters consume recorded tensors; they do not run uploaded algorithms.
- `components/`: project editor, new-project dialog, and walkthrough navigation.
- `App.tsx`: workspace state, saving, execution, and history.

The full recorded path appears on a dark canvas, left to right, with parallel branches stacked vertically. Select a node to expand its input → operation → result in a readable central panel, with interactive values, shapes, and slices. The path remains behind the panel. **Code** opens the linked inspector; selecting an executed source line also expands its operation. **Back to journey**, Escape within the expanded view, or **Fit entire journey** restores the overview. Drag or scroll to pan and pinch or Ctrl/Command + scroll to zoom. Keyboard navigation on the canvas supports arrows to pan, `+`/`-` to zoom, and `0`/Home to fit. **Reveal steps** optionally reveals the path in execution order during playback. The left rail opens editing and saved runs without leaving the canvas.

`TensorCard` gives every tensor a local selection, hover readout, one keyboard tab stop, arrow-key navigation, paging, and slice controls. **Axes** changes only the visible plane; coordinate arrays always retain the recorded tensor's axis order. `tensors/plane.ts` owns coordinate/view calculations and stride-based storage positions. Operation presenters supply linked highlights separately from the local selection: inputs and operands remain inspectable even when no exact transformation relationship is supported. Exact layout mappings retain all inverse memberships for overlapping windows.

The same tensor card is used for layout changes, attention operands, and generic inspection. The presenter registry independently defines index tracking, dot products, normalization, and generic inspection. The 2D plane remains available for dense numeric inspection. `TensorVolume` projects indexed 3D cells in both node thumbnails and a separately enlarged, rotatable dialog. Gaps carry exact omitted ranges; no layer is represented by a decorative blank rectangle. These are logical views, not drawings of physical memory. See `volume.ts` for sampling and geometry, and `volume.test.ts` for rank, gap, and coordinate invariants.

Checks:

```sh
npm test
npm run build
npm run format:check
```

Regenerate types from the committed contract with `npm run generate:api`. Export an updated backend contract first when Python models change; instructions are in the backend README.
