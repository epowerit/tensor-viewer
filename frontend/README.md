# TensorViewer frontend

```sh
npm ci
npm run dev
```

Run the Python backend on port 8000. Vite runs on port 5173 and proxies `/api` to the backend; set `PORT` and `TENSORVIEWER_API_URL` to use other ports. `npm run build` produces the bundle and `npm run preview` previews it (configure an API reverse proxy for a standalone deployment).

## Layout

`App.tsx` owns workspace state, saving, execution, shape checks, history, and links, and arranges the interface:

- **Title bar**: project search (Ctrl/⌘ + K), Run, save state, and toggles for the tensor shelf and the code panel.
- **Left rail**: Projects and files, Run history, and Settings.
- **Starting tensor strip**: the input's values, shape, and data type, with Examples and ＋ Operation for console projects.
- **Canvas**: the recorded journey, or the builder for canvas projects. Selecting a recorded step frames its operands and results directly on the canvas. **Inspect** opens detailed values and explanations; returning keeps the same transformation in view. On the canvas, Enter opens inspection, Escape returns to the model overview, and Space plays or pauses. Fit (0) shows the entire model.
- **Tensor shelf**: Tensors, Run notes, and Printed output for the displayed run.
- **Code panel**: the source editor, resizable, optional.
- **Status bar**: run state, note counts, shape checking, the selected step and cell, and the recording mode.

## Modules

- `api/`: request client and generated OpenAPI types. Responses include Pydantic defaults; UI aliases make those defaults required. `scriptRun` presents console runs in the coordinates of the user's script.
- `shell/`: the explorer, run history, input strip, tensor shelf, status bar, and the collector behind Run notes (`problems.ts`).
- `editor/`: the code editor (a text area over a highlighted layer), syntax coloring, shape-aware completion, shape glyphs, and tensor hover previews.
- `console/`: console examples and snippets, per-line results from a run or shape check (`script.ts`), and the variables behind the tensor shelf.
- `journey/`: the recorded-dependency graph, branch layout, stage folding, pan/zoom canvas, tensor thumbnails, the expanded transformation view, and the linked source inspector. `sceneGraph.ts` projects each active operand and result into a separate tensor actor while preserving canonical operation IDs; `sceneSemantics.ts` supplies roles and explanations from verified recorded metadata. `cellMotion.ts` plans bounded verified coordinate paths; `CanvasCellMotion` projects the rendered voxel faces into those paths. `useSceneClock` shares pause-preserving playback progress without rerendering the graph on every frame.
- `operations/`: the operation scene and its presenters. `relations.ts` evaluates recorded cell rules, `layoutMorph.ts` turns verified mappings into whole-tensor motion, `diagnosis.ts` explains failed operations, and `mutation.ts` finds the cells an in-place write changed. Dedicated views cover addition, joins, linear projection, convolution, pooling, layer normalization, patch embedding, spatial grouping, and window attention. Presenters consume recorded tensors; none runs the user's algorithm.
- `tensors/`: the reusable tensor card, paged value loading, coordinate transforms, the indexed 3D volume, and the pixel view.
- `builder/`: the component canvas, toolbox, connections, inferred shape previews, and custom components.
- `inputs/`: input settings, the saved-input library, `.npy` upload, multiple forward inputs, the sample image, and sentence tokens.
- `sources/`, `weights/`: source files, Git import, dependency environments, and the checkpoint library.
- `workspace/`: the command palette, shareable links, and run comparison.
- `components/`: the settings editor, new-project dialog, and the journey container (`Walkthrough`).

Canvas cell tracing uses `journey/cellContributors.ts` to resolve one result coordinate to at most 64 structural operand uses. It validates recorded metadata and preserves exact total counts without evaluating model code. `CanvasCellProbe` reads only the selected snapshot value; the existing volume renderer highlights the subset present in its bounded geometry. Selection indices, parameters, and repeated operands remain explicit. Unsupported relationships leave the rest of the scene visible.

## How views stay honest

Each view validates what it shows against the recorded trace before offering it. A lesson name from the backend is a claim; the matching frontend module re-checks shapes, data types, and arguments, and falls back to general inspection when they do not agree. Arithmetic shown in a view explains a recorded result and never replaces it. Anything derived from a shape check rather than a run is marked as predicted and cannot open a step.

`TensorCard` gives every tensor a local selection, hover readout, one keyboard tab stop, arrow-key navigation, paging, and slice controls. **Axes** changes only the visible plane; coordinate arrays always retain the recorded tensor's axis order. `tensors/plane.ts` owns coordinate and view calculations and stride-based storage positions. Presenters supply linked highlights separately from the local selection, so inputs and operands remain inspectable even when no exact relationship is supported.

`TensorVolume` projects indexed 3D cells in both node thumbnails and a separately enlarged, rotatable dialog. The dialog's **Slice** mode uses a `Plane` to project any two axes front-on, preserving the selected coordinate and fixing every undisplayed axis explicitly. It uses the same bounded geometry and recorded value loader. Axis selection changes the view only; returning to the volume preserves its camera. Gaps carry exact omitted ranges. These are logical views, not drawings of physical memory. See `volume.ts` for sampling, displayed-axis navigation, and geometry.

Rendering is bounded everywhere: 8 × 8 or 16 × 16 cell windows, 256 values per request, 256 cells for whole-tensor motion, and constant-size rule evaluation for tensors of any size.

## Checks

```sh
npm test
npm run build
npm run format:check
```

Tests are unit tests of pure modules; there are no browser-level tests. Regenerate types from the committed contract with `npm run generate:api`. Export an updated backend contract first when Python models change; instructions are in the backend README.
