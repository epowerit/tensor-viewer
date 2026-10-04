import type { components } from "./schema";

export type NormalizationStatistics =
  components["schemas"]["LayerNormalizationStatistics"];
export type ReductionStatistics = components["schemas"]["ReductionStatistics"];
export type SoftmaxStatistics = components["schemas"]["SoftmaxStatistics"];

// The backend emits defaults in every response. Required makes those defaults
// explicit to UI components while the generated schema remains unmodified.
export type Draft = Omit<
  Required<components["schemas"]["ProjectDraft"]>,
  | "input"
  | "capture_mode"
  | "blueprint"
  | "input_name"
  | "input_binding"
  | "additional_inputs"
  | "weights"
  | "files"
  | "entry_path"
  | "import_root"
  | "repository"
  | "environment"
  | "script"
> & {
  input: Omit<
    Required<components["schemas"]["InputSpec"]>,
    "random_stream" | "uploaded" | "text" | "edits" | "precision" | "knockout"
  > & {
    /** The sentence behind a "text" input. */
    text?: string | null;
    /** Cells set after the input is made: a what-if run's change. */
    edits?: components["schemas"]["InputEdit"][];
    /** The dtype a what-if run computed in, when not the input's. */
    precision?: Precision | null;
    /** The step result a what-if run replaced as it was made. */
    knockout?: components["schemas"]["Knockout"] | null;
    random_stream?: "model" | "input";
    uploaded?: components["schemas"]["UploadedTensor"] | null;
  };
  capture_mode?: "values" | "shapes";
  blueprint?: Blueprint | null;
  script?: string | null;
  input_name?: string;
  input_binding?: "positional" | "keyword";
  additional_inputs?: ForwardInput[];
  weights?: SavedWeights | null;
  files?: Record<string, string>;
  entry_path?: string;
  import_root?: string;
  repository?: Record<string, string> | null;
  environment?: string | null;
};
export type SavedWeights = components["schemas"]["SavedWeights"];
export type WeightCheck = Required<components["schemas"]["WeightCheck"]>;
export type ForwardInput = {
  name: string;
  binding: "positional" | "keyword";
  input: Draft["input"];
};
export type InputFixtureDraft = {
  name: string;
  input: Draft["input"];
  capture_mode: "values" | "shapes";
};
export type InputFixture = InputFixtureDraft & {
  id: string;
  created_at: string;
};
export type ComponentSpec = {
  id: string;
  kind: string;
  parameters: Record<string, number>;
  custom?: CustomComponent | null;
  arguments?: Record<string, unknown> | null;
  sources?: string[] | null;
};
export type CustomComponentDraft = Required<
  components["schemas"]["CustomComponentDraft"]
>;
export type CustomComponent = CustomComponentDraft & {
  id: string;
  created_at: string;
};
export type Blueprint = { has_input: boolean; components: ComponentSpec[] };
export type CompositionPlan = components["schemas"]["CompositionPlan"];
export type ToolboxItem = {
  custom?: CustomComponent;
  kind: string;
  title: string;
  group: string;
  description: string;
  parameters: {
    key: string;
    label: string;
    default: number;
    min: number;
    max: number;
  }[];
};
export type Project = Draft &
  Pick<components["schemas"]["Project"], "id" | "created_at" | "updated_at">;
export type Template = Omit<components["schemas"]["Template"], "project"> & {
  project: Draft;
};
export type Run = Omit<components["schemas"]["Run"], "project" | "trace"> & {
  project: Draft;
  trace: Omit<
    Required<components["schemas"]["Trace"]>,
    | "operations"
    | "tensors"
    | "module_calls"
    | "weight_check"
    | "warnings"
    | "runtime"
    | "learn_curve"
  > & {
    operations: Operation[];
    tensors: Record<string, Tensor>;
    module_calls?: ModuleCall[];
    weight_check?: WeightCheck | null;
    warnings?: string[];
    runtime?: Record<string, string>;
    /** After training steps: the value they aimed at, before each and after. */
    learn_curve?: (number | null)[] | null;
  };
};
export type ModuleCall = Required<components["schemas"]["ModuleCall"]>;
export type RunSummary = components["schemas"]["RunSummary"];
export type InputEdit = components["schemas"]["InputEdit"];
/** One training step on every weight, toward raising or lowering a value. */
export type LearnStep = components["schemas"]["LearnStep"];
/** One step's result replaced as the run makes it: zero, its mean, or a patch. */
export type Knockout = Omit<
  components["schemas"]["Knockout"],
  "patch_path" | "output"
> & {
  /** Which of the step's results; the first when left out. */
  output?: number;
};
export type KnockoutSweep = Omit<
  components["schemas"]["KnockoutSweep"],
  "patch_path"
>;
export type SweepResult = components["schemas"]["SweepResult"];
/** A dtype a what-if run can compute in. */
export type Precision = "bfloat16" | "float16" | "float64";
export type Sensitivity = components["schemas"]["Sensitivity"];
export type GradientFlow = components["schemas"]["GradientFlow"];
export type Evaluation = components["schemas"]["Evaluation"];
export type WatchSeries = components["schemas"]["WatchSeries"];
export type WeightReport = components["schemas"]["WeightReport"];
export type WeightSpectrum = components["schemas"]["WeightSpectrum"];
/** What-if runs' ids: they live a while on the backend, in no history. */
export const isWhatIf = (runId: string | null | undefined) =>
  !!runId?.startsWith("what-if-");
export type LatestRun = components["schemas"]["LatestRun"];
export type LoopStep = components["schemas"]["LoopStep"];
export type Operation = Omit<
  Required<components["schemas"]["Operation"]>,
  "lesson" | "mutations" | "loops"
> & {
  mutations?: components["schemas"]["TensorMutation"][];
  /** Enclosing loops, outermost first; runs saved before loops have none. */
  loops?: LoopStep[];
  lesson: Omit<
    Required<components["schemas"]["Lesson"]>,
    "mapping_rule" | "patch_size" | "relation"
  > & {
    /** Absent on runs recorded before cell rules existed. */
    relation?: Record<string, unknown> | null;
    mapping_rule?: "identity" | "permutation" | "unfold" | "roll" | null;
    patch_size?: number[] | null;
  };
};
export type Histogram = components["schemas"]["Histogram"];
export type Tensor = Omit<
  Required<components["schemas"]["TensorState"]>,
  "value_source" | "histogram"
> & {
  value_source?: "inline" | "paged" | "shape";
  /** How the values are spread; runs recorded before histograms have none. */
  histogram?: Histogram | null;
};

async function request<T>(
  path: string,
  method = "GET",
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    method,
    headers: body
      ? {
          "Content-Type":
            body instanceof Blob
              ? "application/octet-stream"
              : "application/json",
        }
      : undefined,
    body: body instanceof Blob ? body : body ? JSON.stringify(body) : undefined,
    signal,
  });
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    const message =
      typeof result.detail === "string"
        ? result.detail
        : Array.isArray(result.detail)
          ? result.detail.map((d: { msg: string }) => d.msg).join(" ")
          : `Request failed (${response.status})`;
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

export type SourceImport = components["schemas"]["SourceImport"];
/** A ready-to-run project read from code alone, with anything it could not use. */
export type ReadProject = Omit<
  components["schemas"]["ReadProject"],
  "draft"
> & {
  draft: Draft;
};
export type LibraryEntry = components["schemas"]["LibraryEntry"];
export type RuntimeEnvironment = components["schemas"]["RuntimeEnvironment"];
export const api = {
  importGit: (repository: string, revision: string, subdirectory: string) =>
    request<SourceImport>("/sources/git", "POST", {
      repository,
      revision,
      subdirectory,
    }),
  environments: (signal?: AbortSignal) =>
    request<RuntimeEnvironment[]>("/environments", "GET", undefined, signal),
  setupEnvironment: (requirements: string[]) =>
    request<RuntimeEnvironment>("/environments", "POST", { requirements }),
  weights: (signal?: AbortSignal) =>
    request<SavedWeights[]>("/weights", "GET", undefined, signal),
  uploadWeights: (file: File, name: string, signal?: AbortSignal) =>
    request<SavedWeights>(
      `/weights/upload?${new URLSearchParams({ name, file_name: file.name })}`,
      "POST",
      file,
      signal,
    ),
  checkWeights: (draft: Draft, signal?: AbortSignal) =>
    request<WeightCheck>("/weights/check", "POST", draft, signal),
  uploadInput: (file: File, name: string, signal?: AbortSignal) =>
    request<InputFixture>(
      `/input-fixtures/upload?${new URLSearchParams({ name, file_name: file.name })}`,
      "POST",
      file,
      signal,
    ),
  inputFixtures: (signal?: AbortSignal) =>
    request<InputFixture[]>("/input-fixtures", "GET", undefined, signal),
  saveInputFixture: (fixture: InputFixtureDraft, signal?: AbortSignal) =>
    request<InputFixture>("/input-fixtures", "POST", fixture, signal),
  projects: () => request<Project[]>("/projects"),
  latestRuns: () => request<LatestRun[]>("/latest-runs"),
  templates: () => request<Template[]>("/templates"),
  // Pasted code, uploaded files, and library projects all become projects here.
  readSource: (code: string, name?: string, model?: string) =>
    request<ReadProject>("/sources/read", "POST", {
      code,
      name: name || null,
      model: model || null,
    }),
  library: () => request<LibraryEntry[]>("/library"),
  installLibrary: () => request<Project[]>("/library/install", "POST"),
  toolbox: () => request<ToolboxItem[]>("/toolbox"),
  saveComponent: (component: CustomComponentDraft) =>
    request<CustomComponent>("/components", "POST", component),
  checkComposition: (draft: Draft, signal?: AbortSignal) =>
    request<CompositionPlan>(
      "/compose/check",
      "POST",
      {
        blueprint: draft.blueprint,
        input: draft.input,
        capture_mode: draft.capture_mode ?? "values",
      },
      signal,
    ),
  compose: (draft: Draft, signal?: AbortSignal) =>
    request<CompositionPlan>(
      "/compose",
      "POST",
      {
        blueprint: draft.blueprint,
        input: draft.input,
        capture_mode: draft.capture_mode ?? "values",
      },
      signal,
    ),
  create: (draft: Draft) => request<Project>("/projects", "POST", draft),
  /** Removes a project with its runs; it cannot be undone server-side. */
  deleteProject: (id: string) => request<void>(`/projects/${id}`, "DELETE"),
  deleteRun: (id: string) => request<void>(`/runs/${id}`, "DELETE"),
  /** Clears a project's history of the runs recorded before one of them. */
  deleteRunsBefore: (projectId: string, runId: string) =>
    request<{ deleted: number }>(
      `/projects/${projectId}/runs?before=${encodeURIComponent(runId)}`,
      "DELETE",
    ),
  save: (id: string, draft: Draft) =>
    request<Project>(`/projects/${id}`, "PUT", draft),
  run: (id: string) =>
    request<Run>(`/projects/${id}/runs`, "POST").then(scriptRun),
  runs: (id: string) => request<RunSummary[]>(`/projects/${id}/runs`),
  getRun: (id: string) => request<Run>(`/runs/${id}`).then(scriptRun),
  /**
   * The run's code again with some input cells set. The backend keeps the
   * result for a while, outside the project's history.
   */
  whatIf: (
    runId: string,
    edits: InputEdit[],
    precision: Precision | null,
    learn: LearnStep | null = null,
    knockout: Knockout | null = null,
  ) =>
    request<Run>(`/runs/${runId}/what-if`, "POST", {
      edits,
      precision,
      learn,
      knockout,
    }).then(scriptRun),
  /**
   * Each slice of one step's result knocked out in turn, and how far the
   * model's output moved each time. Nothing is saved.
   */
  knockoutSweep: (runId: string, sweep: KnockoutSweep) =>
    request<SweepResult>(`/runs/${runId}/knockout-sweep`, "POST", sweep),
  /** Every weight's norm and singular values. */
  weightSpectra: (
    runId: string,
    against: string | null = null,
    signal?: AbortSignal,
  ) =>
    request<WeightReport>(
      `/runs/${runId}/weights`,
      "POST",
      { against },
      signal,
    ),
  /** A watch expression as one number at each step its names change. */
  evaluateSeries: (runId: string, expression: string, signal?: AbortSignal) =>
    request<WatchSeries>(
      `/runs/${runId}/evaluate-series`,
      "POST",
      { expression },
      signal,
    ),
  /** A watch expression over the run's named tensors at a step. */
  evaluate: (
    runId: string,
    expression: string,
    at: string | null,
    signal?: AbortSignal,
  ) =>
    request<Evaluation>(
      `/runs/${runId}/evaluate`,
      "POST",
      { expression, at },
      signal,
    ),
  /** The gradient's size at every tensor, for one cell or a whole sum. */
  gradients: (runId: string, tensorId: string, index: number | null) =>
    request<GradientFlow>(`/runs/${runId}/gradients`, "POST", {
      tensor_id: tensorId,
      index,
    }),
  /** ∂ one result cell / ∂ each input cell, from one backward pass. */
  sensitivity: (runId: string, tensorId: string, index: number) =>
    request<Sensitivity>(`/runs/${runId}/sensitivity`, "POST", {
      tensor_id: tensorId,
      index,
    }),
  /** A shapes-only dry run of a draft. Nothing is saved on the server. */
  shapeCheck: (draft: Draft, signal?: AbortSignal) =>
    request<Pick<Run, "project" | "trace">>(
      "/shape-check",
      "POST",
      draft,
      signal,
    ).then((result) =>
      scriptRun({
        ...result,
        id: "shape-check",
        project_id: "",
        created_at: new Date().toISOString(),
      } as Run),
    ),
  normalizationStatistics: (
    run: string,
    operation: string,
    group: number,
    signal?: AbortSignal,
  ) =>
    request<NormalizationStatistics>(
      `/runs/${run}/operations/${operation}/normalization?group=${group}`,
      "GET",
      undefined,
      signal,
    ),
  reductionStatistics: (
    run: string,
    operation: string,
    output: number,
    signal?: AbortSignal,
  ) =>
    request<ReductionStatistics>(
      `/runs/${run}/operations/${operation}/reduction?output_index=${output}`,
      "GET",
      undefined,
      signal,
    ),
  softmaxStatistics: (
    run: string,
    operation: string,
    group: number,
    signal?: AbortSignal,
  ) =>
    request<SoftmaxStatistics>(
      `/runs/${run}/operations/${operation}/softmax?group=${group}`,
      "GET",
      undefined,
      signal,
    ),
  tensorValues: (
    run: string,
    tensor: string,
    indices: number[],
    signal?: AbortSignal,
  ) =>
    request<{ indices: number[]; values: (number | string)[] }>(
      `/runs/${run}/tensors/${tensor}/values?indices=${indices.join(",")}`,
      "GET",
      undefined,
      signal,
    ),
  /** Change summaries for (tensor, earlier tensor) pairs; null where unknown. */
  compareRun: (
    run: string,
    otherRun: string,
    pairs: [string, string][],
    signal?: AbortSignal,
  ) =>
    request<{ results: (ChangeSummary | null)[] }>(
      `/runs/${run}/compare`,
      "POST",
      { other_run: otherRun, pairs },
      signal,
    ),
  /**
   * Whole-tensor questions answered by the backend, for tensors whose values
   * stay in a snapshot: a value search, the extremes, plane margins, and
   * totals over a rectangle.
   */
  tensorQuery: <T>(
    run: string,
    tensor: string,
    question:
      | "search"
      | "landmarks"
      | "margins"
      | "region"
      | "axis"
      | "thumbnails"
      | "sums",
    params: Record<string, string | number | boolean>,
    signal?: AbortSignal,
  ) =>
    request<T>(
      `/runs/${run}/tensors/${tensor}/${question}?${new URLSearchParams(
        Object.entries(params).map(([key, value]) => [key, String(value)]),
      )}`,
      "GET",
      undefined,
      signal,
    ),
};

/** How a tensor changed since an earlier run; low and high bound the change. */
export type ChangeSummary = {
  changed: number;
  compared: number;
  low: number | string;
  high: number | string;
  /** The largest and mean |difference| over finite pairs. */
  max_abs?: number | string;
  mean_abs?: number | string;
  /** torch.allclose with its default tolerances. */
  allclose?: boolean;
};

export type SearchResult = {
  count: number;
  indices: number[];
  truncated: boolean;
};
export type RegionResult = {
  count: number;
  sum: number | string | null;
  mean: number | string | null;
  std: number | string | null;
  min: number | string | null;
  max: number | string | null;
  broken: number;
  values: (number | string)[] | null;
};

export function toDraft(project: Project | Draft): Draft {
  return {
    name: project.name,
    code: project.code,
    class_name: project.class_name,
    constructor: project.constructor,
    input: project.input,
    input_name: project.input_name ?? "x",
    input_binding: project.input_binding ?? "positional",
    additional_inputs: project.additional_inputs ?? [],
    weights: project.weights ?? null,
    files: project.files ?? {},
    entry_path: project.entry_path ?? "model.py",
    import_root: project.import_root ?? ".",
    repository: project.repository ?? null,
    environment: project.environment ?? null,
    capture_mode: project.capture_mode ?? "values",
    blueprint: project.blueprint ?? null,
    script: project.script ?? null,
  };
}

/**
 * A console run executes a generated wrapper module. Present it in the
 * coordinates of the script the user wrote, so every view shows their lines.
 */
export function scriptRun(run: Run): Run {
  const script = run.project.script;
  if (script == null) return run;
  const offset =
    run.project.code
      .split("\n")
      .findIndex((line) => line.startsWith("    def forward(")) + 1;
  if (!offset) return run;
  const count = script.split("\n").length;
  const line = (value: number) => Math.max(1, Math.min(count, value - offset));
  return {
    ...run,
    project: { ...run.project, code: script },
    trace: {
      ...run.trace,
      error:
        run.trace.error?.line != null && !run.trace.error.file
          ? {
              ...run.trace.error,
              line: line(run.trace.error.line),
              // Python names the wrapper's line in its message too.
              message: run.trace.error.message.replace(
                /\(<tensorviewer-project>, line (\d+)\)/g,
                (_, at: string) => `(line ${line(Number(at))})`,
              ),
            }
          : run.trace.error,
      operations: run.trace.operations.map((op) => ({
        ...op,
        source:
          op.source && !op.source.file
            ? { ...op.source, line: line(op.source.line) }
            : op.source,
        loops: op.loops?.map((step) =>
          step.file ? step : { ...step, line: line(step.line) },
        ),
      })),
    },
  };
}

// Generated Python is derived state. A refreshed shape check must not make an
// unchanged canvas look edited, or make a saved execution appear out of date.
export function draftSignature(project: Project | Draft): string {
  const draft = toDraft(project);
  // A what-if run's cell edits, precision, and knockout are not a change to
  // the project.
  const normal = ({
    edits: _edits,
    precision: _precision,
    knockout: _knockout,
    ...input
  }: Draft["input"]) => ({
    ...input,
    uploaded: input.uploaded ?? null,
    text: input.text ?? null,
  });
  draft.input = normal(draft.input);
  draft.additional_inputs = draft.additional_inputs?.map((item) => ({
    ...item,
    input: normal(item.input),
  }));
  const snapshot =
    draft.script != null
      ? // Generated wrapper code is derived from the script.
        { ...draft, code: "", class_name: "", constructor: {} }
      : draft.blueprint
        ? {
            name: draft.name,
            input: draft.input,
            capture_mode: draft.capture_mode,
            weights: draft.weights,
            blueprint: {
              ...draft.blueprint,
              components: draft.blueprint.components.map((c) => ({
                ...c,
                custom: c.custom ?? null,
                arguments: c.arguments ?? null,
                sources: c.sources ?? null,
              })),
            },
          }
        : draft;
  return JSON.stringify(snapshot, (_key, value) =>
    value && typeof value === "object" && !Array.isArray(value)
      ? Object.fromEntries(
          Object.keys(value)
            .sort()
            .map((key) => [key, value[key]]),
        )
      : value,
  );
}
