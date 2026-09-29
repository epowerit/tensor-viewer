import type { components } from "./schema";

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
> & {
  input: Omit<
    Required<components["schemas"]["InputSpec"]>,
    "random_stream" | "uploaded"
  > & {
    random_stream?: "model" | "input";
    uploaded?: components["schemas"]["UploadedTensor"] | null;
  };
  capture_mode?: "values" | "shapes";
  blueprint?: Blueprint | null;
  input_name?: string;
  input_binding?: "positional" | "keyword";
  additional_inputs?: ForwardInput[];
};
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
    "operations" | "tensors" | "module_calls"
  > & {
    operations: Operation[];
    tensors: Record<string, Tensor>;
    module_calls?: ModuleCall[];
  };
};
export type ModuleCall = Required<components["schemas"]["ModuleCall"]>;
export type RunSummary = components["schemas"]["RunSummary"];
export type Operation = Omit<
  Required<components["schemas"]["Operation"]>,
  "lesson"
> & {
  lesson: Omit<
    Required<components["schemas"]["Lesson"]>,
    "mapping_rule" | "patch_size"
  > & {
    mapping_rule?: "identity" | "permutation" | "unfold" | null;
    patch_size?: number[] | null;
  };
};
export type Tensor = Omit<
  Required<components["schemas"]["TensorState"]>,
  "value_source"
> & { value_source?: "inline" | "paged" | "shape" };

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
  return response.json();
}

export const api = {
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
  templates: () => request<Template[]>("/templates"),
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
  save: (id: string, draft: Draft) =>
    request<Project>(`/projects/${id}`, "PUT", draft),
  run: (id: string) => request<Run>(`/projects/${id}/runs`, "POST"),
  runs: (id: string) => request<RunSummary[]>(`/projects/${id}/runs`),
  getRun: (id: string) => request<Run>(`/runs/${id}`),
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
    capture_mode: project.capture_mode ?? "values",
    blueprint: project.blueprint ?? null,
  };
}

// Generated Python is derived state. A refreshed shape check must not make an
// unchanged canvas look edited, or make a saved execution appear out of date.
export function draftSignature(project: Project | Draft): string {
  const draft = toDraft(project);
  draft.input = { ...draft.input, uploaded: draft.input.uploaded ?? null };
  draft.additional_inputs = draft.additional_inputs?.map((item) => ({
    ...item,
    input: { ...item.input, uploaded: item.input.uploaded ?? null },
  }));
  const snapshot = draft.blueprint
    ? {
        name: draft.name,
        input: draft.input,
        capture_mode: draft.capture_mode,
        blueprint: {
          ...draft.blueprint,
          components: draft.blueprint.components.map((c) => ({
            ...c,
            custom: c.custom ?? null,
            arguments: c.arguments ?? null,
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
