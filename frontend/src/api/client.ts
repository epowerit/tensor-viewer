import type { components } from "./schema";

// The backend emits defaults in every response. Required makes those defaults
// explicit to UI components while the generated schema remains unmodified.
export type Draft = Omit<
  Required<components["schemas"]["ProjectDraft"]>,
  "input" | "capture_mode" | "blueprint"
> & {
  input: Required<components["schemas"]["InputSpec"]>;
  capture_mode?: "values" | "shapes";
  blueprint?: Blueprint | null;
};
export type ComponentSpec = {
  id: string;
  kind: string;
  parameters: Record<string, number>;
};
export type Blueprint = { has_input: boolean; components: ComponentSpec[] };
export type CompositionPlan = components["schemas"]["CompositionPlan"];
export type ToolboxItem = {
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
    "operations" | "tensors"
  > & { operations: Operation[]; tensors: Record<string, Tensor> };
};
export type RunSummary = components["schemas"]["RunSummary"];
export type Operation = Omit<
  Required<components["schemas"]["Operation"]>,
  "lesson"
> & {
  lesson: Omit<Required<components["schemas"]["Lesson"]>, "mapping_rule"> & {
    mapping_rule?: "identity" | "permutation" | "unfold" | null;
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
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
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
  projects: () => request<Project[]>("/projects"),
  templates: () => request<Template[]>("/templates"),
  toolbox: () => request<ToolboxItem[]>("/toolbox"),
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
    capture_mode: project.capture_mode ?? "values",
    blueprint: project.blueprint ?? null,
  };
}
