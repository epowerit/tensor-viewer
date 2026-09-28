import type { components } from "./schema";

// The backend emits defaults in every response. Required makes those defaults
// explicit to UI components while the generated schema remains unmodified.
export type Draft = Omit<
  Required<components["schemas"]["ProjectDraft"]>,
  "input"
> & { input: Required<components["schemas"]["InputSpec"]> };
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
> & { lesson: Required<components["schemas"]["Lesson"]> };
export type Tensor = Required<components["schemas"]["TensorState"]>;

async function request<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
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
  create: (draft: Draft) => request<Project>("/projects", "POST", draft),
  save: (id: string, draft: Draft) =>
    request<Project>(`/projects/${id}`, "PUT", draft),
  run: (id: string) => request<Run>(`/projects/${id}/runs`, "POST"),
  runs: (id: string) => request<RunSummary[]>(`/projects/${id}/runs`),
  getRun: (id: string) => request<Run>(`/runs/${id}`),
};

export function toDraft(project: Project | Draft): Draft {
  return {
    name: project.name,
    code: project.code,
    class_name: project.class_name,
    constructor: project.constructor,
    input: project.input,
  };
}
