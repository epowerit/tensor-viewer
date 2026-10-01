/** A place in the workspace that a link can reopen. */
export type WorkspaceLocation = {
  project?: string;
  run?: string;
  /** Journey node: an operation id, or `input-<tensor>` for a root tensor. */
  node?: string;
  cell?: number;
};

const ID = /^[A-Za-z0-9_-]{1,80}$/;

/** Read a location from a URL fragment, ignoring anything malformed. */
export function parseLocation(hash: string): WorkspaceLocation {
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const location: WorkspaceLocation = {};
  for (const key of ["project", "run", "node"] as const) {
    const value = params.get(key);
    if (value && ID.test(value)) location[key] = value;
  }
  const cell = params.get("cell");
  if (cell && /^\d{1,15}$/.test(cell)) location.cell = Number(cell);
  // A run or node is only meaningful inside its project and run.
  if (!location.project) return {};
  if (!location.run) return { project: location.project };
  if (!location.node) return { project: location.project, run: location.run };
  return location;
}

export function formatLocation(location: WorkspaceLocation): string {
  const params = new URLSearchParams();
  if (location.project) {
    params.set("project", location.project);
    if (location.run) {
      params.set("run", location.run);
      if (location.node) {
        params.set("node", location.node);
        if (location.cell) params.set("cell", String(location.cell));
      }
    }
  }
  const text = params.toString();
  return text ? `#${text}` : "";
}
