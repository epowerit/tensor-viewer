import { useCallback, useEffect, useState } from "react";

const key = (projectId: string) => `tensorviewer.pins.${projectId}`;

function read(projectId: string | null): string[] {
  if (!projectId) return [];
  try {
    const stored = JSON.parse(localStorage.getItem(key(projectId)) ?? "[]");
    return Array.isArray(stored)
      ? stored.filter((name) => typeof name === "string")
      : [];
  } catch {
    return [];
  }
}

const CHANGED = "tensorviewer:pins";

function write(projectId: string, names: string[]) {
  try {
    localStorage.setItem(key(projectId), JSON.stringify(names));
  } catch {
    // Without storage, pins last until the page reloads.
  }
}

/**
 * Pins follow a tensor the code renamed (`scores` now written `attn`), so a
 * save that renames a variable keeps it pinned.
 */
export function renamePins(projectId: string, renames: Map<string, string>) {
  const pinned = read(projectId);
  if (!renames.size || !pinned.some((name) => renames.has(name))) return;
  write(projectId, [
    ...new Set(pinned.map((name) => renames.get(name) ?? name)),
  ]);
  window.dispatchEvent(new CustomEvent(CHANGED, { detail: projectId }));
}

/**
 * Tensor names pinned to the front of the shelf, remembered per project like
 * a debugger's watch list, so they stay put across runs.
 */
export function usePins(projectId: string | null) {
  const [pins, setPins] = useState(() => read(projectId));
  useEffect(() => {
    setPins(read(projectId));
    const reread = (event: Event) => {
      if ((event as CustomEvent).detail === projectId) setPins(read(projectId));
    };
    window.addEventListener(CHANGED, reread);
    return () => window.removeEventListener(CHANGED, reread);
  }, [projectId]);
  const toggle = useCallback(
    (name: string) => {
      setPins((current) => {
        const next = current.includes(name)
          ? current.filter((pinned) => pinned !== name)
          : [...current, name];
        if (projectId) write(projectId, next);
        return next;
      });
    },
    [projectId],
  );
  return [new Set(pins), toggle] as const;
}
