import type { ComponentSpec } from "../api/client";
export function componentSources(
  components: ComponentSpec[],
  index: number,
): string[] {
  const c = components[index],
    previous = index ? components[index - 1].id : "input";
  return (
    c.sources ??
    (["add_join", "concat_join", "stack_join"].includes(c.kind)
      ? [previous, "input"]
      : [previous])
  );
}
export function hasBranches(components: ComponentSpec[]) {
  return components.some((_, i) => {
    const sources = componentSources(components, i);
    return (
      sources.length > 1 || sources[0] !== (i ? components[i - 1].id : "input")
    );
  });
}
