/**
 * Axes as a sentence names them: "axis 1 (experts)", "axes 1, 2 (heads,
 * tokens)". Placeholder names such as "axis 2" add nothing and are left out.
 */
export function axisPhrase(axes: number[], names: string[] = []): string {
  const label = `${axes.length === 1 ? "axis" : "axes"} ${axes.join(", ")}`;
  const named = axes.map((axis) => names[axis] ?? "");
  return named.every((name) => name && !/^axis \d+$/.test(name))
    ? `${label} (${named.join(", ")})`
    : label;
}
