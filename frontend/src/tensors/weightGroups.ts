/**
 * A model's weights in the order the run reads them, grouped by the layer
 * that holds them, the way the canvas groups its steps:
 * `blocks.0.attention.qkv.weight` sits in `blocks.0.attention` as
 * `qkv.weight`, and the model's own top-level weights (`token.weight`,
 * `norm.bias`) in a group with no path. A group is a run of weights read one
 * after another from one layer, so a block's first norm, its attention, its
 * second norm and its feed-forward come in the order the code runs them, and
 * the final norm comes last, apart from the embeddings. A weight read at the
 * same step as its bias comes first; weights never read go last.
 */
export function groupWeights<T extends { name: string }>(
  weights: T[],
  firstRead: (weight: T) => number | null,
): { path: string; weights: { weight: T; label: string }[] }[] {
  const order = (weight: T) => firstRead(weight) ?? Number.POSITIVE_INFINITY;
  const pathOf = (weight: T) => weight.name.split(".").slice(0, -2).join(".");
  const sorted = [...weights].sort(
    (a, b) => order(a) - order(b) || b.name.localeCompare(a.name),
  );
  const groups: { path: string; weights: { weight: T; label: string }[] }[] =
    [];
  for (const weight of sorted) {
    const path = pathOf(weight);
    const label = path ? weight.name.slice(path.length + 1) : weight.name;
    const last = groups.at(-1);
    if (last && last.path === path) last.weights.push({ weight, label });
    else groups.push({ path, weights: [{ weight, label }] });
  }
  return groups;
}
