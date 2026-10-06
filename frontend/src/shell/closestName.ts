/** The edit distance between two names: inserts, deletes and swaps. */
function distance(a: string, b: string): number {
  let previous = Array.from({ length: b.length + 1 }, (_, at) => at);
  for (let i = 1; i <= a.length; i++) {
    const current = [i];
    for (let j = 1; j <= b.length; j++)
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
    previous = current;
  }
  return previous[b.length];
}

/**
 * The name a mistyped one most likely meant, as an editor suggests: the
 * nearest by edit distance, within a third of its length (at least one
 * edit), or null when none is that close.
 */
export function closestName(missing: string, names: string[]): string | null {
  const within = Math.max(1, Math.floor(missing.length / 3));
  let best: { name: string; away: number } | null = null;
  for (const name of names) {
    if (name === missing) continue;
    const away = distance(missing.toLowerCase(), name.toLowerCase());
    if (away <= within && (!best || away < best.away)) best = { name, away };
  }
  return best?.name ?? null;
}

/** The name a Python NameError says is missing, if the message is one. */
export const missingName = (message: string) =>
  /NameError: name '([A-Za-z_]\w*)' is not defined/.exec(message)?.[1] ?? null;
