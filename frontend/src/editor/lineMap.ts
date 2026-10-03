/** Above this many line pairs, an edited region is mapped by position only. */
const ALIGNED = 250_000;

/**
 * Where each 1-based line of `before` is in `after`, as an editor moves its
 * markers when text changes: lines above and below the edit keep their place
 * (shifted by the lines it added or removed), lines inside it follow their
 * text, and an edited line stays where its edit is. Null for a line removed
 * with nothing left in its place.
 */
export function lineMap(
  before: string,
  after: string,
): (line: number) => number | null {
  if (before === after) return (line) => line;
  const a = before.split("\n"),
    b = after.split("\n");
  let top = 0;
  while (top < a.length && top < b.length && a[top] === b[top]) top++;
  let bottom = 0;
  while (
    bottom < a.length - top &&
    bottom < b.length - top &&
    a[a.length - 1 - bottom] === b[b.length - 1 - bottom]
  )
    bottom++;
  const rows = a.length - top - bottom,
    columns = b.length - top - bottom;
  // Lines of the edited region that survive it, by longest common run.
  const moved = new Map<number, number>();
  if (rows && columns && rows * columns <= ALIGNED) {
    const width = columns + 1;
    const table = new Uint32Array((rows + 1) * width);
    for (let i = rows - 1; i >= 0; i--)
      for (let j = columns - 1; j >= 0; j--)
        table[i * width + j] =
          a[top + i] === b[top + j] && a[top + i].trim()
            ? table[(i + 1) * width + j + 1] + 1
            : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
    let i = 0,
      j = 0;
    while (i < rows && j < columns) {
      if (a[top + i] === b[top + j] && a[top + i].trim()) moved.set(i++, j++);
      else if (table[(i + 1) * width + j] >= table[i * width + j + 1]) i++;
      else j++;
    }
  }
  return (line) => {
    const i = line - 1;
    if (i < 0 || i >= a.length) return null;
    if (i < top) return line;
    if (i >= a.length - bottom) return line + b.length - a.length;
    const at = moved.get(i - top);
    if (at !== undefined) return top + at + 1;
    // An edited line stays where its edit is: as far past the surviving line
    // above it as it was, short of the surviving line below.
    let above = -1,
      aboveAt = -1,
      belowAt = columns;
    for (const [from, to] of moved)
      if (from < i - top && from > above) [above, aboveAt] = [from, to];
      else if (from > i - top && to < belowAt) belowAt = to;
    const to = aboveAt + (i - top - above);
    return to < belowAt
      ? top + to + 1
      : belowAt > aboveAt + 1
        ? top + belowAt
        : null;
  };
}

/** Lines moved with an edit, those removed dropped, sorted and distinct. */
export function followLines(
  lines: number[],
  before: string,
  after: string,
): number[] {
  const map = lineMap(before, after);
  return [...new Set(lines.flatMap((line) => map(line) ?? []))].sort(
    (x, y) => x - y,
  );
}
