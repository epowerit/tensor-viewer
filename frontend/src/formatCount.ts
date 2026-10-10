const counts = new Intl.NumberFormat();

/**
 * A count as the reader's locale writes it (12,345), as `toLocaleString`
 * does, through one shared formatter: `toLocaleString` sets one up on every
 * call, some 40 times slower, and shapes and counts are written on every
 * render of every card.
 */
export function formatCount(value: number): string {
  return counts.format(value);
}
