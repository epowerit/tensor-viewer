export type Command = {
  id: string;
  group: "Actions" | "Steps" | "Calls" | "Tensors" | "Projects" | "Examples";
  label: string;
  /** Secondary text, also searched. */
  detail?: string;
  shortcut?: string;
  disabled?: boolean;
  run: () => void;
};

const ORDER: Command["group"][] = [
  "Actions",
  "Steps",
  "Calls",
  "Tensors",
  "Projects",
  "Examples",
];

/**
 * Commands whose text contains every typed word. Matches at the start of the
 * label come first; otherwise the original order within each group is kept.
 */
export function filterCommands(
  commands: Command[],
  query: string,
  limit = 40,
): Command[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const scored = commands.flatMap((command, position) => {
    const label = command.label.toLowerCase();
    const text = `${label} ${command.detail?.toLowerCase() ?? ""} ${command.group.toLowerCase()}`;
    if (!words.every((word) => text.includes(word))) return [];
    const leading = words.length && label.startsWith(words[0]) ? 0 : 1;
    const inLabel = words.every((word) => label.includes(word)) ? 0 : 1;
    // Words typed next to each other count more when they stand together:
    // "pass 2" is a step's pass, not a 2 somewhere in its shape.
    const together = words
      .slice(1)
      .filter((word, i) => text.includes(`${words[i]} ${word}`)).length;
    return [
      {
        command,
        position,
        together,
        rank: words.length ? leading + inLabel : 0,
      },
    ];
  });
  scored.sort(
    (a, b) =>
      a.rank - b.rank ||
      b.together - a.together ||
      ORDER.indexOf(a.command.group) - ORDER.indexOf(b.command.group) ||
      a.position - b.position,
  );
  return scored.slice(0, limit).map((item) => item.command);
}
