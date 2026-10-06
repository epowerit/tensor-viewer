export type PanelTab =
  | "problems"
  | "variables"
  | "flow"
  | "watch"
  | "weights"
  | "lens"
  | "trace"
  | "map"
  | "attention"
  | "output";

type TabInfo = {
  id: PanelTab;
  /** The tab's name in the panel. */
  label: string;
  /** Its name in the command palette, where it stands alone. */
  command: string;
  detail: string;
  /**
   * How the tab opens, as the canvas opens a step: a plain title, one
   * sentence on what it shows, and short hints on how to read it.
   */
  intro: { title: string; text: string; read?: string[] };
};

/**
 * The panel's tabs in two groups, each in the order of the work: what the
 * run recorded, from its tensors to what it printed and what it found; then
 * the model, from its weights to how its layers read the input.
 */
export const PANEL_GROUPS: { name: string; tabs: TabInfo[] }[] = [
  {
    name: "Run",
    tabs: [
      {
        id: "variables",
        label: "Tensors",
        command: "Tensor shelf",
        detail: "Every named tensor at the playback position",
        intro: {
          title: "Tensors at this step",
          text: "Every named tensor as it stands at the step playback is on: its shape, type, values, and the memory they take.",
          read: [
            "unlit cards are not computed yet",
            "a strip shows each state of a name written more than once",
            "point at a card to find it on the canvas",
          ],
        },
      },
      {
        id: "flow",
        label: "Flow",
        command: "Flow table",
        detail: "Every step's shape and values, in order",
        intro: {
          title: "Every step, in order",
          text: "One row per operation the run recorded: what it made and from what, its shape and values, how long it took, and its line.",
          read: [
            "point at a row to trace it on the canvas",
            "click a row to go there",
          ],
        },
      },
      {
        id: "watch",
        label: "Watch",
        command: "Watch expressions",
        detail: "Python over the run's tensors at the playback position",
        intro: {
          title: "Your own expressions",
          text: "Python over the run's tensors, worked out again at the step playback is on, like a debugger's watch list.",
        },
      },
      {
        id: "output",
        label: "Output",
        command: "Printed output",
        detail: "What the run printed",
        intro: {
          title: "What the code printed",
          text: "Everything print() wrote while the run was recorded, then the run's totals.",
        },
      },
      {
        id: "problems",
        label: "Run notes",
        command: "Run notes",
        detail: "Errors, insights, and contract checks",
        intro: {
          title: "What the run found",
          text: "Errors, likely mistakes, and broken contracts, each linked to its step and line.",
          read: ["click a note to open its step"],
        },
      },
    ],
  },
  {
    name: "Model",
    tabs: [
      {
        id: "weights",
        label: "Weights",
        command: "Weights and their spectra",
        detail: "Every weight's norm, singular values, condition and rank",
        intro: {
          title: "The model's weights",
          text: "Each layer's learned weights, in the order the model reads them: how big each is, and how many independent directions a matrix really uses.",
          read: [
            "‖W‖ overall size",
            "σ max its largest stretch",
            "Condition how unevenly it stretches",
            "Directions used how many it really uses, of the most it could",
            "marked rows are read by the step playback is on",
          ],
        },
      },
      {
        id: "attention",
        label: "Attention",
        command: "Attention maps",
        detail: "Every attention head's weights at a glance, and one up close",
        intro: {
          title: "Where each word looks",
          text: "Each small square is one attention head: a row is a word reading, and its columns are the words it reads from.",
          read: [
            "brighter means more attention",
            "choose a head to enlarge it",
            "click a word to follow it in every tab",
          ],
        },
      },
      {
        id: "map",
        label: "Map",
        command: "Map of a tensor's rows",
        detail:
          "Each word's vector on its two main directions, or its path across layers",
        intro: {
          title: "Words as points",
          text: "Each word's vector, flattened onto the two directions it varies most. Words close together, the model treats alike.",
          read: [
            "Similarity compares every pair of words",
            "Across layers draws each word's path through the model",
          ],
        },
      },
      {
        id: "lens",
        label: "Logit lens",
        command: "Logit lens",
        detail: "What each layer of a language model would predict",
        intro: {
          title: "What each layer would predict",
          text: "Each row reads one layer as if it were the last: the word the model would predict next, after each word of the input.",
          read: [
            "brighter means surer",
            "bright text already agrees with the final prediction",
            "the marked row holds the step playback is on",
          ],
        },
      },
      {
        id: "trace",
        label: "Causal trace",
        command: "Causal trace",
        detail: "Where a change in the input matters, layer by position",
        intro: {
          title: "Where a change matters",
          text: "Copies one layer's state at one word from the other run (the run before, or a what-if's recorded run) into this one, and measures how much of the other run's result comes back.",
          read: [
            "100% means that one spot carries the whole difference",
            "≠ marks a word that changed",
          ],
        },
      },
    ],
  },
];

export const PANEL_TABS: TabInfo[] = PANEL_GROUPS.flatMap(
  (group) => group.tabs,
);

export const panelTab = (id: PanelTab) =>
  PANEL_TABS.find((tab) => tab.id === id)!;
