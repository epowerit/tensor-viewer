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
      },
      {
        id: "flow",
        label: "Flow",
        command: "Flow table",
        detail: "Every step's shape and values, in order",
      },
      {
        id: "watch",
        label: "Watch",
        command: "Watch expressions",
        detail: "Python over the run's tensors at the playback position",
      },
      {
        id: "output",
        label: "Output",
        command: "Printed output",
        detail: "What the run printed",
      },
      {
        id: "problems",
        label: "Run notes",
        command: "Run notes",
        detail: "Errors, insights, and contract checks",
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
      },
      {
        id: "attention",
        label: "Attention",
        command: "Attention maps",
        detail: "Every attention head's weights at a glance, and one up close",
      },
      {
        id: "map",
        label: "Map",
        command: "Map of a tensor's rows",
        detail:
          "Each word's vector on its two main directions, or its path across layers",
      },
      {
        id: "lens",
        label: "Logit lens",
        command: "Logit lens",
        detail: "What each layer of a language model would predict",
      },
      {
        id: "trace",
        label: "Causal trace",
        command: "Causal trace",
        detail: "Where a change in the input matters, layer by position",
      },
    ],
  },
];

export const PANEL_TABS: TabInfo[] = PANEL_GROUPS.flatMap(
  (group) => group.tabs,
);

export const panelTab = (id: PanelTab) =>
  PANEL_TABS.find((tab) => tab.id === id)!;
