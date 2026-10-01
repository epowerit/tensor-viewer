/** Open optional settings around an invalid field before moving keyboard focus. */
export function revealField(field: HTMLElement | null) {
  if (!field) return;
  let parent: HTMLElement | null = field.parentElement;
  while (parent) {
    if (parent instanceof HTMLDetailsElement) parent.open = true;
    parent = parent.parentElement;
  }
  field.focus({ preventScroll: true });
  field.scrollIntoView({ block: "nearest" });
}
