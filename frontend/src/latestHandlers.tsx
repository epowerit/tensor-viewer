import {
  memo,
  useRef,
  type ComponentType,
  type FunctionComponent,
} from "react";

/**
 * A memoized component whose event handlers (props named `on…`) keep one
 * identity and call the latest one passed. A parent rendering on every
 * keystroke with inline handlers then leaves it alone unless its data
 * changes, and it never calls a handler from an earlier render. A handler
 * that comes or goes still changes the props: its presence can show a
 * control.
 */
export function withLatestHandlers<P extends object>(
  Component: ComponentType<P>,
): FunctionComponent<P> {
  const Inner = memo(Component) as unknown as ComponentType<P>;
  function Latest(props: P) {
    const latest = useRef(props);
    latest.current = props;
    const stable = useRef(new Map<string, (...args: unknown[]) => unknown>());
    const passed = { ...props } as Record<string, unknown>;
    for (const [name, value] of Object.entries(passed)) {
      if (!/^on[A-Z]/.test(name) || typeof value !== "function") continue;
      let call = stable.current.get(name);
      if (!call) {
        call = (...args) =>
          (
            (latest.current as Record<string, unknown>)[name] as (
              ...args: unknown[]
            ) => unknown
          )?.(...args);
        stable.current.set(name, call);
      }
      passed[name] = call;
    }
    return <Inner {...(passed as P)} />;
  }
  Latest.displayName = `Latest(${Component.displayName ?? Component.name})`;
  return Latest;
}
