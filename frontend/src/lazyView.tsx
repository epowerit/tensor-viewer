import { lazy, useState, type ComponentType } from "react";

/**
 * A component loaded apart, like `lazy`, that renders at once when its
 * module has already arrived. `lazy` suspends on its first render even then,
 * and React holds a fallback it has shown for 300 ms before the view.
 * `preload` fetches it ahead.
 */
export function lazyView<P extends object>(
  load: () => Promise<ComponentType<P>>,
) {
  let loaded: ComponentType<P> | null = null;
  let pending: Promise<ComponentType<P>> | null = null;
  const preload = () => {
    pending ??= load().then(
      (view) => (loaded = view),
      (error: unknown) => {
        // A failed fetch is not remembered, so a later render tries again.
        pending = null;
        throw error;
      },
    );
    return pending;
  };
  const Lazy = lazy(() => preload().then((view) => ({ default: view })));
  function View(props: P) {
    // Chosen once per mount: a module that arrives later never remounts it.
    const [Shown] = useState<ComponentType<P>>(() => loaded ?? Lazy);
    return <Shown {...props} />;
  }
  return Object.assign(View, { preload });
}

/** Fetches modules once the browser is idle, so opening them does not wait. */
export function whenIdle(load: () => void) {
  if (typeof window === "undefined") return;
  if ("requestIdleCallback" in window) window.requestIdleCallback(load);
  else setTimeout(load, 2000);
}
