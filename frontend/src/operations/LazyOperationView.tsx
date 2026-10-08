import { lazy, Suspense, type ComponentProps } from "react";

// The step inspector (Inspect and the lesson view) loads apart from the
// workbench: it shows only on demand. Once the browser is idle it is fetched
// ahead, so opening it does not wait.
const load = () => import("./OperationView");
const Inner = lazy(() =>
  load().then((module) => ({ default: module.OperationView })),
);
if (typeof window !== "undefined") {
  const prefetch = () => void load().catch(() => undefined);
  if ("requestIdleCallback" in window) window.requestIdleCallback(prefetch);
  else setTimeout(prefetch, 2000);
}

type Props = ComponentProps<typeof import("./OperationView").OperationView>;

export function OperationView(props: Props) {
  return (
    <Suspense fallback={<p className="panel-empty">Opening the step…</p>}>
      <Inner {...props} />
    </Suspense>
  );
}
