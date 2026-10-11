import { Suspense, type ComponentProps } from "react";
import { lazyView, whenIdle } from "../lazyView";

// The step inspector (Inspect and the lesson view) loads apart from the
// workbench: it shows only on demand. Once the browser is idle it is fetched
// ahead with every step's lesson, so opening it does not wait.
const Inner = lazyView(() =>
  import("./OperationView").then((module) => {
    whenIdle(module.preloadLessons);
    return module.OperationView;
  }),
);
whenIdle(() => void Inner.preload().catch(() => undefined));

type Props = ComponentProps<typeof import("./OperationView").OperationView>;

export function OperationView(props: Props) {
  return (
    <Suspense fallback={<p className="panel-empty">Opening the step…</p>}>
      <Inner {...props} />
    </Suspense>
  );
}
