import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

/** Subscribe from the moving scene, so animation does not rerender the graph. */
export interface SceneClock {
  subscribe: (listener: () => void) => () => void;
  getSnapshot: () => number;
  getServerSnapshot: () => number;
}

export interface SceneClockOptions {
  key: string;
  playing: boolean;
  speed: number;
  onComplete: () => void;
  duration?: number;
}

export interface SceneScheduler {
  now: () => number;
  request: (callback: (time: number) => void) => number;
  cancel: (frame: number) => void;
}

function browserScheduler(): SceneScheduler | undefined {
  if (typeof requestAnimationFrame === "undefined") return undefined;
  return {
    now: () => performance.now(),
    request: (callback) => requestAnimationFrame(callback),
    cancel: (frame) => cancelAnimationFrame(frame),
  };
}

const positive = (value: number, fallback: number) =>
  Number.isFinite(value) && value > 0 ? value : fallback;

/** A single operation's elapsed time, independent of React and frame rate. */
export function createSceneClock(scheduler = browserScheduler()) {
  const listeners = new Set<() => void>();
  let key: string | undefined;
  let progress = 0;
  let playing = false;
  let speed = 1;
  let duration = 2800;
  let previousTime = 0;
  let frame: number | undefined;
  let completed = false;
  let onComplete = () => {};

  function publish(next: number) {
    if (progress === next) return;
    progress = next;
    for (const listener of listeners) listener();
  }

  function advance(time: number) {
    if (playing)
      publish(
        Math.min(
          1,
          progress + (Math.max(0, time - previousTime) * speed) / duration,
        ),
      );
    previousTime = time;
  }

  function cancel() {
    if (frame !== undefined) scheduler?.cancel(frame);
    frame = undefined;
  }

  function finish() {
    if (completed || progress < 1) return;
    completed = true;
    playing = false;
    onComplete();
  }

  function tick(time: number) {
    frame = undefined;
    if (!playing) return;
    advance(time);
    if (progress === 1) finish();
    else frame = scheduler?.request(tick);
  }

  return {
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getSnapshot: () => progress,
    getServerSnapshot: () => 0,
    configure(options: SceneClockOptions) {
      cancel();
      const time = scheduler?.now() ?? 0;
      if (key !== options.key) {
        key = options.key;
        completed = false;
        publish(0);
      } else advance(time);
      previousTime = time;
      speed = positive(options.speed, 1);
      duration = positive(options.duration ?? 2800, 2800);
      onComplete = options.onComplete;
      playing = options.playing;
      if (!playing) return;
      if (progress === 1) finish();
      else frame = scheduler?.request(tick);
    },
    /** Effect cleanup pauses elapsed time without discarding the current frame. */
    stop() {
      cancel();
      advance(scheduler?.now() ?? previousTime);
      playing = false;
    },
  };
}

const useBrowserLayoutEffect =
  typeof window === "undefined" ? useEffect : useLayoutEffect;

export function useSceneClock({
  key,
  playing,
  speed,
  onComplete,
  duration = 2800,
}: SceneClockOptions): SceneClock {
  const [clock] = useState(() => createSceneClock());
  const completion = useRef(onComplete);
  completion.current = onComplete;
  useBrowserLayoutEffect(() => {
    clock.configure({
      key,
      playing,
      speed,
      duration,
      onComplete: () => completion.current(),
    });
    return clock.stop;
  }, [clock, key, playing, speed, duration]);
  return clock;
}

export function useSceneProgress(clock: SceneClock) {
  return useSyncExternalStore(
    clock.subscribe,
    clock.getSnapshot,
    clock.getServerSnapshot,
  );
}
