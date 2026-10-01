import { describe, expect, it, vi } from "vitest";
import {
  createSceneClock,
  type SceneClockOptions,
  type SceneScheduler,
} from "./useSceneClock";

function setup() {
  let time = 0;
  let nextFrame = 0;
  const pending = new Map<number, (time: number) => void>();
  const scheduler: SceneScheduler = {
    now: () => time,
    request(callback) {
      pending.set(++nextFrame, callback);
      return nextFrame;
    },
    cancel: (frame) => {
      pending.delete(frame);
    },
  };
  const clock = createSceneClock(scheduler);
  const complete = vi.fn();
  let options: SceneClockOptions = {
    key: "run:op0",
    playing: true,
    speed: 1,
    duration: 2800,
    onComplete: complete,
  };
  function configure(changes: Partial<SceneClockOptions> = {}) {
    options = { ...options, ...changes };
    clock.configure(options);
  }
  function advance(milliseconds: number, renderFrame = true) {
    time += milliseconds;
    if (!renderFrame) return;
    const callbacks = [...pending.values()];
    pending.clear();
    for (const callback of callbacks) callback(time);
  }
  return { clock, configure, advance, complete, pending };
}

describe("shared operation playback clock", () => {
  it("publishes elapsed progress, completes once, and holds the final scene", () => {
    const { clock, configure, advance, complete, pending } = setup();
    const listener = vi.fn();
    const unsubscribe = clock.subscribe(listener);
    configure();
    advance(700);
    expect(clock.getSnapshot()).toBe(0.25);
    advance(9000);
    expect(clock.getSnapshot()).toBe(1);
    expect(complete).toHaveBeenCalledOnce();
    expect(pending.size).toBe(0);
    configure({ playing: false });
    configure({ playing: true });
    expect(clock.getSnapshot()).toBe(1);
    expect(complete).toHaveBeenCalledOnce();
    unsubscribe();
    listener.mockClear();
    configure({ key: "run:op1" });
    expect(listener).not.toHaveBeenCalled();
  });

  it("excludes paused time and resumes from the same point", () => {
    const { clock, configure, advance, complete } = setup();
    configure();
    advance(700);
    configure({ playing: false });
    advance(60_000);
    expect(clock.getSnapshot()).toBe(0.25);
    configure({ playing: true });
    advance(700);
    expect(clock.getSnapshot()).toBe(0.5);
    expect(complete).not.toHaveBeenCalled();
    advance(1400);
    expect(complete).toHaveBeenCalledOnce();
  });

  it("applies speed changes to remaining time, including time between frames", () => {
    const { clock, configure, advance, complete } = setup();
    configure();
    advance(700, false);
    configure({ speed: 2 });
    expect(clock.getSnapshot()).toBe(0.25);
    advance(700);
    expect(clock.getSnapshot()).toBe(0.75);
    configure({ speed: 0.5 });
    advance(1400);
    expect(clock.getSnapshot()).toBe(1);
    expect(complete).toHaveBeenCalledOnce();
  });

  it("resets only for a new operation or an explicit replay key", () => {
    const { clock, configure, advance, complete, pending } = setup();
    configure();
    advance(1400);
    configure({ key: "run:op1" });
    expect(clock.getSnapshot()).toBe(0);
    expect(pending.size).toBe(1);
    advance(2800);
    expect(complete).toHaveBeenCalledOnce();
    configure({ key: "run:op1:replay" });
    expect(clock.getSnapshot()).toBe(0);
    advance(2800);
    expect(complete).toHaveBeenCalledTimes(2);
  });

  it("cancels scheduled work on cleanup and uses the latest completion callback", () => {
    const { clock, configure, advance, complete, pending } = setup();
    const latest = vi.fn();
    configure();
    advance(700);
    clock.stop();
    expect(pending.size).toBe(0);
    advance(9000);
    expect(clock.getSnapshot()).toBe(0.25);
    configure({ onComplete: latest });
    advance(2100);
    expect(complete).not.toHaveBeenCalled();
    expect(latest).toHaveBeenCalledOnce();
  });

  it("offers a stable server snapshot without starting a server timer", () => {
    const clock = createSceneClock();
    clock.configure({
      key: "op0",
      playing: true,
      speed: 1,
      onComplete: vi.fn(),
    });
    expect(clock.getSnapshot()).toBe(0);
    expect(clock.getServerSnapshot()).toBe(0);
    clock.stop();
  });
});
