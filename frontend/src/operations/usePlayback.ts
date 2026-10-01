import { useEffect, useLayoutEffect, useRef, useState } from "react";

export function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return reduced;
}

/** A cancellable 0 → 1 timeline that restarts whenever `resetKey` changes. */
export function usePlayback(
  duration: number,
  resetKey: string,
  autoplay = false,
) {
  const reduced = useReducedMotion();
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const current = useRef(0);
  current.current = progress;
  useLayoutEffect(() => {
    // With reduced motion, show the result rather than a frozen start.
    setProgress(autoplay && reduced ? 1 : 0);
    setPlaying(autoplay && !reduced);
  }, [resetKey]);
  useEffect(() => {
    if (!playing || reduced) return;
    let frame = 0;
    let start: number | null = null;
    const from = current.current;
    const tick = (time: number) => {
      if (start === null) start = time;
      const next = Math.min(1, from + (time - start) / duration);
      setProgress(next);
      if (next < 1) frame = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, reduced, duration]);
  useEffect(() => {
    const pause = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", pause);
    return () => document.removeEventListener("visibilitychange", pause);
  }, []);
  useEffect(() => {
    if (reduced) setPlaying(false);
  }, [reduced]);
  return {
    progress,
    playing,
    reduced,
    seek(value: number) {
      setPlaying(false);
      setProgress(Math.max(0, Math.min(1, value)));
    },
    toggle() {
      if (playing) setPlaying(false);
      else {
        if (current.current >= 1) setProgress(0);
        setPlaying(true);
      }
    },
  };
}
