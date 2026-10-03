/**
 * When something happened, as a person would say it: "just now", "12 min
 * ago", "3 h ago" today, "yesterday", a weekday within the week, and the
 * date before that.
 */
export function since(then: Date, now = new Date()) {
  const seconds = (now.getTime() - then.getTime()) / 1000;
  if (Number.isNaN(seconds)) return "";
  if (seconds < 45) return "just now";
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))} min ago`;
  const day = (date: Date) =>
    new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const days = Math.round((day(now) - day(then)) / 86_400_000);
  if (days <= 0) return `${Math.round(seconds / 3600)} h ago`;
  if (days === 1) return "yesterday";
  if (days < 7) return then.toLocaleDateString([], { weekday: "long" });
  return then.toLocaleDateString([], {
    month: "short",
    day: "numeric",
    ...(then.getFullYear() !== now.getFullYear() ? { year: "numeric" } : {}),
  });
}
