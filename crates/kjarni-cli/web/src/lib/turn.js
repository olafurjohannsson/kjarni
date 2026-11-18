// Messages sent while Kjarni works, and how full its prompt is.

/** `queued` without the first message equal to `text`, and whether there
 * was one. */
export function takeQueued(queued, text) {
  const i = queued.indexOf(text.trim());
  if (i < 0) return { queued, found: false };
  return { queued: [...queued.slice(0, i), ...queued.slice(i + 1)], found: true };
}

/** How full the prompt is, as a whole percentage of the budget; null when
 * unknown. */
export function contextShare(context) {
  if (!context?.budget) return null;
  return Math.min(100, Math.round((context.used / context.budget) * 100));
}

/** Tokens the way people say them: 950, 16K, 1.5K. */
export function formatTokens(n) {
  if (n < 1000) return String(n);
  const k = n / 1000;
  return `${k >= 10 || Number.isInteger(k) ? Math.round(k) : k.toFixed(1)}K`;
}

/** What the context meter says when opened. */
export function describeContext(context) {
  const share = contextShare(context);
  if (share === null) return '';
  return `${formatTokens(context.used)} of ${formatTokens(context.budget)} tokens (${share}%). Past that, Kjarni summarizes the earlier messages to make room.`;
}
