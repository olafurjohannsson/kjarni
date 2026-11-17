// Words for the settings page: model sizes, download progress and how a
// change of model is going.

/** A parameter count in millions as people say it: "752M", "4.2B". */
export function formatParams(millions) {
  if (!millions) return '';
  if (millions < 1000) return `${millions}M`;
  const billions = millions / 1000;
  return `${billions < 10 ? billions.toFixed(1) : Math.round(billions)}B`;
}

/** Bytes as "640 MB" or "1.2 GB". */
export function formatBytes(bytes) {
  const mb = bytes / (1024 * 1024);
  if (mb < 1024) return `${Math.round(mb)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

/** Whether a change of model is under way. */
export function switching(phase) {
  return phase?.state === 'downloading' || phase?.state === 'loading';
}

/** One line on how a change is going, or '' when none is. */
export function describePhase(phase, bytes, current) {
  switch (phase?.state) {
    case 'downloading':
      return `Downloading ${phase.model}… ${bytes ? `${formatBytes(bytes)} so far` : 'starting'}`;
    case 'loading':
      return `Loading ${phase.model}. New messages wait until it is ready.`;
    case 'failed':
      return `Could not switch to ${phase.model}: ${phase.message}. Still running ${current}.`;
    default:
      return '';
  }
}

/** Facts whose text has every word of `query`, ignoring case. */
export function filterFacts(facts, query) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return facts;
  return facts.filter((f) => {
    const text = f.text.toLowerCase();
    return words.every((w) => text.includes(w));
  });
}
