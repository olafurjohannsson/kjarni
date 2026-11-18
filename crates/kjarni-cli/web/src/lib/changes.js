// The files a conversation changed, as the changes panel shows them.

/** Lines added and removed across all files, and how many files. */
export function totals(changes) {
  let added = 0;
  let removed = 0;
  for (const c of changes ?? []) {
    added += c.added ?? 0;
    removed += c.removed ?? 0;
  }
  return { files: changes?.length ?? 0, added, removed };
}

/** A word for a file's state, or '' for a plain change. */
export function statusLabel(change) {
  if (change.status === 'created') return 'new';
  if (change.status === 'deleted') return 'deleted';
  if (change.status === 'unchanged') return 'changed back';
  return '';
}

/** The line a hunk starts at, as its heading shows it. */
export function hunkHeading(hunk) {
  return `Line ${hunk.new_start}`;
}

/** "3 files changed" and the like, for the button that opens the panel. */
export function countLabel(n) {
  return n === 1 ? '1 file changed' : `${n} files changed`;
}
