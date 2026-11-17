import { test } from 'node:test';
import assert from 'node:assert/strict';
import { countLabel, hunkHeading, statusLabel, totals } from '../src/lib/changes.js';

test('totals add up the lines of every file', () => {
  const changes = [
    { path: 'a.md', status: 'modified', added: 3, removed: 1 },
    { path: 'big.bin', status: 'modified', unreadable: true },
    { path: 'b.txt', status: 'created', added: 2, removed: 0 },
  ];
  assert.deepEqual(totals(changes), { files: 3, added: 5, removed: 1 });
  assert.deepEqual(totals(null), { files: 0, added: 0, removed: 0 });
});

test('files are labelled by what happened to them', () => {
  assert.equal(statusLabel({ status: 'created' }), 'new');
  assert.equal(statusLabel({ status: 'unchanged' }), 'changed back');
  assert.equal(statusLabel({ status: 'modified' }), '');
  assert.equal(hunkHeading({ old_start: 14, new_start: 16 }), 'Line 16');
  assert.equal(countLabel(1), '1 file changed');
  assert.equal(countLabel(4), '4 files changed');
});
