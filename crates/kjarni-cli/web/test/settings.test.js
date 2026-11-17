import { test } from 'node:test';
import assert from 'node:assert/strict';
import { describePhase, filterFacts, formatBytes, formatParams, switching } from '../src/lib/settings.js';

test('model sizes read as people say them', () => {
  assert.equal(formatParams(752), '752M');
  assert.equal(formatParams(4206), '4.2B');
  assert.equal(formatParams(27781), '28B');
  assert.equal(formatParams(null), '');
  assert.equal(formatBytes(640 * 1024 * 1024), '640 MB');
  assert.equal(formatBytes(1.25 * 1024 ** 3), '1.3 GB');
});

test('a change of model is described while it runs and when it fails', () => {
  assert.equal(switching({ state: 'downloading', model: 'qwen3.5-9b' }), true);
  assert.equal(switching({ state: 'failed', model: 'qwen3.5-9b', message: 'x' }), false);
  assert.equal(describePhase({ state: 'idle' }, null, 'qwen3.5-4b'), '');
  assert.equal(
    describePhase({ state: 'downloading', model: 'qwen3.5-9b' }, 2 * 1024 ** 3, 'qwen3.5-4b'),
    'Downloading qwen3.5-9b… 2.0 GB so far',
  );
  assert.equal(
    describePhase({ state: 'failed', model: 'qwen3.5-9b', message: 'not enough GPU memory' }, null, 'qwen3.5-4b'),
    'Could not switch to qwen3.5-9b: not enough GPU memory. Still running qwen3.5-4b.',
  );
});

test('memory is filtered by every word, in any order', () => {
  const facts = [
    { id: 1, text: 'Prefers short answers' },
    { id: 2, text: 'Works on Kjarni, a Rust library' },
  ];
  assert.deepEqual(filterFacts(facts, '  ').map((f) => f.id), [1, 2]);
  assert.deepEqual(filterFacts(facts, 'rust kjarni').map((f) => f.id), [2]);
  assert.deepEqual(filterFacts(facts, 'SHORT').map((f) => f.id), [1]);
});
