import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  applyEvent,
  callDetail,
  fromHistory,
  newTurnState,
  splitReasoning,
  summarize,
  visibleText,
} from '../src/lib/transcript.js';

test('reasoning splits from the answer, opened or not', () => {
  assert.deepEqual(splitReasoning('<think>\nhmm\n</think>\n\nHello.'), {
    reasoning: 'hmm',
    text: 'Hello.',
    thinking: false,
  });
  assert.deepEqual(splitReasoning('hmm</think>Hi'), { reasoning: 'hmm', text: 'Hi', thinking: false });
  assert.deepEqual(splitReasoning('Just text'), { reasoning: '', text: 'Just text', thinking: false });
  // Still thinking, with the close tag half arrived.
  assert.deepEqual(splitReasoning('<think>\nweighing it</thi'), {
    reasoning: 'weighing it',
    text: '',
    thinking: true,
  });
});

test('tool calls are cut from the visible text, even half arrived', () => {
  assert.equal(visibleText('Let me look.\n<tool_call>\n<function=read_file>'), 'Let me look.');
  assert.equal(visibleText('Let me look.\n<tool_ca'), 'Let me look.');
  assert.equal(visibleText('a < b'), 'a < b');
});

test('summaries match the daemon: first line, 80 characters', () => {
  assert.equal(summarize('read_file', { path: 'notes.md' }), 'read_file notes.md');
  assert.equal(summarize('write_file', { path: 'a.txt', content: 'x' }), 'write_file a.txt');
  assert.equal(summarize('run_command', { command: 'ls\npwd' }), 'run_command ls…');
  assert.equal(summarize('run_command', { command: 'x'.repeat(90) }), `run_command ${'x'.repeat(80)}…`);
  assert.equal(summarize('recall', {}), 'recall');
  // The file an edit changes, not the text it replaces.
  assert.equal(
    summarize('edit_file', { old_text: 'a', new_text: 'b', path: 'notes.md' }),
    'edit_file notes.md',
  );
});

test('paths in the working folder are shown relative to it', () => {
  const workdir = '/home/ada/kjarni';
  assert.equal(callDetail({ path: '/home/ada/kjarni/notes/today.md' }, workdir), 'notes/today.md');
  assert.equal(callDetail({ path: '/home/ada/kjarni-old/x.md' }, workdir), '/home/ada/kjarni-old/x.md');
  assert.equal(callDetail({ path: 'C:\\Users\\ada\\kjarni\\a.txt' }, 'C:\\Users\\ada\\kjarni'), 'a.txt');
  // Shortened only after the folder is taken off.
  const deep = '/home/ada/kjarni/' + 'd/'.repeat(30) + 'f.md';
  assert.equal(callDetail({ path: deep }, workdir), 'd/'.repeat(30) + 'f.md');
});

test('saved messages pair each call with its result', () => {
  const blocks = fromHistory([
    { role: 'user', text: 'What do bananas cost?' },
    {
      role: 'assistant',
      text: '',
      reasoning: 'check the files',
      tool_calls: [{ name: 'search_files', arguments: { query: 'bananas' } }],
    },
    { role: 'tool', text: 'prices.txt:1: bananas 120 kr' },
    {
      role: 'assistant',
      text: 'Delete it?',
      reasoning: null,
      tool_calls: [{ name: 'run_command', arguments: { command: 'rm prices.txt' } }],
    },
    { role: 'tool', text: 'The user declined this call, so it did not run.' },
    { role: 'assistant', text: 'Bananas cost 120 kr.', reasoning: null, tool_calls: [] },
  ]);
  assert.deepEqual(
    blocks.map((b) => [b.type, b.text ?? b.summary, b.status ?? null]),
    [
      ['user', 'What do bananas cost?', null],
      ['assistant', '', null],
      ['tool', 'search_files bananas', 'done'],
      ['assistant', 'Delete it?', null],
      ['tool', 'run_command rm prices.txt', 'denied'],
      ['assistant', 'Bananas cost 120 kr.', null],
    ],
  );
  assert.equal(blocks[2].output, 'prices.txt:1: bananas 120 kr');
});

test('a live turn: text, a call that waits for approval, and its result', () => {
  const s = newTurnState([]);
  const events = [
    { seq: 1, type: 'user', text: 'Write a note' },
    { seq: 2, type: 'text', delta: '<think>\nok' },
    { seq: 3, type: 'text', delta: '</think>\n\nWriting it.\n<tool_' },
    { seq: 4, type: 'text', delta: 'call>\n<function=write_file>' },
    {
      seq: 5,
      type: 'tool_call',
      call: 1,
      name: 'write_file',
      summary: 'write_file note.txt',
      access: 'write',
      arguments: { path: 'note.txt' },
    },
    { seq: 6, type: 'approval', approval: 6, call: 1, summary: 'write_file note.txt', access: 'write' },
  ];
  for (const e of events) applyEvent(s, e);
  assert.equal(s.blocks.length, 3);
  assert.deepEqual(
    { reasoning: s.blocks[1].reasoning, text: s.blocks[1].text, thinking: s.blocks[1].thinking },
    { reasoning: 'ok', text: 'Writing it.', thinking: false },
  );
  assert.equal(s.blocks[2].status, 'waiting');
  assert.equal(s.blocks[2].approval, 6);

  applyEvent(s, { seq: 7, type: 'approved', approval: 6, decision: 'once' });
  assert.equal(s.blocks[2].status, 'running');
  applyEvent(s, { seq: 8, type: 'tool_result', call: 1, ok: true, denied: false, output: 'Wrote 5 bytes' });
  assert.equal(s.blocks[2].status, 'done');

  // The next reply starts a new text block after the call.
  applyEvent(s, { seq: 9, type: 'text', delta: '<think>\n</think>\n\nDone.' });
  applyEvent(s, { seq: 10, type: 'done', text: 'Done.', steps: 2, tool_calls: 1, stopped_early: false });
  assert.equal(s.blocks.length, 4);
  assert.equal(s.blocks[3].text, 'Done.');
  assert.equal(s.ended, 'done');
  assert.equal(s.lastSeq, 10);
});

test('a denial and a stop end their calls', () => {
  const s = newTurnState([]);
  applyEvent(s, { seq: 1, type: 'tool_call', call: 1, name: 'run_command', summary: 'run_command ls', access: 'execute', arguments: {} });
  applyEvent(s, { seq: 2, type: 'approval', approval: 2, call: 1, summary: '', access: 'execute' });
  applyEvent(s, { seq: 3, type: 'approved', approval: 2, decision: 'deny' });
  assert.equal(s.blocks[0].status, 'denied');

  applyEvent(s, { seq: 4, type: 'tool_call', call: 2, name: 'fetch_url', summary: 'fetch_url x', access: 'network', arguments: {} });
  applyEvent(s, { seq: 5, type: 'stopped' });
  assert.equal(s.blocks[1].status, 'failed');
  assert.equal(s.blocks.at(-1).type, 'stopped');
  assert.equal(s.ended, 'stopped');
});
