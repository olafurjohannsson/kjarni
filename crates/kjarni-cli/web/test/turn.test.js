import { test } from 'node:test';
import assert from 'node:assert/strict';
import { contextShare, describeContext, formatTokens, takeQueued } from '../src/lib/turn.js';
import { countdown, deviceAddresses, isLoopback, loginLink } from '../src/lib/signin.js';

test('a queued message leaves the queue once, when the agent reads it', () => {
  const queued = ['also bread', 'and eggs', 'also bread'];
  assert.deepEqual(takeQueued(queued, ' also bread '), { queued: ['and eggs', 'also bread'], found: true });
  assert.deepEqual(takeQueued(queued, 'milk'), { queued, found: false });
});

test('the context meter reads as a share of the budget', () => {
  assert.equal(contextShare({ used: 4096, budget: 16384 }), 25);
  assert.equal(contextShare({ used: 20000, budget: 16384 }), 100);
  assert.equal(contextShare(null), null);
  assert.equal(formatTokens(950), '950');
  assert.equal(formatTokens(1500), '1.5K');
  assert.equal(formatTokens(16384), '16K');
  assert.equal(
    describeContext({ used: 4096, budget: 16384 }),
    '4.1K of 16K tokens (25%). Past that, Kjarni summarizes the earlier messages to make room.',
  );
});

test('another device is offered the addresses it can reach', () => {
  assert.deepEqual(deviceAddresses('http://127.0.0.1:7710', []), []);
  assert.deepEqual(deviceAddresses('http://127.0.0.1:7710', ['box.tail1.ts.net']), ['https://box.tail1.ts.net']);
  assert.deepEqual(deviceAddresses('https://box.tail1.ts.net', ['box.tail1.ts.net']), ['https://box.tail1.ts.net']);
  assert.deepEqual(deviceAddresses('http://192.168.1.5:7710', ['box.tail1.ts.net']), [
    'http://192.168.1.5:7710',
    'https://box.tail1.ts.net',
  ]);
  assert.equal(isLoopback('localhost'), true);
  assert.equal(isLoopback('kjarni.localhost'), true);
  assert.equal(loginLink('https://box.tail1.ts.net', 'ab12'), 'https://box.tail1.ts.net/login?code=ab12');
  assert.equal(countdown(65.2), '1:06');
  assert.equal(countdown(-3), '0:00');
});
