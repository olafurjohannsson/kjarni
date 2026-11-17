import { test } from 'node:test';
import assert from 'node:assert/strict';
import { describeTime, describeWhen, fromForm, toForm } from '../src/lib/schedules.js';

test('schedules are described the way people say them', () => {
  assert.equal(describeWhen({ kind: 'daily', time: '07:30', days: [] }), 'Every day at 07:30');
  assert.equal(describeWhen({ kind: 'daily', time: '07:00', days: ['fri', 'mon'] }), 'Mon, Fri at 07:00');
  assert.equal(describeWhen({ kind: 'daily', time: '08:00', days: ['mon', 'tue', 'wed', 'thu', 'fri'] }), 'Weekdays at 08:00');
  assert.equal(describeWhen({ kind: 'daily', time: '10:00', days: ['sun', 'sat'] }), 'Weekends at 10:00');
  assert.equal(describeWhen({ kind: 'every', minutes: 60 }), 'Every hour');
  assert.equal(describeWhen({ kind: 'every', minutes: 360 }), 'Every 6 hours');
  assert.equal(describeWhen({ kind: 'every', minutes: 45 }), 'Every 45 minutes');
});

test('times are relative when near and dated when not', () => {
  const now = new Date(2026, 10, 17, 12, 0); // Tuesday noon, local time
  const at = (d, h, m) => new Date(2026, 10, d, h, m).getTime() / 1000;
  assert.equal(describeTime(at(17, 12, 20), now), 'in 20 min');
  assert.equal(describeTime(at(17, 11, 50), now), '10 min ago');
  assert.equal(describeTime(at(17, 18, 0), now), 'today 18:00');
  assert.equal(describeTime(at(18, 7, 30), now), 'tomorrow 07:30');
  assert.equal(describeTime(at(16, 7, 30), now), 'yesterday 07:30');
  assert.match(describeTime(at(20, 7, 30), now), /^Fri\S* 07:30$/);
  assert.equal(describeTime(null), '');
});

test('the form round-trips a schedule, permissions grouped', () => {
  const schedule = {
    name: 'Briefing',
    prompt: 'Read the news',
    when: { kind: 'daily', time: '07:30', days: ['sat', 'mon'] },
    allow: ['fetch_url', 'edit_file', 'write_file'],
    notify: false,
    enabled: true,
  };
  const form = toForm(schedule);
  assert.deepEqual(form.permissions, ['web', 'files']);
  assert.deepEqual(fromForm(form), {
    ...schedule,
    when: { kind: 'daily', time: '07:30', days: ['mon', 'sat'] },
    allow: ['fetch_url', 'write_file', 'edit_file'],
  });

  const every = toForm({ ...schedule, when: { kind: 'every', minutes: 120 } });
  assert.equal(every.amount, 2);
  assert.equal(every.unit, 'hours');
  assert.deepEqual(fromForm({ ...every, amount: '30', unit: 'minutes' }).when, { kind: 'every', minutes: 30 });

  const fresh = toForm(null);
  assert.equal(fresh.kind, 'daily');
  assert.equal(fresh.notify, true);
});
