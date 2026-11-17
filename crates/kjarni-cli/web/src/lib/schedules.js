// Schedules as the page shows and edits them. The daemon stores a schedule
// as { name, prompt, when, allow, notify, enabled }, where `when` is
// { kind: 'daily', time: 'HH:MM', days: [...] } or { kind: 'every', minutes }.

export const DAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'];
const DAY_NAMES = { mon: 'Mon', tue: 'Tue', wed: 'Wed', thu: 'Thu', fri: 'Fri', sat: 'Sat', sun: 'Sun' };

// What a schedule may do without asking, as the user thinks of it, and the
// daemon's tools behind each. Reading files and notes always runs.
export const PERMISSIONS = [
  { key: 'web', label: 'Open web pages', icon: 'globe', tools: ['fetch_url'] },
  { key: 'files', label: 'Change files', icon: 'pencil', tools: ['write_file', 'edit_file'] },
  { key: 'commands', label: 'Run commands', icon: 'terminal', tools: ['run_command'] },
  { key: 'memory', label: 'Update memory', icon: 'bookmark', tools: ['remember', 'forget'] },
];

/** "Every day at 07:30", "Mon, Fri at 07:00", "Every 2 hours". */
export function describeWhen(when) {
  if (!when) return '';
  if (when.kind === 'every') {
    const m = when.minutes;
    if (m % 1440 === 0) return m === 1440 ? 'Every day' : `Every ${m / 1440} days`;
    if (m % 60 === 0) return m === 60 ? 'Every hour' : `Every ${m / 60} hours`;
    return `Every ${m} minutes`;
  }
  const days = when.days ?? [];
  const weekdays = ['mon', 'tue', 'wed', 'thu', 'fri'];
  let on;
  if (days.length === 0 || days.length === 7) on = 'Every day';
  else if (days.length === 5 && weekdays.every((d) => days.includes(d))) on = 'Weekdays';
  else if (days.length === 2 && days.includes('sat') && days.includes('sun')) on = 'Weekends';
  else on = DAYS.filter((d) => days.includes(d)).map((d) => DAY_NAMES[d]).join(', ');
  return `${on} at ${when.time}`;
}

/** A unix time said the way people say it: "in 5 min", "today 07:30",
 * "tomorrow 07:30", "Fri 07:30", "21 Nov 07:30". */
export function describeTime(seconds, now = new Date()) {
  if (seconds == null) return '';
  const date = new Date(seconds * 1000);
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
  const diff = (date - now) / 60000;
  if (Math.abs(diff) < 60) {
    const m = Math.round(Math.abs(diff));
    if (m === 0) return 'now';
    return diff > 0 ? `in ${m} min` : `${m} min ago`;
  }
  const day = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((day(date) - day(now)) / 86400000);
  if (days === 0) return `today ${time}`;
  if (days === 1) return `tomorrow ${time}`;
  if (days === -1) return `yesterday ${time}`;
  if (days > 1 && days < 7) return `${date.toLocaleDateString([], { weekday: 'short' })} ${time}`;
  return `${date.toLocaleDateString([], { day: 'numeric', month: 'short' })} ${time}`;
}

/** The form's fields for a schedule, or for a new one. */
export function toForm(schedule) {
  const when = schedule?.when ?? { kind: 'daily', time: '07:30', days: [] };
  const allow = schedule?.allow ?? [];
  const every = when.kind === 'every' ? when.minutes : 60;
  return {
    name: schedule?.name ?? '',
    prompt: schedule?.prompt ?? '',
    kind: when.kind,
    time: when.kind === 'daily' ? when.time : '07:30',
    days: when.kind === 'daily' ? [...(when.days ?? [])] : [],
    amount: every % 60 === 0 ? every / 60 : every,
    unit: every % 60 === 0 ? 'hours' : 'minutes',
    permissions: PERMISSIONS.filter((p) => p.tools.some((t) => allow.includes(t))).map((p) => p.key),
    notify: schedule?.notify ?? true,
    enabled: schedule?.enabled ?? true,
  };
}

/** The schedule the daemon stores, from the form's fields. */
export function fromForm(form) {
  const when =
    form.kind === 'every'
      ? { kind: 'every', minutes: Math.round(Number(form.amount) * (form.unit === 'hours' ? 60 : 1)) }
      : { kind: 'daily', time: form.time, days: DAYS.filter((d) => form.days.includes(d)) };
  return {
    name: form.name.trim(),
    prompt: form.prompt.trim(),
    when,
    allow: PERMISSIONS.filter((p) => form.permissions.includes(p.key)).flatMap((p) => p.tools),
    notify: form.notify,
    enabled: form.enabled,
  };
}

/** A topic name nobody will guess, for ntfy.sh where topics are public to
 * anyone who knows the name. */
export function randomTopic() {
  const bytes = crypto.getRandomValues(new Uint8Array(9));
  return 'kjarni-' + Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

// Ideas for a first schedule.
export const EXAMPLES = [
  {
    name: 'Morning briefing',
    prompt:
      'Read these pages and give me a short briefing of what is new, five bullet points at most:\n- https://\n- https://',
    when: { kind: 'daily', time: '07:30', days: ['mon', 'tue', 'wed', 'thu', 'fri'] },
    allow: ['fetch_url'],
  },
  {
    name: 'Weekly review',
    prompt: 'Look through my notes and list the open tasks and anything with a date coming up in the next week.',
    when: { kind: 'daily', time: '18:00', days: ['sun'] },
    allow: [],
  },
  {
    name: 'Keep an eye on a page',
    prompt:
      'Open https:// and tell me in one line what it says about <the thing you care about>. Start with CHANGED if it differs from what you remembered last time, and remember what it says now.',
    when: { kind: 'every', minutes: 360 },
    allow: ['fetch_url', 'remember'],
  },
];
