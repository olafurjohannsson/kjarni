// The daemon's HTTP API. The browser is signed in by an HttpOnly cookie, so
// requests carry no token of their own.

export class SignedOut extends Error {}

async function request(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? {} : { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: 'same-origin',
  });
  if (response.status === 401) throw new SignedOut();
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      message = (await response.json()).error?.message ?? message;
    } catch {
      // Not JSON; the status says enough.
    }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.headers.get('content-type')?.includes('json') ? response.json() : null;
}

export const api = {
  status: () => request('GET', '/api/status'),
  conversations: () => request('GET', '/api/conversations').then((r) => r.conversations),
  create: () => request('POST', '/api/conversations').then((r) => r.id),
  conversation: (id) => request('GET', `/api/conversations/${encodeURIComponent(id)}`),
  remove: (id) => request('DELETE', `/api/conversations/${encodeURIComponent(id)}`),
  send: (id, text) => request('POST', `/api/conversations/${encodeURIComponent(id)}/messages`, { text }),
  answer: (id, approval, decision, reason) =>
    request('POST', `/api/conversations/${encodeURIComponent(id)}/approvals/${approval}`, { decision, reason }),
  changes: (id) => request('GET', `/api/conversations/${encodeURIComponent(id)}/changes`).then((r) => r.changes),
  stop: (id) => request('POST', `/api/conversations/${encodeURIComponent(id)}/stop`),
  signIn: (token) => request('POST', '/login', { token }),
  schedules: () => request('GET', '/api/schedules'),
  createSchedule: (spec) => request('POST', '/api/schedules', spec),
  updateSchedule: (id, spec) => request('PUT', `/api/schedules/${encodeURIComponent(id)}`, spec),
  deleteSchedule: (id) => request('DELETE', `/api/schedules/${encodeURIComponent(id)}`),
  runSchedule: (id) => request('POST', `/api/schedules/${encodeURIComponent(id)}/run`).then((r) => r.conversation),
  setNotifications: (settings) => request('PUT', '/api/notifications', settings),
  testNotification: () => request('POST', '/api/notifications/test'),
  memory: () => request('GET', '/api/memory').then((r) => r.facts),
  remember: (text) => request('POST', '/api/memory', { text }),
  forget: (id) => request('DELETE', `/api/memory/${id}`),
  models: () => request('GET', '/api/models'),
  switchModel: (model) => request('PUT', '/api/models/current', { model }),
};

/** Follows a conversation's events after `after`, calling `onEvent` for
 * each. Returns a function that stops following. The browser reconnects by
 * itself, resuming from the last event it saw. */
export function follow(id, after, onEvent, onTrouble) {
  const source = new EventSource(`/api/conversations/${encodeURIComponent(id)}/events?after=${after}`);
  source.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data));
    } catch (e) {
      console.error('bad event', e);
    }
  };
  source.onerror = () => onTrouble?.(source.readyState);
  return () => source.close();
}
