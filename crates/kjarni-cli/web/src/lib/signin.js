// Signing another device in with a one-time link.

const LOOPBACK = ['localhost', '127.0.0.1', '[::1]', '::1'];

export function isLoopback(hostname) {
  return LOOPBACK.includes(hostname) || hostname.endsWith('.localhost');
}

/** The addresses another device can open the daemon at: this page's own
 * when it is not loopback, then each name the daemon was allowed to answer
 * to, over HTTPS as `tailscale serve` provides it. */
export function deviceAddresses(origin, allowedHosts) {
  const out = [];
  const here = new URL(origin);
  if (!isLoopback(here.hostname)) out.push(here.origin);
  for (const host of allowedHosts ?? []) {
    const address = `https://${host}`;
    if (!out.includes(address)) out.push(address);
  }
  return out;
}

export function loginLink(address, code) {
  return `${address}/login?code=${encodeURIComponent(code)}`;
}

/** "1:05" for 65 seconds. */
export function countdown(seconds) {
  const s = Math.max(0, Math.ceil(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
