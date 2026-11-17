<script>
  import Mark from './Mark.svelte';
  import { api } from '../lib/api.js';

  // Shown when this browser has no sign-in. On the daemon's own computer,
  // `kjarni daemon open` signs in; from another device, a code shown on a
  // signed-in one does, or the token from the daemon's token file.
  let { onsignedin } = $props();

  let code = $state('');
  let token = $state('');
  let error = $state('');
  let busy = $state(false);
  let withToken = $state(false);

  let digits = $derived(code.replace(/\D/g, ''));

  async function attempt(call, message) {
    busy = true;
    error = '';
    try {
      await call();
      onsignedin();
    } catch {
      error = message;
    } finally {
      busy = false;
    }
  }

  function submitCode(e) {
    e.preventDefault();
    if (digits.length !== 8) return;
    attempt(() => api.signInWithCode(digits), 'That code did not work. Codes last two minutes and work once; show a new one.');
  }

  function submitToken(e) {
    e.preventDefault();
    if (!token.trim()) return;
    attempt(() => api.signIn(token.trim()), 'That token did not work. Copy it again from the token file.');
  }
</script>

<main>
  <div class="card">
    <Mark size={40} blink />
    <h1>Sign in to Kjarni</h1>
    <p>On the computer Kjarni runs on, this signs you in:</p>
    <pre>kjarni daemon open</pre>
    <div class="or"><span>or, from another device</span></div>
    {#if !withToken}
      <form onsubmit={submitCode}>
        <label for="code">Enter the code from <b>Settings → Other devices</b> on a signed-in device</label>
        <input
          id="code"
          class="code"
          bind:value={code}
          inputmode="numeric"
          autocomplete="one-time-code"
          maxlength="9"
          placeholder="0000 0000"
        />
        {#if error}<div class="error">{error}</div>{/if}
        <button type="submit" disabled={busy || digits.length !== 8}>Sign in</button>
      </form>
      <button class="link" onclick={() => ((withToken = true), (error = ''))}>Use the token file instead</button>
    {:else}
      <form onsubmit={submitToken}>
        <label for="token">Paste the token from the daemon's <code>token</code> file</label>
        <input id="token" type="password" bind:value={token} autocomplete="current-password" spellcheck="false" placeholder="Token" />
        {#if error}<div class="error">{error}</div>{/if}
        <button type="submit" disabled={busy || !token.trim()}>Sign in</button>
      </form>
      <button class="link" onclick={() => ((withToken = false), (error = ''))}>Use a code instead</button>
    {/if}
  </div>
</main>

<style>
  main {
    height: 100%;
    display: grid;
    place-items: center;
    padding: 24px 16px;
    overflow-y: auto;
  }
  .card {
    width: min(100%, 420px);
    display: grid;
    gap: 12px;
    animation: kj-rise 0.3s ease-out;
  }
  h1 {
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.035em;
    line-height: 1.15;
    margin-top: 12px;
  }
  p,
  label {
    color: var(--muted);
    font-size: 15px;
  }
  pre {
    font-family: var(--mono);
    font-size: 13px;
    background: var(--well);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    color: var(--text);
  }
  pre::before {
    content: '$ ';
    color: var(--core);
  }
  .or {
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--subtle);
    font-size: 12.5px;
    margin: 10px 0 2px;
  }
  .or::before,
  .or::after {
    content: '';
    flex: 1;
    border-top: 1px solid var(--border);
  }
  form {
    display: grid;
    gap: 10px;
  }
  code {
    font-family: var(--code);
    font-size: 0.9em;
    color: var(--text2);
  }
  input {
    height: 46px;
    padding: 0 14px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--bg2);
    outline: none;
    font-size: 16px;
  }
  input:focus {
    border-color: var(--border-strong);
  }
  button {
    height: 46px;
    border-radius: 10px;
    background: var(--core);
    color: var(--ink);
    font-weight: 650;
  }
  button:not(:disabled):hover {
    background: var(--core-hover);
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .code {
    font-family: var(--mono);
    font-size: 22px;
    letter-spacing: 0.18em;
    text-align: center;
    height: 54px;
  }
  label b {
    color: var(--text2);
    font-weight: 600;
  }
  .link {
    height: auto;
    background: none;
    color: var(--muted);
    font-weight: 500;
    font-size: 14px;
    justify-self: center;
    padding: 4px;
  }
  .link:not(:disabled):hover {
    background: none;
    color: var(--text);
  }
  .error {
    color: var(--bad);
    font-size: 14px;
  }
</style>
