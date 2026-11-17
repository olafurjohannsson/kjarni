<script>
  import Mark from './Mark.svelte';
  import { api } from '../lib/api.js';

  // Shown when this browser has no sign-in. On the daemon's own computer,
  // `kjarni daemon open` signs in; from anywhere else, the token does.
  let { onsignedin } = $props();

  let token = $state('');
  let error = $state('');
  let busy = $state(false);

  async function submit(e) {
    e.preventDefault();
    if (!token.trim()) return;
    busy = true;
    error = '';
    try {
      await api.signIn(token.trim());
      onsignedin();
    } catch {
      error = 'That token did not work. Copy it again from the token file.';
    } finally {
      busy = false;
    }
  }
</script>

<main>
  <div class="card">
    <Mark size={40} blink />
    <h1>Sign in to Kjarni</h1>
    <p>On the computer Kjarni runs on, this signs you in:</p>
    <pre>kjarni daemon open</pre>
    <div class="or"><span>or, from another device</span></div>
    <form onsubmit={submit}>
      <label for="token">Paste the token from the daemon's <code>token</code> file</label>
      <input id="token" type="password" bind:value={token} autocomplete="current-password" spellcheck="false" placeholder="Token" />
      {#if error}<div class="error">{error}</div>{/if}
      <button type="submit" disabled={busy || !token.trim()}>Sign in</button>
    </form>
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
  .error {
    color: var(--bad);
    font-size: 14px;
  }
</style>
