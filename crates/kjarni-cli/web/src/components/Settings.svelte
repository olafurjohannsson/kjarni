<script>
  import { onMount } from 'svelte';
  import Icon from './Icon.svelte';
  import { api, SignedOut } from '../lib/api.js';
  import { describePhase, formatParams, switching } from '../lib/settings.js';
  import { countdown, deviceAddresses } from '../lib/signin.js';

  // The model the daemon runs, with the others it can switch to, and what
  // else it was started with.
  let { status, onchange, onsignedout } = $props();

  let models = $state(null);
  // A code for signing another device in, and the seconds it has left.
  let device = $state(null);
  let left = $state(0);
  let showing = $state(false);

  let addresses = $derived(deviceAddresses(location.origin, status?.allowed_hosts));
  let problem = $state('');
  let asked = $state(null);

  let busy = $derived(switching(models?.phase));
  let line = $derived(models ? describePhase(models.phase, models.downloaded_bytes, models.current) : '');

  async function load() {
    try {
      const before = models;
      models = await api.models();
      // A change just ended: the sidebar shows the model too.
      if (before && switching(before.phase) && !switching(models.phase)) onchange?.();
    } catch (e) {
      if (e instanceof SignedOut) onsignedout?.();
      else problem = e.message;
    }
  }

  async function pick(name) {
    asked = name;
    problem = '';
    try {
      await api.switchModel(name);
      await load();
    } catch (e) {
      if (e instanceof SignedOut) onsignedout?.();
      else problem = e.message;
    } finally {
      asked = null;
    }
  }

  async function toggleThinking() {
    problem = '';
    const on = !models.thinking;
    models.thinking = on;
    try {
      await api.setThinking(on);
    } catch (e) {
      models.thinking = !on;
      if (e instanceof SignedOut) onsignedout?.();
      else problem = e.message;
    }
  }

  async function showCode() {
    showing = true;
    problem = '';
    try {
      const made = await api.deviceCode();
      device = { code: made.code, until: Date.now() + made.expires_in * 1000 };
      left = made.expires_in;
    } catch (e) {
      if (e instanceof SignedOut) onsignedout?.();
      else problem = e.message;
    } finally {
      showing = false;
    }
  }

  $effect(() => {
    if (!device) return;
    const tick = setInterval(() => {
      left = (device.until - Date.now()) / 1000;
      if (left <= 0) {
        device = null;
        clearInterval(tick);
      }
    }, 500);
    return () => clearInterval(tick);
  });

  onMount(() => {
    load();
    // Quickly while a change runs, to show its progress.
    let timer;
    const poll = async () => {
      await load();
      timer = setTimeout(poll, switching(models?.phase) ? 1000 : 10000);
    };
    timer = setTimeout(poll, 1000);
    return () => clearTimeout(timer);
  });
</script>

<div class="page">
  <header>
    <h1>Settings</h1>
    <p>The model Kjarni runs, and what else it was started with.</p>
  </header>

  <section>
    <h2>Model</h2>
    {#if models}
      {#if line}
        <div class="phase" class:failed={models.phase.state === 'failed'}>
          {#if busy}<span class="spinner"></span>{:else}<Icon name="alert" size={16} />{/if}
          <span>{line}</span>
        </div>
      {/if}
      {#if problem}<div class="problem"><Icon name="alert" size={15} /> {problem}</div>{/if}

      <div class="choices" role="radiogroup" aria-label="Model">
        {#if !models.choices.some((c) => c.name === models.current)}
          <div class="choice current" role="radio" aria-checked="true">
            <span class="mark"><Icon name="check" size={15} /></span>
            <span class="name mono">{models.current}</span>
            <span class="size"></span>
            <span class="state">Running on the {models.gpu ? 'GPU' : 'CPU'}</span>
          </div>
        {/if}
        {#each models.choices as c (c.name)}
          {@const current = c.name === models.current}
          {@const target = busy && models.phase.model === c.name}
          <button
            class="choice"
            class:current
            role="radio"
            aria-checked={current}
            disabled={current || busy || asked !== null}
            onclick={() => pick(c.name)}
          >
            <span class="mark">
              {#if target}<span class="spinner"></span>{:else if current}<Icon name="check" size={15} />{/if}
            </span>
            <span class="name mono">{c.name}</span>
            <span class="size">{formatParams(c.params_millions)}</span>
            <span class="state">
              {#if current}
                Running on the {models.gpu ? 'GPU' : 'CPU'}
              {:else if c.downloaded}
                Downloaded
              {:else}
                Downloads first
              {/if}
            </span>
          </button>
        {/each}
      </div>
      <p class="hint">
        Larger models answer better and need more memory: the 4B runs in about 3 GB, the 9B in about 6 GB. A switch waits
        for replies in progress to finish, and new messages wait until the model is loaded.
      </p>
      {#if models.pinned && models.pinned !== models.current}
        <p class="hint">
          The daemon was started with
          <code>{models.choices.some((c) => c.name === models.pinned) ? '--model' : '--model-path'} {models.pinned}</code>,
          so it loads that again at its next start.
        </p>
      {/if}
    {:else if problem}
      <div class="problem"><Icon name="alert" size={15} /> {problem}</div>
    {/if}
  </section>

  {#if models}
    <section>
      <h2>Thinking</h2>
      <div class="row">
        <div>
          <div class="label">Think before answering</div>
          <p class="hint">
            The model reasons first, which you can open under each reply. Answers come later and use more of the
            reply's tokens. Applies from the next message.
          </p>
        </div>
        <button
          class="switch"
          class:on={models.thinking}
          role="switch"
          aria-checked={models.thinking}
          aria-label="Think before answering"
          onclick={toggleThinking}><span></span></button
        >
      </div>
    </section>
  {/if}

  <section>
    <h2>Other devices</h2>
    {#if addresses.length}
      <p class="lead">
        Open Kjarni at
        {#each addresses as a, i}{#if i > 0}{' or '}{/if}<code>{a}</code>{/each}
        on your phone or another computer, then enter a code from here.
      </p>
    {:else}
      <p class="lead">
        Other devices cannot reach this daemon yet. To use it from your phone over Tailscale, start it with
        <code>--allow-host</code> and your machine's Tailscale name, then run <code>tailscale serve --bg 7710</code>.
      </p>
    {/if}
    {#if device}
      <div class="device">
        <div class="code mono" aria-label="Sign-in code">{device.code.slice(0, 4)} {device.code.slice(4)}</div>
        <div class="left">Works once, for {countdown(left)}</div>
      </div>
    {:else}
      <div>
        <button class="show" onclick={showCode} disabled={showing}>Show a sign-in code</button>
      </div>
    {/if}
  </section>

  {#if status}
    <section>
      <h2>This daemon</h2>
      <dl>
        <dt>Working folder</dt>
        <dd class="mono">{status.workdir}</dd>
        <dt>Runs on</dt>
        <dd>{status.gpu ? 'GPU' : 'CPU'}</dd>
        <dt>Memory</dt>
        <dd>{status.memory ? 'On' : 'Off'}</dd>
        <dt>Version</dt>
        <dd class="mono">{status.version}</dd>
      </dl>
      <p class="hint">These are set when the daemon starts: see <code>kjarni daemon install --help</code>.</p>
    </section>
  {/if}
</div>

<style>
  .page {
    display: grid;
    gap: 28px;
    padding-bottom: 48px;
    animation: kj-rise 0.25s ease-out;
  }
  h1 {
    font-size: clamp(26px, 4vw, 32px);
    font-weight: 700;
    letter-spacing: -0.03em;
  }
  header p {
    color: var(--muted);
    font-size: 15px;
    margin-top: 4px;
  }
  section {
    display: grid;
    gap: 12px;
  }
  .row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
  }
  .label {
    font-weight: 600;
  }
  .switch {
    flex: none;
    width: 40px;
    height: 24px;
    margin-top: 2px;
    border-radius: 99px;
    background: var(--bg4);
    border: 1px solid var(--border-strong);
    position: relative;
    transition: background 0.15s;
  }
  .switch span {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--muted);
    transition: transform 0.15s, background 0.15s;
  }
  .switch.on {
    background: var(--core-soft);
    border-color: var(--core-line);
  }
  .switch.on span {
    transform: translateX(16px);
    background: var(--core);
  }
  h2 {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--subtle);
  }
  .mono,
  code {
    font-family: var(--code);
    font-size: 0.92em;
  }
  .phase {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    border-radius: var(--radius);
    border: 1px solid var(--core-line);
    background: linear-gradient(var(--core-soft), var(--core-soft)), var(--bg2);
    font-size: 14px;
  }
  .phase.failed {
    border-color: rgba(229, 115, 95, 0.4);
    background: rgba(229, 115, 95, 0.08);
  }
  .phase.failed :global(svg) {
    color: var(--bad);
    flex: none;
  }
  .problem {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--bad);
    font-size: 14px;
  }
  .choices {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg2);
    overflow: hidden;
  }
  .choice {
    width: 100%;
    display: grid;
    grid-template-columns: 20px minmax(0, 1fr) auto;
    grid-template-areas: 'mark name size' 'mark state state';
    align-items: center;
    gap: 2px 12px;
    padding: 12px 16px 12px 14px;
    text-align: left;
    transition: background 0.12s;
  }
  .choice + .choice {
    border-top: 1px solid var(--border);
  }
  .choice:not(:disabled):hover {
    background: var(--bg3);
  }
  .choice:disabled {
    cursor: default;
  }
  .choice.current {
    background: var(--bg3);
  }
  .mark {
    grid-area: mark;
    display: grid;
    place-items: center;
    color: var(--ok);
  }
  .name {
    grid-area: name;
    font-size: 14px;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .size {
    grid-area: size;
    font-size: 12.5px;
    color: var(--muted);
  }
  .state {
    grid-area: state;
    font-size: 12.5px;
    color: var(--subtle);
  }
  .current .state {
    color: var(--ok);
  }
  .spinner {
    width: 13px;
    height: 13px;
    flex: none;
    border-radius: 50%;
    border: 1.75px solid var(--border-strong);
    border-top-color: var(--core);
    animation: kj-spin 0.7s linear infinite;
  }
  .lead {
    font-size: 14px;
    color: var(--muted);
    line-height: 1.55;
    overflow-wrap: anywhere;
  }
  .show {
    height: 40px;
    padding: 0 16px;
    border-radius: 10px;
    border: 1px solid var(--border-strong);
    background: var(--bg2);
    font-weight: 600;
    font-size: 14px;
    transition: background 0.12s;
  }
  .show:not(:disabled):hover {
    background: var(--bg3);
  }
  .device {
    display: grid;
    justify-items: center;
    gap: 6px;
    padding: 20px 16px;
    border: 1px solid var(--core-line);
    border-radius: var(--radius);
    background: linear-gradient(var(--core-soft), var(--core-soft)), var(--bg2);
  }
  .device .code {
    font-family: var(--mono);
    font-size: clamp(28px, 7vw, 36px);
    letter-spacing: 0.14em;
    color: var(--text);
  }
  .left {
    font-size: 13px;
    color: var(--muted);
  }
  .hint {
    font-size: 13px;
    color: var(--subtle);
    line-height: 1.55;
  }
  dl {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 10px 20px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg2);
    padding: 14px 16px;
    font-size: 14px;
  }
  dt {
    color: var(--muted);
  }
  dd {
    overflow-wrap: anywhere;
  }
  @media (max-width: 560px) {
    dl {
      grid-template-columns: 1fr;
      gap: 2px;
    }
    dd + dt {
      margin-top: 8px;
    }
  }
</style>
