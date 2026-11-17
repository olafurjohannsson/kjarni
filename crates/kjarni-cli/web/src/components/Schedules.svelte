<script>
  import { onMount } from 'svelte';
  import Icon from './Icon.svelte';
  import { api, SignedOut } from '../lib/api.js';
  import {
    DAYS,
    EXAMPLES,
    PERMISSIONS,
    describeTime,
    describeWhen,
    fromForm,
    randomTopic,
    toForm,
  } from '../lib/schedules.js';

  // Things the daemon does by itself: the list, a form to make or change
  // one, and where notifications go.
  let { onopen, onsignedout } = $props();

  let schedules = $state([]);
  let notify = $state(null);
  let loaded = $state(false);
  let problem = $state('');

  // The schedule being edited: 'new', an id, or null.
  let editing = $state(null);
  let form = $state(toForm(null));
  let saving = $state(false);
  let formProblem = $state('');
  let confirming = $state(null);

  let topic = $state('');
  let token = $state('');
  let notifyState = $state('');
  let notifyProblem = $state('');

  let now = $state(new Date());

  async function load() {
    try {
      const r = await api.schedules();
      schedules = r.schedules;
      notify = r.notify;
      if (!loaded) topic = r.notify?.url ?? '';
      loaded = true;
      problem = '';
    } catch (e) {
      if (e instanceof SignedOut) onsignedout?.();
      else problem = e.message;
    }
  }

  function edit(schedule) {
    editing = schedule?.id ?? 'new';
    form = toForm(schedule);
    formProblem = '';
  }

  function startFrom(example) {
    edit(null);
    form = toForm({ ...example, notify: true, enabled: true });
  }

  async function save(event) {
    event.preventDefault();
    saving = true;
    formProblem = '';
    try {
      const spec = fromForm(form);
      if (editing === 'new') await api.createSchedule(spec);
      else await api.updateSchedule(editing, spec);
      editing = null;
      await load();
    } catch (e) {
      formProblem = e.message;
    } finally {
      saving = false;
    }
  }

  async function toggle(schedule) {
    try {
      await api.updateSchedule(schedule.id, { ...fromForm(toForm(schedule)), enabled: !schedule.enabled });
      await load();
    } catch (e) {
      problem = e.message;
    }
  }

  async function runNow(schedule) {
    try {
      const conversation = await api.runSchedule(schedule.id);
      onopen(conversation);
    } catch (e) {
      problem = e.message;
    }
  }

  async function remove(schedule) {
    confirming = null;
    try {
      await api.deleteSchedule(schedule.id);
      if (editing === schedule.id) editing = null;
      await load();
    } catch (e) {
      problem = e.message;
    }
  }

  function toggleDay(day) {
    form.days = form.days.includes(day) ? form.days.filter((d) => d !== day) : [...form.days, day];
  }

  function togglePermission(key) {
    form.permissions = form.permissions.includes(key)
      ? form.permissions.filter((k) => k !== key)
      : [...form.permissions, key];
  }

  async function saveNotify(event) {
    event?.preventDefault();
    notifyProblem = '';
    notifyState = 'saving';
    try {
      const settings = { url: topic.trim(), open_url: location.origin };
      if (token.trim()) settings.token = token.trim();
      await api.setNotifications(settings);
      token = '';
      await load();
      notifyState = topic.trim() ? 'saved' : 'off';
    } catch (e) {
      notifyProblem = e.message;
      notifyState = '';
    }
  }

  async function testNotify() {
    notifyProblem = '';
    notifyState = 'testing';
    try {
      await api.testNotification();
      notifyState = 'sent';
    } catch (e) {
      notifyProblem = e.message;
      notifyState = '';
    }
  }

  async function turnOff() {
    topic = '';
    token = '';
    await saveNotify();
  }

  const results = {
    running: ['Running', 'running'],
    done: ['Finished', 'ok'],
    error: ['Failed', 'bad'],
    stopped: ['Stopped', 'muted'],
  };

  onMount(() => {
    load();
    // Runs start and end by themselves; keep the list current.
    const refresh = setInterval(() => {
      now = new Date();
      if (!editing) load();
    }, 10000);
    return () => clearInterval(refresh);
  });
</script>

<div class="page">
  <header>
    <div>
      <h1>Schedules</h1>
      <p>Kjarni runs these on its own and lets you know when they finish.</p>
    </div>
    {#if editing === null}
      <button class="primary" onclick={() => edit(null)}><Icon name="plus" size={17} /> New schedule</button>
    {/if}
  </header>

  {#if problem}<div class="problem"><Icon name="alert" size={15} /> {problem}</div>{/if}

  {#snippet editor()}
    <form class="editor" onsubmit={save}>
      <label class="field">
        <span>Name</span>
        <input bind:value={form.name} placeholder="Morning briefing" maxlength="80" required />
      </label>
      <label class="field">
        <span>What should Kjarni do?</span>
        <textarea bind:value={form.prompt} rows="4" placeholder="Read these pages and give me a short briefing…" required></textarea>
      </label>

      <div class="field">
        <span>When</span>
        <div class="segmented" role="radiogroup" aria-label="When">
          <button type="button" role="radio" aria-checked={form.kind === 'daily'} class:on={form.kind === 'daily'} onclick={() => (form.kind = 'daily')}>At a time</button>
          <button type="button" role="radio" aria-checked={form.kind === 'every'} class:on={form.kind === 'every'} onclick={() => (form.kind = 'every')}>Repeatedly</button>
        </div>
        {#if form.kind === 'daily'}
          <div class="row">
            <input type="time" bind:value={form.time} required aria-label="Time" />
            <div class="days" aria-label="Days">
              {#each DAYS as day}
                <button type="button" class:on={form.days.includes(day)} aria-pressed={form.days.includes(day)} onclick={() => toggleDay(day)}>
                  {day.slice(0, 2)}
                </button>
              {/each}
            </div>
          </div>
          <div class="hint">{form.days.length === 0 ? 'Every day. Pick days to run on only those.' : describeWhen(fromForm(form).when)}</div>
        {:else}
          <div class="row">
            <span class="every">Every</span>
            <input type="number" min="1" bind:value={form.amount} required aria-label="How often" class="amount" />
            <select bind:value={form.unit} aria-label="Unit">
              <option value="minutes">minutes</option>
              <option value="hours">hours</option>
            </select>
          </div>
        {/if}
      </div>

      <div class="field">
        <span>Allowed without asking</span>
        <div class="permissions">
          {#each PERMISSIONS as p}
            <button type="button" class:on={form.permissions.includes(p.key)} aria-pressed={form.permissions.includes(p.key)} onclick={() => togglePermission(p.key)}>
              <Icon name={p.icon} size={16} />
              {p.label}
              <span class="tick"><Icon name="check" size={14} /></span>
            </button>
          {/each}
        </div>
        <div class="hint">Reading your files and notes is always allowed. Anything else it tries is declined, since nobody is there to ask.</div>
      </div>

      <label class="check">
        <input type="checkbox" bind:checked={form.notify} />
        <span>Notify me when it finishes{#if !notify}<span class="hint"> (set up notifications below)</span>{/if}</span>
      </label>

      {#if formProblem}<div class="problem"><Icon name="alert" size={15} /> {formProblem}</div>{/if}
      <div class="actions">
        <button class="primary" disabled={saving}>{editing === 'new' ? 'Create schedule' : 'Save'}</button>
        <button type="button" class="ghost" onclick={() => (editing = null)}>Cancel</button>
      </div>
    </form>
  {/snippet}

  {#if editing === 'new'}{@render editor()}{/if}

  {#if loaded && schedules.length === 0 && editing === null}
    <div class="empty">
      <Icon name="clock" size={28} />
      <p>Nothing scheduled yet. Start from one of these, or make your own.</p>
      <div class="examples">
        {#each EXAMPLES as example}
          <button onclick={() => startFrom(example)}>
            <span class="name">{example.name}</span>
            <span class="when">{describeWhen(example.when)}</span>
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <div class="list">
    {#each schedules as s (s.id)}
      {#if editing === s.id}
        {@render editor()}
      {:else}
        <article class:off={!s.enabled}>
          <div class="top">
            <div class="title">
              <h2>{s.name}</h2>
              <div class="when"><Icon name="clock" size={14} /> {describeWhen(s.when)}</div>
            </div>
            <button class="switch" role="switch" aria-checked={s.enabled} aria-label={s.enabled ? 'Turn off' : 'Turn on'} class:on={s.enabled} onclick={() => toggle(s)}><span></span></button>
          </div>
          <p class="prompt">{s.prompt}</p>
          <div class="status">
            {#if s.running}
              <span class="chip running"><span class="dot"></span> Running</span>
            {:else if s.enabled}
              <span class="next">Next {describeTime(s.next_run, now)}</span>
            {:else}
              <span class="next">Off</span>
            {/if}
            {#if s.last_conversation && s.last_result !== 'running'}
              <button class="last" onclick={() => onopen(s.last_conversation)}>
                <span class="chip {results[s.last_result]?.[1] ?? 'muted'}">{results[s.last_result]?.[0] ?? 'Ran'}</span>
                {describeTime(s.last_run, now)}
                <Icon name="chevron" size={13} />
              </button>
            {/if}
            {#if s.allow.length}
              <span class="may">
                {#each PERMISSIONS.filter((p) => p.tools.some((t) => s.allow.includes(t))) as p}
                  <span title={p.label}><Icon name={p.icon} size={14} /></span>
                {/each}
              </span>
            {/if}
          </div>
          <div class="buttons">
            {#if confirming === s.id}
              <span class="confirm">Delete “{s.name}”?</span>
              <button class="danger" onclick={() => remove(s)}>Delete</button>
              <button class="ghost" onclick={() => (confirming = null)}>Keep</button>
            {:else}
              <button class="secondary" disabled={s.running} onclick={() => runNow(s)}><Icon name="play" size={14} /> Run now</button>
              <button class="ghost" onclick={() => edit(s)}><Icon name="pencil" size={14} /> Edit</button>
              <button class="ghost icon" onclick={() => (confirming = s.id)} aria-label="Delete schedule" title="Delete"><Icon name="trash" size={15} /></button>
            {/if}
          </div>
        </article>
      {/if}
    {/each}
  </div>

  <section class="notify">
    <div class="heading"><Icon name="bell" size={17} /> <h2>Phone notifications</h2></div>
    <p>
      Kjarni can tell your phone when a schedule finishes, through <a href="https://ntfy.sh" target="_blank" rel="noopener noreferrer">ntfy</a>.
      Install the ntfy app, subscribe to a topic, and put the same topic's address here. Tapping a notification opens the run here.
    </p>
    <form onsubmit={saveNotify}>
      <div class="row">
        <input class="grow mono" bind:value={topic} placeholder="https://ntfy.sh/your-topic" aria-label="Topic address" />
        {#if !topic}
          <button type="button" class="ghost" onclick={() => (topic = 'https://ntfy.sh/' + randomTopic())}>Pick a name</button>
        {/if}
      </div>
      <input type="password" bind:value={token} placeholder={notify?.has_token ? 'Access token saved; type to replace' : 'Access token, if your server needs one'} aria-label="Access token" autocomplete="off" />
      <div class="hint">
        Anyone who knows a topic's name on ntfy.sh can read it, so use a long name nobody would guess, or your own ntfy server.
        The finished reply is what the notification says.
      </div>
      {#if notifyProblem}<div class="problem"><Icon name="alert" size={15} /> {notifyProblem}</div>{/if}
      <div class="actions">
        <button class="primary" disabled={notifyState === 'saving'}>Save</button>
        {#if notify}
          <button type="button" class="secondary" disabled={notifyState === 'testing'} onclick={testNotify}>Send a test</button>
          <button type="button" class="ghost" onclick={turnOff}>Turn off</button>
        {/if}
        <span class="saved">
          {#if notifyState === 'saved'}<Icon name="check" size={14} /> Saved{/if}
          {#if notifyState === 'sent'}<Icon name="check" size={14} /> Sent; check your phone{/if}
          {#if notifyState === 'off'}Notifications are off{/if}
        </span>
      </div>
    </form>
  </section>
</div>

<style>
  .page {
    display: grid;
    gap: 18px;
    padding-bottom: 48px;
    animation: kj-rise 0.25s ease-out;
  }
  header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }
  h1 {
    font-size: clamp(26px, 4vw, 32px);
    font-weight: 700;
    letter-spacing: -0.03em;
  }
  header p,
  .notify p {
    color: var(--muted);
    font-size: 15px;
    margin-top: 4px;
  }
  h2 {
    font-size: 16.5px;
    font-weight: 650;
    letter-spacing: -0.01em;
  }
  button {
    display: inline-flex;
    align-items: center;
    gap: 7px;
  }
  .primary,
  .secondary,
  .ghost,
  .danger {
    height: 38px;
    padding: 0 14px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 14px;
    transition: background 0.12s, border-color 0.12s, color 0.12s;
  }
  .primary {
    background: var(--core);
    color: var(--ink);
  }
  .primary:not(:disabled):hover {
    background: var(--core-hover);
  }
  .secondary {
    background: var(--bg3);
    border: 1px solid var(--border-strong);
  }
  .secondary:not(:disabled):hover {
    background: var(--bg4);
  }
  .ghost {
    color: var(--muted);
  }
  .ghost:hover {
    color: var(--text);
    background: var(--bg3);
  }
  .ghost.icon {
    padding: 0 10px;
  }
  .danger {
    color: var(--bad);
    border: 1px solid rgba(229, 115, 95, 0.4);
  }
  button:disabled {
    opacity: 0.55;
    cursor: default;
  }
  .problem {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--bad);
    font-size: 14px;
  }
  .hint {
    font-size: 13px;
    color: var(--subtle);
  }

  .list {
    display: grid;
    gap: 12px;
  }
  article {
    border: 1px solid var(--border);
    background: var(--bg2);
    border-radius: var(--radius);
    padding: 16px;
    display: grid;
    gap: 10px;
    min-width: 0;
  }
  article.off {
    opacity: 0.65;
  }
  .top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
  }
  .title {
    min-width: 0;
  }
  .when {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 13.5px;
    margin-top: 3px;
  }
  .prompt {
    color: var(--text2);
    font-size: 14.5px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .status {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px 14px;
    font-size: 13px;
    color: var(--muted);
  }
  .last {
    color: var(--muted);
    font-size: 13px;
    gap: 6px;
  }
  .last:hover {
    color: var(--text);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 99px;
    border: 1px solid var(--border-strong);
  }
  .chip.ok {
    color: var(--ok);
    border-color: rgba(127, 182, 133, 0.4);
  }
  .chip.bad {
    color: var(--bad);
    border-color: rgba(229, 115, 95, 0.4);
  }
  .chip.running {
    color: var(--accent);
    border-color: var(--core-line);
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--core);
    animation: kj-pulse 1.4s ease-in-out infinite;
  }
  .may {
    display: inline-flex;
    gap: 6px;
    color: var(--subtle);
    margin-left: auto;
  }
  .buttons {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 2px;
  }
  .confirm {
    font-size: 14px;
    color: var(--text2);
    margin-right: 4px;
  }
  .switch {
    flex: none;
    width: 40px;
    height: 24px;
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

  .editor {
    border: 1px solid var(--core-line);
    background: linear-gradient(var(--core-soft), var(--core-soft)), var(--bg2);
    border-radius: var(--radius);
    padding: 18px;
    display: grid;
    gap: 16px;
    animation: kj-rise 0.2s ease-out;
  }
  .field {
    display: grid;
    gap: 8px;
  }
  .field > span {
    font-weight: 600;
    font-size: 14px;
  }
  input:not([type='checkbox']),
  textarea,
  select {
    width: 100%;
    background: var(--well);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    padding: 10px 12px;
    font: inherit;
    font-size: 15px;
    color: var(--text);
  }
  input:focus,
  textarea:focus,
  select:focus {
    outline: none;
    border-color: var(--core-line);
  }
  textarea {
    resize: vertical;
    min-height: 96px;
    line-height: 1.5;
  }
  input[type='time'],
  .amount,
  select {
    width: auto;
  }
  .amount {
    width: 84px;
  }
  .mono {
    font-family: var(--code);
    font-size: 14px;
  }
  .row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .every {
    color: var(--muted);
  }
  .segmented {
    display: inline-flex;
    justify-self: start;
    background: var(--well);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 3px;
  }
  .segmented button {
    height: 32px;
    padding: 0 14px;
    border-radius: 8px;
    font-size: 14px;
    color: var(--muted);
  }
  .segmented .on {
    background: var(--bg4);
    color: var(--text);
  }
  .days {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .days button {
    width: 40px;
    height: 36px;
    justify-content: center;
    border-radius: 9px;
    border: 1px solid var(--border-strong);
    font-size: 13px;
    text-transform: capitalize;
    color: var(--muted);
  }
  .days .on {
    background: var(--core-soft);
    border-color: var(--core-line);
    color: var(--text);
  }
  .permissions {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 8px;
  }
  .permissions button {
    height: 42px;
    padding: 0 12px;
    border-radius: 10px;
    border: 1px solid var(--border-strong);
    background: var(--well);
    color: var(--text2);
    font-size: 14px;
  }
  /* The permission's own icon; the tick inside it keeps the tick's colour. */
  .permissions button > :global(svg) {
    color: var(--subtle);
  }
  .tick {
    margin-left: auto;
    display: grid;
    width: 18px;
    height: 18px;
    place-items: center;
    border-radius: 5px;
    border: 1px solid var(--border-strong);
    color: transparent;
  }
  .permissions .on {
    border-color: var(--core-line);
    background: var(--core-soft);
    color: var(--text);
  }
  .permissions .on .tick {
    background: var(--core);
    border-color: var(--core);
    color: var(--ink);
  }
  .check {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14.5px;
  }
  .check input {
    width: 18px;
    height: 18px;
    accent-color: var(--core);
  }
  .actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
  }
  .saved {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 13.5px;
    color: var(--ok);
  }

  .empty {
    display: grid;
    justify-items: center;
    gap: 12px;
    text-align: center;
    padding: 24px 0 8px;
    color: var(--subtle);
  }
  .empty p {
    color: var(--muted);
    font-size: 15px;
  }
  .examples {
    width: 100%;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
    margin-top: 4px;
  }
  .examples button {
    display: grid;
    gap: 4px;
    text-align: left;
    padding: 14px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--bg2);
    transition: background 0.12s, border-color 0.12s;
  }
  .examples button:hover {
    background: var(--bg3);
    border-color: var(--border-strong);
  }
  .examples .name {
    font-weight: 600;
    color: var(--text);
  }
  .examples .when {
    margin: 0;
  }

  .notify {
    border-top: 1px solid var(--border);
    padding-top: 22px;
    margin-top: 8px;
    display: grid;
    gap: 10px;
  }
  .notify .heading {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .notify .heading :global(svg) {
    color: var(--accent);
  }
  .notify a {
    color: var(--accent);
  }
  .notify form {
    display: grid;
    gap: 10px;
    margin-top: 4px;
  }

  @media (max-width: 560px) {
    .editor {
      padding: 14px;
    }
    .days button {
      width: 36px;
    }
    .may {
      margin-left: 0;
    }
  }
</style>
