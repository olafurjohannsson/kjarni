<script>
  import Icon from './Icon.svelte';
  import { callDetail } from '../lib/transcript.js';

  // One tool call: a line saying what it does, its result folded under it,
  // and, while it waits for the user, the question with its answers.
  let { block, onanswer, workdir = '' } = $props();

  let open = $state(false);
  let answering = $state(false);
  // Declining asks what to do instead before it is sent.
  let declining = $state(false);
  let reason = $state('');

  const kinds = {
    read_file: ['file', 'Read'],
    list_directory: ['folder', 'Looked in'],
    write_file: ['pencil', 'Write'],
    edit_file: ['pencil', 'Edit'],
    search_files: ['search', 'Searched files for'],
    search_documents: ['search', 'Searched documents for'],
    run_command: ['terminal', 'Run'],
    fetch_url: ['globe', 'Open'],
    remember: ['bookmark', 'Remember'],
    recall: ['bookmark', 'Recalled'],
    forget: ['bookmark', 'Forget'],
    transcribe: ['mic', 'Transcribed'],
    search_images: ['image', 'Searched photos for'],
  };

  const questions = {
    write: 'Kjarni wants to change a file',
    execute: 'Kjarni wants to run a command',
    network: 'Kjarni wants to go online',
  };

  let kind = $derived(kinds[block.name] ?? ['chip', block.name]);
  let detail = $derived(callDetail(block.arguments, workdir));
  let waiting = $derived(block.status === 'waiting');

  // What the user is asked to allow, in full: the file's new text, the
  // whole command, the address.
  let preview = $derived.by(() => {
    const a = block.arguments ?? {};
    if (block.name === 'write_file') return a.content;
    if (block.name === 'run_command') return a.command;
    if (block.name === 'fetch_url') return a.url;
    const values = Object.values(a);
    return values.length ? JSON.stringify(a, null, 2) : null;
  });

  // An edit shows as the lines it takes out and the lines it puts in.
  let diff = $derived.by(() => {
    if (block.name !== 'edit_file') return null;
    const a = block.arguments ?? {};
    if (typeof a.old_text !== 'string' || typeof a.new_text !== 'string') return null;
    return [
      ...a.old_text.split('\n').map((text) => ({ sign: '-', text })),
      ...a.new_text.split('\n').map((text) => ({ sign: '+', text })),
    ];
  });

  async function answer(decision, words) {
    answering = true;
    try {
      await onanswer(block.approval, decision, words?.trim() || undefined);
    } finally {
      answering = false;
    }
  }

  function decline(event) {
    event.preventDefault();
    answer('deny', reason);
  }

  function onReasonKey(event) {
    if (event.key === 'Enter' && !event.shiftKey) decline(event);
    if (event.key === 'Escape') declining = false;
  }

  function focus(node) {
    node.focus();
  }
</script>

<div class="step" class:waiting data-status={block.status}>
  <button class="line" onclick={() => (open = !open)} disabled={!block.output && !preview && !diff} aria-expanded={open}>
    <span class="state">
      {#if block.status === 'running'}
        <span class="spinner"></span>
      {:else if block.status === 'waiting'}
        <span class="dot"></span>
      {:else if block.status === 'done'}
        <Icon name="check" size={14} />
      {:else}
        <Icon name="x" size={14} />
      {/if}
    </span>
    <span class="icon"><Icon name={kind[0]} size={15} /></span>
    <span class="label">{kind[1]}</span>
    {#if detail}<span class="detail mono">{detail}</span>{/if}
    {#if block.status === 'denied'}<span class="tag">declined</span>{/if}
    {#if block.status === 'failed'}<span class="tag">failed</span>{/if}
    {#if block.output || preview || diff}
      <span class="chev" class:open><Icon name="chevron" size={14} /></span>
    {/if}
  </button>

  {#if waiting}
    <div class="ask">
      <div class="question">
        <Icon name="shield" size={16} />
        <span>{questions[block.access] ?? `Kjarni wants to use ${block.name}`}</span>
      </div>
      {#if diff}
        <pre class="diff">{#each diff as line}<span class={line.sign === '+' ? 'add' : 'del'}>{line.sign} {line.text}
</span>{/each}</pre>
      {:else if preview}
        <pre class="preview">{preview}</pre>
      {/if}
      {#if declining}
        <form class="why" onsubmit={decline}>
          <textarea
            bind:value={reason}
            use:focus
            onkeydown={onReasonKey}
            rows="2"
            maxlength="2000"
            placeholder="What should Kjarni do instead? (optional)"
            aria-label="What should Kjarni do instead?"
          ></textarea>
          <div class="answers">
            <button class="secondary" type="submit" disabled={answering}>Decline</button>
            <button class="ghost" type="button" disabled={answering} onclick={() => (declining = false)}>Back</button>
          </div>
        </form>
      {:else}
        <div class="answers">
          <button class="primary" disabled={answering} onclick={() => answer('once')}>Allow once</button>
          <button class="secondary" disabled={answering} onclick={() => answer('always')} title="Allow {block.name} for the rest of this conversation">
            Always allow
          </button>
          <button class="ghost" disabled={answering} onclick={() => (declining = true)}>Decline…</button>
        </div>
      {/if}
    </div>
  {:else if open}
    <div class="body">
      {#if diff}
        <pre class="diff">{#each diff as line}<span class={line.sign === '+' ? 'add' : 'del'}>{line.sign} {line.text}
</span>{/each}</pre>
      {:else if preview && block.name !== 'read_file'}
        <pre class="preview">{preview}</pre>
      {/if}
      {#if block.output}
        <pre class="output">{block.output}</pre>
      {/if}
    </div>
  {/if}
  {#if block.status === 'denied' && block.reason}
    <div class="said"><span>You said</span> {block.reason}</div>
  {/if}
</div>

<style>
  .step {
    min-width: 0;
    animation: kj-rise 0.18s ease-out;
  }
  .diff span {
    display: block;
  }
  .diff .del {
    color: #e8a597;
    background: rgba(229, 115, 95, 0.1);
  }
  .diff .add {
    color: #a9d4ad;
    background: rgba(127, 182, 133, 0.1);
  }
  .line {
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 9px;
    width: 100%;
    min-height: 34px;
    padding: 5px 10px 5px 8px;
    border-radius: 9px;
    text-align: left;
    color: var(--muted);
    font-size: 14px;
    transition: background 0.12s;
  }
  .line:not(:disabled):hover {
    background: var(--bg2);
    color: var(--text2);
  }
  .line:disabled {
    cursor: default;
  }
  .state {
    width: 16px;
    display: grid;
    place-items: center;
    flex: none;
  }
  [data-status='done'] .state {
    color: var(--ok);
  }
  [data-status='denied'] .state,
  [data-status='failed'] .state {
    color: var(--bad);
  }
  .spinner {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 1.75px solid var(--border-strong);
    border-top-color: var(--core);
    animation: kj-spin 0.7s linear infinite;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--core);
    animation: kj-pulse 1.4s ease-in-out infinite;
  }
  .icon {
    display: grid;
    color: var(--subtle);
    flex: none;
  }
  .label {
    flex: none;
    color: var(--text2);
  }
  .detail {
    font-size: 12px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .tag {
    flex: none;
    font-size: 11.5px;
    padding: 1px 7px;
    border-radius: 99px;
    border: 1px solid var(--border-strong);
    color: var(--muted);
  }
  .chev {
    margin-left: auto;
    display: grid;
    color: var(--subtle);
    transition: transform 0.15s;
    flex: none;
  }
  .chev.open {
    transform: rotate(90deg);
  }
  .body {
    margin: 4px 0 8px 33px;
    display: grid;
    gap: 8px;
  }
  pre {
    font-family: var(--code);
    font-size: 12.5px;
    line-height: 1.55;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    max-height: 320px;
    overflow: auto;
    background: var(--well);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 10px 12px;
    color: var(--text2);
  }

  .ask {
    margin: 6px 0 10px;
    border: 1px solid var(--core-line);
    background: linear-gradient(var(--core-soft), var(--core-soft)), var(--bg2);
    border-radius: var(--radius);
    padding: 14px;
    display: grid;
    gap: 12px;
    animation: kj-rise 0.2s ease-out;
  }
  .question {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    color: var(--text);
  }
  .question :global(svg) {
    color: var(--accent);
  }
  .ask pre {
    max-height: 240px;
    background: var(--well);
  }
  .why {
    display: grid;
    gap: 10px;
  }
  .why textarea {
    width: 100%;
    resize: vertical;
    min-height: 60px;
    background: var(--well);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    padding: 10px 12px;
    font: inherit;
    font-size: 15px;
    line-height: 1.45;
    color: var(--text);
  }
  .why textarea:focus {
    outline: none;
    border-color: var(--core-line);
  }
  .said {
    margin: 4px 0 2px 30px;
    font-size: 13.5px;
    color: var(--text2);
    overflow-wrap: anywhere;
  }
  .said span {
    color: var(--subtle);
    margin-right: 4px;
  }
  .answers {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .answers button {
    height: 38px;
    padding: 0 16px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 14.5px;
    transition: background 0.12s, border-color 0.12s;
  }
  .answers button:disabled {
    opacity: 0.6;
    cursor: default;
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
  .ghost:not(:disabled):hover {
    color: var(--text);
  }
  @media (max-width: 560px) {
    .answers {
      display: grid;
      grid-template-columns: 1fr 1fr;
    }
    .primary {
      grid-column: 1 / -1;
    }
    .answers button {
      height: 44px;
    }
  }
</style>
