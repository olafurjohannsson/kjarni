<script>
  import { onMount } from 'svelte';
  import Icon from './Icon.svelte';
  import { api, SignedOut } from '../lib/api.js';
  import { hunkHeading, statusLabel, totals } from '../lib/changes.js';

  // The files a conversation changed, each with its diff from before the
  // conversation first wrote it. `version` changes when a file is written,
  // to load the list again.
  let { id, version = 0, onclose, onsignedout } = $props();

  let changes = $state(null);
  let problem = $state('');
  let shut = $state(new Set());

  let sum = $derived(totals(changes));

  async function load() {
    try {
      changes = await api.changes(id);
      problem = '';
    } catch (e) {
      if (e instanceof SignedOut) onsignedout?.();
      else problem = e.message;
    }
  }

  function toggle(path) {
    const next = new Set(shut);
    if (next.has(path)) next.delete(path);
    else next.add(path);
    shut = next;
  }

  $effect(() => {
    version;
    load();
  });

  onMount(() => {
    const onKey = (e) => e.key === 'Escape' && onclose?.();
    addEventListener('keydown', onKey);
    return () => removeEventListener('keydown', onKey);
  });
</script>

<button class="scrim" onclick={onclose} aria-label="Close changes" tabindex="-1"></button>
<aside class="panel" aria-label="Changed files">
  <header>
    <div>
      <h2>Changed files</h2>
      {#if changes?.length}
        <p>
          {sum.files} {sum.files === 1 ? 'file' : 'files'} ·
          <span class="add">+{sum.added}</span> <span class="del">−{sum.removed}</span>
        </p>
      {/if}
    </div>
    <button class="close" onclick={onclose} aria-label="Close"><Icon name="x" size={18} /></button>
  </header>

  <div class="list">
    {#if problem}
      <div class="problem"><Icon name="alert" size={15} /> {problem}</div>
    {:else if changes && changes.length === 0}
      <p class="empty">This conversation has not changed any files.</p>
    {:else if changes}
      {#each changes as c (c.full_path)}
        {@const label = statusLabel(c)}
        {@const open = !shut.has(c.full_path)}
        <section class="file">
          <button class="name" onclick={() => toggle(c.full_path)} aria-expanded={open} title={c.full_path}>
            <span class="chev" class:open><Icon name="chevron" size={14} /></span>
            <span class="path mono">{c.path}</span>
            {#if label}<span class="tag">{label}</span>{/if}
            {#if !c.unreadable}
              <span class="counts mono"><span class="add">+{c.added}</span> <span class="del">−{c.removed}</span></span>
            {/if}
          </button>
          {#if open}
            {#if c.unreadable}
              <p class="note">Too large, or not text, to show.</p>
            {:else if c.hunks.length === 0}
              <p class="note">Written back to what it was.</p>
            {:else}
              <div class="diff">
                {#each c.hunks as h}
                  <div class="hunk">{hunkHeading(h)}</div>
                  <pre>{#each h.lines as line}<span class={line.sign === '+' ? 'add' : line.sign === '-' ? 'del' : 'same'}>{line.sign} {line.text}
</span>{/each}</pre>
                {/each}
              </div>
            {/if}
          {/if}
        </section>
      {/each}
      <p class="foot">Files changed by commands Kjarni ran are not listed.</p>
    {:else}
      <div class="loading"><span class="spinner"></span></div>
    {/if}
  </div>
</aside>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    z-index: 30;
    background: rgba(0, 0, 0, 0.45);
    animation: fade 0.15s ease-out;
  }
  .panel {
    position: fixed;
    inset: 0 0 0 auto;
    z-index: 31;
    width: min(720px, 92vw);
    display: flex;
    flex-direction: column;
    background: var(--bg);
    border-left: 1px solid var(--border);
    box-shadow: -20px 0 50px rgba(0, 0, 0, 0.4);
    animation: slide 0.2s cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    padding: 18px 16px 14px 20px;
    padding-top: max(18px, env(safe-area-inset-top));
    border-bottom: 1px solid var(--border);
  }
  h2 {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  header p {
    margin-top: 2px;
    font-size: 13px;
    color: var(--muted);
  }
  .close {
    width: 36px;
    height: 36px;
    flex: none;
    border-radius: 10px;
    display: grid;
    place-items: center;
    color: var(--muted);
  }
  .close:hover {
    background: var(--bg3);
    color: var(--text);
  }
  .list {
    flex: 1;
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 14px 16px max(24px, env(safe-area-inset-bottom));
    display: grid;
    align-content: start;
    gap: 12px;
  }
  .file {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg2);
    overflow: hidden;
  }
  .name {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    text-align: left;
    font-size: 13.5px;
  }
  .name:hover {
    background: var(--bg3);
  }
  .chev {
    display: grid;
    color: var(--subtle);
    transition: transform 0.12s;
  }
  .chev.open {
    transform: rotate(90deg);
  }
  .path {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text);
  }
  .mono {
    font-family: var(--code);
  }
  .tag {
    flex: none;
    font-size: 11.5px;
    padding: 1px 7px;
    border-radius: 99px;
    border: 1px solid var(--border-strong);
    color: var(--muted);
  }
  .counts {
    flex: none;
    font-size: 12.5px;
  }
  .add {
    color: var(--ok);
  }
  .del {
    color: var(--bad);
  }
  .diff {
    border-top: 1px solid var(--border);
    background: var(--well);
    overflow-x: auto;
  }
  .hunk {
    padding: 6px 12px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    color: var(--subtle);
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
  }
  .hunk:not(:first-child) {
    border-top: 1px solid var(--border);
  }
  pre {
    margin: 0;
    padding: 6px 0;
    font-family: var(--code);
    font-size: 12.5px;
    line-height: 1.55;
    min-width: max-content;
  }
  pre span {
    display: block;
    padding: 0 12px;
    white-space: pre;
  }
  pre .same {
    color: var(--muted);
  }
  pre .del {
    color: #e8a597;
    background: rgba(229, 115, 95, 0.1);
  }
  pre .add {
    color: #a9d4ad;
    background: rgba(127, 182, 133, 0.1);
  }
  .note {
    padding: 10px 14px 12px;
    border-top: 1px solid var(--border);
    font-size: 13.5px;
    color: var(--muted);
  }
  .empty,
  .foot {
    font-size: 13.5px;
    color: var(--muted);
    text-align: center;
    padding: 6px 0;
  }
  .foot {
    font-size: 12.5px;
    color: var(--subtle);
  }
  .problem {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--bad);
    font-size: 14px;
  }
  .loading {
    display: grid;
    place-items: center;
    padding: 40px 0;
  }
  .spinner {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    border: 2px solid var(--border-strong);
    border-top-color: var(--core);
    animation: kj-spin 0.7s linear infinite;
  }
  @keyframes slide {
    from {
      transform: translateX(24px);
      opacity: 0;
    }
  }
  @keyframes fade {
    from {
      opacity: 0;
    }
  }
  @media (max-width: 560px) {
    .panel {
      width: 100vw;
      border-left: none;
    }
  }
</style>
