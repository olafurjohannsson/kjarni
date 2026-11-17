<script>
  import { onMount } from 'svelte';
  import Icon from './Icon.svelte';
  import { api, SignedOut } from '../lib/api.js';
  import { filterFacts } from '../lib/settings.js';

  // What Kjarni remembers between conversations: to read, add to and
  // correct by forgetting.
  let { onsignedout } = $props();

  let facts = $state([]);
  let loaded = $state(false);
  let off = $state(false);
  let problem = $state('');
  let text = $state('');
  let saving = $state(false);
  let query = $state('');
  let confirming = $state(null);

  let newestFirst = $derived([...facts].sort((a, b) => b.created - a.created || b.id - a.id));
  let shown = $derived(filterFacts(newestFirst, query));

  function fail(e) {
    if (e instanceof SignedOut) onsignedout?.();
    else problem = e.message;
  }

  async function load() {
    try {
      facts = await api.memory();
      off = false;
      problem = '';
    } catch (e) {
      if (e.message?.startsWith('memory is off')) off = true;
      else fail(e);
    } finally {
      loaded = true;
    }
  }

  async function remember(event) {
    event.preventDefault();
    if (!text.trim()) return;
    saving = true;
    try {
      const fact = await api.remember(text.trim());
      if (!facts.some((f) => f.id === fact.id)) facts = [...facts, fact];
      text = '';
      problem = '';
    } catch (e) {
      fail(e);
    } finally {
      saving = false;
    }
  }

  async function forget(id) {
    confirming = null;
    try {
      await api.forget(id);
      facts = facts.filter((f) => f.id !== id);
    } catch (e) {
      fail(e);
    }
  }

  function when(created) {
    const date = new Date(created * 1000);
    const sameYear = date.getFullYear() === new Date().getFullYear();
    return date.toLocaleDateString([], { day: 'numeric', month: 'short', year: sameYear ? undefined : 'numeric' });
  }

  onMount(load);
</script>

<div class="page">
  <header>
    <h1>Memory</h1>
    <p>What Kjarni remembers about you and your work. It recalls what fits each conversation.</p>
  </header>

  {#if off}
    <div class="empty">
      <Icon name="bookmark" size={28} />
      <p>Memory is off: the daemon was started with <code>--no-memory</code>.</p>
    </div>
  {:else if loaded}
    <form class="add" onsubmit={remember}>
      <input bind:value={text} placeholder="Something Kjarni should remember" maxlength="2000" aria-label="Something Kjarni should remember" />
      <button class="primary" disabled={saving || !text.trim()}><Icon name="plus" size={16} /> Remember</button>
    </form>

    {#if problem}<div class="problem"><Icon name="alert" size={15} /> {problem}</div>{/if}

    {#if facts.length === 0}
      <div class="empty">
        <Icon name="bookmark" size={28} />
        <p>Nothing remembered yet. Add something above, or tell Kjarni in a conversation, such as “remember that I like short answers”.</p>
      </div>
    {:else}
      {#if facts.length > 6}
        <label class="search">
          <Icon name="search" size={16} />
          <input bind:value={query} type="search" placeholder="Search {facts.length} memories" aria-label="Search memories" />
        </label>
      {/if}

      <ul>
        {#each shown as f (f.id)}
          <li>
            {#if confirming === f.id}
              <div class="confirm">
                <span>Forget this?</span>
                <button class="yes" onclick={() => forget(f.id)}>Forget</button>
                <button onclick={() => (confirming = null)}>Keep</button>
              </div>
            {:else}
              <p>{f.text}</p>
              <span class="date">{when(f.created)}</span>
              <button class="remove" onclick={() => (confirming = f.id)} aria-label="Forget" title="Forget">
                <Icon name="trash" size={15} />
              </button>
            {/if}
          </li>
        {:else}
          <li class="none">Nothing matches “{query}”.</li>
        {/each}
      </ul>
    {/if}
  {/if}
</div>

<style>
  .page {
    display: grid;
    gap: 18px;
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
  input {
    width: 100%;
    background: var(--well);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 15px;
    color: var(--text);
  }
  input:focus {
    outline: none;
    border-color: var(--core-line);
  }
  code {
    font-family: var(--code);
    font-size: 0.9em;
  }
  .add {
    display: flex;
    gap: 8px;
  }
  .add input {
    flex: 1;
    min-width: 0;
  }
  .primary {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    flex: none;
    height: 42px;
    padding: 0 14px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 14px;
    background: var(--core);
    color: var(--ink);
    transition: background 0.12s;
  }
  .primary:not(:disabled):hover {
    background: var(--core-hover);
  }
  .primary:disabled {
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
  .search {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--subtle);
  }
  .search input {
    flex: 1;
  }
  ul {
    list-style: none;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg2);
    overflow: hidden;
  }
  li {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 10px 12px 16px;
    min-height: 52px;
  }
  li + li {
    border-top: 1px solid var(--border);
  }
  li p {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
    line-height: 1.5;
  }
  .date {
    flex: none;
    font-size: 12px;
    color: var(--subtle);
  }
  .remove {
    flex: none;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: grid;
    place-items: center;
    color: var(--subtle);
    opacity: 0;
    transition: opacity 0.12s;
  }
  li:hover .remove,
  .remove:focus-visible {
    opacity: 1;
  }
  .remove:hover {
    color: var(--bad);
    background: var(--bg4);
  }
  @media (pointer: coarse) {
    .remove {
      opacity: 0.6;
    }
  }
  .confirm {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 14px;
    font-size: 14px;
    color: var(--text2);
  }
  .confirm button {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--muted);
  }
  .confirm .yes {
    color: var(--bad);
  }
  .none {
    color: var(--muted);
    font-size: 14px;
  }
  .empty {
    display: grid;
    justify-items: center;
    gap: 10px;
    text-align: center;
    padding: 40px 16px;
    color: var(--subtle);
  }
  .empty p {
    color: var(--muted);
    max-width: 44ch;
  }
  @media (max-width: 560px) {
    .date {
      display: none;
    }
  }
</style>
