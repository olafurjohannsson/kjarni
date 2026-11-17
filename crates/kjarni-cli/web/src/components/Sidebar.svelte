<script>
  import Mark from './Mark.svelte';
  import Icon from './Icon.svelte';

  let { conversations, current, status, page, onselect, onnew, onschedules, ondelete } = $props();

  let confirming = $state(null);

  function when(modified) {
    if (!modified) return '';
    const date = new Date(typeof modified === 'number' ? modified * 1000 : modified);
    if (isNaN(date)) return '';
    const days = Math.floor((startOfDay(new Date()) - startOfDay(date)) / 86400000);
    if (days === 0) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    if (days === 1) return 'Yesterday';
    if (days < 7) return date.toLocaleDateString([], { weekday: 'short' });
    return date.toLocaleDateString([], { day: 'numeric', month: 'short' });
  }

  // The last two folders of a path, which say which folder it is.
  function shortPath(path) {
    const parts = (path ?? '').split(/[\\/]/).filter(Boolean);
    return parts.length > 2 ? '…/' + parts.slice(-2).join('/') : path;
  }

  function startOfDay(d) {
    return new Date(d.getFullYear(), d.getMonth(), d.getDate());
  }
</script>

<aside>
  <div class="brand">
    <Mark size={22} />
    <span class="name">kjarni</span>
  </div>

  <button class="new" onclick={onnew}>
    <Icon name="plus" size={17} />
    New conversation
  </button>
  <button class="link" class:active={page === 'schedules'} onclick={onschedules}>
    <Icon name="clock" size={17} />
    Schedules
  </button>

  <nav aria-label="Conversations">
    {#if conversations.length}
      <div class="heading">Conversations</div>
    {/if}
    {#each conversations as c (c.id)}
      <div class="item" class:active={c.id === current}>
        {#if confirming === c.id}
          <div class="confirm">
            <span>Delete this conversation?</span>
            <button class="yes" onclick={() => (ondelete(c.id), (confirming = null))}>Delete</button>
            <button onclick={() => (confirming = null)}>Keep</button>
          </div>
        {:else}
          <button class="pick" onclick={() => onselect(c.id)}>
            <span class="title">{c.title || 'New conversation'}</span>
            <span class="meta">
              {#if c.busy}<span class="busy" title="Working"></span>{/if}
              {when(c.modified)}
            </span>
          </button>
          <button class="remove" onclick={() => (confirming = c.id)} aria-label="Delete conversation" title="Delete">
            <Icon name="trash" size={15} />
          </button>
        {/if}
      </div>
    {/each}
  </nav>

  {#if status}
    <footer>
      <div class="model">
        <Icon name="chip" size={15} />
        <span class="mono">{status.model}</span>
        <span class="where">{status.gpu ? 'GPU' : 'CPU'}</span>
      </div>
      <div class="folder" title={status.workdir}>
        <Icon name="folder" size={15} />
        <span>{shortPath(status.workdir)}</span>
      </div>
    </footer>
  {/if}
</aside>

<style>
  aside {
    height: 100%;
    display: flex;
    flex-direction: column;
    background: var(--bg2);
    border-right: 1px solid var(--border);
    padding: 14px 10px 10px;
    padding-top: max(14px, env(safe-area-inset-top));
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 10px 16px;
  }
  .name {
    font-weight: 700;
    letter-spacing: -0.035em;
    font-size: 19px;
  }
  .new {
    display: flex;
    align-items: center;
    gap: 9px;
    height: 40px;
    padding: 0 12px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--bg3);
    font-weight: 600;
    font-size: 14.5px;
    transition: background 0.12s, border-color 0.12s;
  }
  .new :global(svg) {
    color: var(--accent);
  }
  .new:hover {
    background: var(--bg4);
    border-color: var(--border-strong);
  }
  .link {
    display: flex;
    align-items: center;
    gap: 9px;
    height: 38px;
    margin-top: 6px;
    padding: 0 12px;
    border-radius: 10px;
    font-size: 14.5px;
    color: var(--text2);
    transition: background 0.12s;
  }
  .link :global(svg) {
    color: var(--subtle);
  }
  .link:hover,
  .link.active {
    background: var(--bg3);
  }
  .link.active :global(svg) {
    color: var(--accent);
  }
  nav {
    flex: 1;
    overflow-y: auto;
    margin: 14px -4px 0;
    padding: 0 4px;
  }
  .heading {
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--subtle);
    padding: 6px 10px 6px;
  }
  .item {
    position: relative;
    display: flex;
    align-items: center;
    border-radius: 9px;
    transition: background 0.12s;
  }
  .item:hover {
    background: var(--bg3);
  }
  .item.active {
    background: var(--bg3);
  }
  .item.active::before {
    content: '';
    position: absolute;
    left: 0;
    top: 9px;
    bottom: 9px;
    width: 2px;
    border-radius: 2px;
    background: var(--core);
  }
  .pick {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 4px 9px 12px;
    text-align: left;
  }
  .title {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 14.5px;
    color: var(--text2);
  }
  .active .title {
    color: var(--text);
  }
  .meta {
    flex: none;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--subtle);
  }
  .busy {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--core);
    animation: kj-pulse 1.4s ease-in-out infinite;
  }
  .remove {
    flex: none;
    width: 30px;
    height: 30px;
    margin-right: 4px;
    border-radius: 7px;
    display: grid;
    place-items: center;
    color: var(--subtle);
    opacity: 0;
    transition: opacity 0.12s;
  }
  .item:hover .remove,
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
    gap: 6px 10px;
    padding: 8px 12px;
    font-size: 13.5px;
    color: var(--text2);
  }
  .confirm button {
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
  }
  .confirm .yes {
    color: var(--bad);
  }
  footer {
    border-top: 1px solid var(--border);
    margin-top: 8px;
    padding: 12px 10px 4px;
    padding-bottom: max(4px, env(safe-area-inset-bottom));
    display: grid;
    gap: 7px;
    font-size: 12.5px;
    color: var(--muted);
  }
  footer > div {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  footer :global(svg) {
    flex: none;
    color: var(--subtle);
  }
  .model .mono {
    font-size: 11.5px;
    color: var(--text2);
  }
  .where {
    font-size: 10.5px;
    font-family: var(--mono);
    padding: 1px 6px;
    border-radius: 5px;
    background: var(--bg4);
    color: var(--muted);
  }
  .folder span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
