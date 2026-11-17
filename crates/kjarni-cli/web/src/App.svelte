<script>
  import { onMount, tick } from 'svelte';
  import { api, follow, SignedOut } from './lib/api.js';
  import { applyEvent, fromHistory, newTurnState } from './lib/transcript.js';
  import Sidebar from './components/Sidebar.svelte';
  import Thread from './components/Thread.svelte';
  import Composer from './components/Composer.svelte';
  import Welcome from './components/Welcome.svelte';
  import SignIn from './components/SignIn.svelte';
  import Icon from './components/Icon.svelte';
  import Mark from './components/Mark.svelte';

  let signedIn = $state(null);
  let status = $state(null);
  let conversations = $state([]);
  let current = $state(null);
  let turn = $state(newTurnState([]));
  let busy = $state(false);
  let summary = $state(null);
  let loading = $state(false);
  let draft = $state('');
  let drawer = $state(false);
  let trouble = $state('');
  let atBottom = $state(true);

  let scroller = $state();
  let composer = $state();
  let unfollow = null;
  // Ignores replies to requests for a conversation the user has left.
  let visit = 0;

  let title = $derived(conversations.find((c) => c.id === current)?.title || (current ? 'Conversation' : 'Kjarni'));

  function idFromPath() {
    const m = location.pathname.match(/^\/c\/([A-Za-z0-9_-]{1,64})$/);
    return m ? m[1] : null;
  }

  function guard(e) {
    if (e instanceof SignedOut) {
      signedIn = false;
      return;
    }
    trouble = e.message || String(e);
    setTimeout(() => (trouble = ''), 6000);
  }

  async function boot() {
    try {
      status = await api.status();
      signedIn = true;
    } catch (e) {
      if (e instanceof SignedOut) signedIn = false;
      else trouble = 'Kjarni is not answering. Is the daemon running?';
      return;
    }
    await refreshList();
    await show(idFromPath(), false);
  }

  async function refreshList() {
    try {
      conversations = await api.conversations();
    } catch (e) {
      guard(e);
    }
  }

  function stopFollowing() {
    unfollow?.();
    unfollow = null;
  }

  function startFollowing(id, after) {
    stopFollowing();
    const mine = visit;
    unfollow = follow(
      id,
      after,
      (event) => {
        if (mine !== visit) return;
        onEvent(event);
      },
      () => {
        // The browser retries by itself; a closed stream means it gave up.
        if (mine === visit && busy) trouble = 'Lost the connection to Kjarni; retrying.';
      },
    );
  }

  /** Shows conversation `id`, or the welcome page for none. */
  async function show(id, push = true) {
    const mine = ++visit;
    stopFollowing();
    drawer = false;
    current = id;
    busy = false;
    summary = null;
    turn = newTurnState([]);
    if (push && location.pathname !== (id ? `/c/${id}` : '/')) {
      history.pushState(null, '', id ? `/c/${id}` : '/');
    }
    if (!id) {
      await tick();
      composer?.focus();
      return;
    }
    loading = true;
    try {
      const c = await api.conversation(id);
      if (mine !== visit) return;
      const state = newTurnState(fromHistory(c.messages));
      for (const e of c.events ?? []) applyEvent(state, e);
      turn = state;
      summary = c.summary;
      busy = c.busy;
      if (busy) startFollowing(id, turn.lastSeq);
      await scrollToEnd(true);
    } catch (e) {
      if (mine !== visit) return;
      if (e.message?.startsWith('there is no')) {
        await show(null);
        return;
      }
      guard(e);
    } finally {
      if (mine === visit) loading = false;
    }
  }

  function onEvent(event) {
    // The page shows the user's message as soon as it is sent.
    const last = turn.blocks.at(-1);
    if (event.type === 'user' && last?.type === 'user' && last.pending && last.text === event.text.trim()) {
      last.pending = false;
      turn.lastSeq = Math.max(turn.lastSeq, event.seq);
      return;
    }
    applyEvent(turn, event);
    trouble = '';
    if (turn.ended) {
      busy = false;
      stopFollowing();
      refreshList();
    }
    scrollToEnd();
  }

  async function send(text) {
    try {
      let id = current;
      if (!id) {
        id = await api.create();
        ++visit;
        current = id;
        history.pushState(null, '', `/c/${id}`);
        turn = newTurnState([]);
        summary = null;
      }
      turn.blocks.push({ type: 'user', text, pending: true });
      turn.ended = null;
      busy = true;
      await scrollToEnd(true);
      await api.send(id, text);
      // The turn's log starts with this message, so following from the
      // start misses nothing.
      startFollowing(id, 0);
      refreshList();
    } catch (e) {
      busy = false;
      const last = turn.blocks.at(-1);
      if (last?.pending) turn.blocks.pop();
      draft = text;
      guard(e);
    }
  }

  async function answer(approval, decision) {
    try {
      await api.answer(current, approval, decision);
    } catch (e) {
      guard(e);
    }
  }

  async function stop() {
    try {
      await api.stop(current);
    } catch (e) {
      guard(e);
    }
  }

  async function remove(id) {
    try {
      await api.remove(id);
      conversations = conversations.filter((c) => c.id !== id);
      if (id === current) await show(null);
    } catch (e) {
      guard(e);
    }
  }

  function pick(prompt) {
    draft = prompt;
    composer?.focus();
  }

  function onScroll() {
    if (!scroller) return;
    atBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
  }

  async function scrollToEnd(force = false) {
    if (!force && !atBottom) return;
    await tick();
    if (scroller) {
      scroller.scrollTop = scroller.scrollHeight;
      atBottom = true;
    }
  }

  onMount(() => {
    boot();
    const onPop = () => show(idFromPath(), false);
    addEventListener('popstate', onPop);
    return () => {
      removeEventListener('popstate', onPop);
      stopFollowing();
    };
  });
</script>

<svelte:head>
  <title>{current ? `${title} · Kjarni` : 'Kjarni'}</title>
</svelte:head>

{#if signedIn === false}
  <SignIn onsignedin={boot} />
{:else if signedIn === null}
  <div class="splash">
    <Mark size={40} blink />
    {#if trouble}<p>{trouble}</p>{/if}
  </div>
{:else}
  <div class="app" class:drawer>
    <div class="side">
      <Sidebar {conversations} {current} {status} onselect={(id) => show(id)} onnew={() => show(null)} ondelete={remove} />
    </div>
    <button class="scrim" onclick={() => (drawer = false)} aria-label="Close menu" tabindex="-1"></button>

    <main>
      <header>
        <button class="icon-button" onclick={() => (drawer = true)} aria-label="Conversations"><Icon name="menu" size={20} /></button>
        <div class="title">{title}</div>
        <button class="icon-button" onclick={() => show(null)} aria-label="New conversation"><Icon name="plus" size={20} /></button>
      </header>

      <div class="scroll" bind:this={scroller} onscroll={onScroll}>
        <div class="column">
          {#if !current}
            <Welcome {status} onpick={pick} />
          {:else if loading}
            <div class="loading"><Mark size={28} blink /></div>
          {:else}
            <Thread blocks={turn.blocks} {busy} {summary} onanswer={answer} workdir={status?.workdir} />
          {/if}
        </div>
      </div>

      <div class="dock">
        <div class="column">
          {#if !atBottom}
            <button class="jump" onclick={() => scrollToEnd(true)} aria-label="Jump to the latest"><Icon name="arrowdown" size={17} /></button>
          {/if}
          {#if trouble}<div class="trouble"><Icon name="alert" size={15} /> {trouble}</div>{/if}
          <Composer bind:this={composer} bind:value={draft} {busy} onsend={send} onstop={stop} />
          <div class="footnote">Runs on your computer. Nothing leaves it.</div>
        </div>
      </div>
    </main>
  </div>
{/if}

<style>
  .splash {
    height: 100%;
    display: grid;
    place-content: center;
    justify-items: center;
    gap: 16px;
    color: var(--muted);
    text-align: center;
    padding: 16px;
  }
  .app {
    height: 100%;
    display: grid;
    grid-template-columns: var(--sidebar) 1fr;
  }
  .side {
    min-height: 0;
  }
  .scrim {
    display: none;
  }
  main {
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    position: relative;
  }
  header {
    display: none;
  }
  .scroll {
    flex: 1;
    overflow-y: auto;
    overscroll-behavior: contain;
    scrollbar-gutter: stable;
  }
  .column {
    width: 100%;
    max-width: var(--column);
    margin: 0 auto;
    padding: 0 24px;
  }
  .scroll .column {
    padding-top: 40px;
    padding-bottom: 24px;
  }
  .loading {
    display: grid;
    place-items: center;
    padding: 20vh 0;
  }
  .dock {
    padding: 0 0 max(12px, env(safe-area-inset-bottom));
    background: linear-gradient(transparent, var(--bg) 24px);
    margin-top: -24px;
    position: relative;
  }
  .dock .column {
    position: relative;
    padding-top: 24px;
  }
  .footnote {
    text-align: center;
    font-size: 12px;
    color: var(--subtle);
    padding-top: 8px;
  }
  .jump {
    position: absolute;
    top: -22px;
    left: 50%;
    transform: translateX(-50%);
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: var(--bg3);
    border: 1px solid var(--border-strong);
    color: var(--text2);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  }
  .trouble {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13.5px;
    color: var(--bad);
    padding: 0 6px 10px;
  }
  .icon-button {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    color: var(--text2);
  }
  .icon-button:hover {
    background: var(--bg3);
  }

  @media (max-width: 860px) {
    .app {
      grid-template-columns: 1fr;
    }
    .side {
      position: fixed;
      inset: 0 auto 0 0;
      width: min(86vw, 320px);
      z-index: 20;
      transform: translateX(-100%);
      transition: transform 0.22s cubic-bezier(0.2, 0.8, 0.2, 1);
      box-shadow: 0 0 40px rgba(0, 0, 0, 0.5);
    }
    .drawer .side {
      transform: none;
    }
    .scrim {
      display: block;
      position: fixed;
      inset: 0;
      z-index: 10;
      background: rgba(0, 0, 0, 0.5);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s;
    }
    .drawer .scrim {
      opacity: 1;
      pointer-events: auto;
    }
    header {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 8px;
      padding-top: max(6px, env(safe-area-inset-top));
      border-bottom: 1px solid var(--border);
      background: var(--bg);
    }
    .title {
      flex: 1;
      min-width: 0;
      text-align: center;
      font-weight: 600;
      font-size: 15px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .column {
      padding: 0 14px;
    }
    .scroll .column {
      padding-top: 20px;
    }
  }
</style>
