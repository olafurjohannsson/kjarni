<script>
  import Mark from './Mark.svelte';
  import Icon from './Icon.svelte';

  let { status, onpick } = $props();

  const hour = new Date().getHours();
  const greeting = hour < 5 ? 'Up late?' : hour < 12 ? 'Good morning.' : hour < 18 ? 'Good afternoon.' : 'Good evening.';

  const ideas = [
    { icon: 'bookmark', title: 'What do you know about me?', prompt: 'What do you remember about me?' },
    { icon: 'search', title: 'Find something in my files', prompt: 'Search my files for ' },
    { icon: 'globe', title: 'Read a page for me', prompt: 'Summarize this page in five bullet points: ' },
    { icon: 'pencil', title: 'Keep a note', prompt: 'Start a note called groceries.md with milk, eggs and coffee.' },
  ];
</script>

<div class="welcome">
  <Mark size={44} />
  <h1>{greeting}</h1>
  <p>
    Your assistant, running on your own computer{#if status?.model}&nbsp;with <span class="model">{status.model}</span>{/if}.
    Nothing you say here leaves it.
  </p>
  <div class="ideas">
    {#each ideas as idea}
      <button onclick={() => onpick(idea.prompt)}>
        <Icon name={idea.icon} size={17} />
        <span>{idea.title}</span>
      </button>
    {/each}
  </div>
</div>

<style>
  .welcome {
    display: grid;
    justify-items: center;
    text-align: center;
    gap: 14px;
    padding: 8vh 0 24px;
    animation: kj-rise 0.3s ease-out;
  }
  h1 {
    font-size: clamp(28px, 5vw, 38px);
    font-weight: 700;
    letter-spacing: -0.035em;
    line-height: 1.1;
    margin-top: 10px;
  }
  p {
    color: var(--muted);
    max-width: 440px;
    font-size: 15.5px;
  }
  .model {
    white-space: nowrap;
    color: var(--text2);
  }
  .ideas {
    margin-top: 22px;
    width: 100%;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
  }
  .ideas button {
    display: flex;
    align-items: center;
    gap: 11px;
    text-align: left;
    padding: 14px 16px;
    border-radius: var(--radius);
    background: var(--bg2);
    border: 1px solid var(--border);
    color: var(--text2);
    font-size: 14.5px;
    transition: border-color 0.15s, background 0.15s, transform 0.15s;
  }
  .ideas button :global(svg) {
    color: var(--accent);
    flex: none;
  }
  .ideas button:hover {
    border-color: var(--border-strong);
    background: var(--bg3);
    transform: translateY(-1px);
  }
  @media (max-width: 560px) {
    .ideas {
      grid-template-columns: 1fr;
    }
    .welcome {
      padding-top: 4vh;
    }
  }
</style>
