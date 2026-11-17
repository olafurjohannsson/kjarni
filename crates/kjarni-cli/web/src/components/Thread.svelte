<script>
  import Reply from './Reply.svelte';
  import ToolStep from './ToolStep.svelte';
  import Icon from './Icon.svelte';
  import { hasContent } from '../lib/transcript.js';

  let { blocks, busy, summary, onanswer, workdir } = $props();

  // Between sending and the first words, and between a tool's result and the
  // next reply, show that Kjarni is on it.
  let working = $derived.by(() => {
    if (!busy) return false;
    const last = blocks.at(-1);
    if (!last) return true;
    if (last.type === 'assistant') return !hasContent(last);
    if (last.type === 'tool') return last.status !== 'waiting' && last.status !== 'running';
    return last.type === 'user';
  });
</script>

<div class="thread">
  {#if summary}
    <div class="summary">Earlier messages were summarized to keep the conversation in memory.</div>
  {/if}
  {#each blocks as block, i (i)}
    {#if block.type === 'user'}
      <div class="user"><div class="bubble">{block.text}</div></div>
    {:else if block.type === 'assistant'}
      {#if hasContent(block)}<Reply {block} />{/if}
    {:else if block.type === 'tool'}
      <ToolStep {block} {onanswer} {workdir} />
    {:else if block.type === 'notice'}
      <div class="note">{block.text}</div>
    {:else if block.type === 'error'}
      <div class="note error"><Icon name="alert" size={15} /> {block.text}</div>
    {:else if block.type === 'stopped'}
      <div class="note">Stopped.</div>
    {/if}
  {/each}
  {#if working}
    <div class="working"><span class="core"></span>Working</div>
  {/if}
</div>

<style>
  .thread {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 14px;
  }
  .user {
    display: flex;
    justify-content: flex-end;
    margin-top: 14px;
    animation: kj-rise 0.18s ease-out;
  }
  .user:first-child {
    margin-top: 0;
  }
  .bubble {
    max-width: min(85%, 600px);
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 18px 18px 6px 18px;
    padding: 10px 16px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .summary,
  .note {
    font-size: 13.5px;
    color: var(--muted);
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .summary {
    justify-content: center;
    padding-bottom: 8px;
    border-bottom: 1px dashed var(--border);
  }
  .error {
    color: var(--bad);
    align-items: flex-start;
  }
  .error :global(svg) {
    margin-top: 3px;
    flex: none;
  }
  .working {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14px;
    color: var(--muted);
  }
  .core {
    width: 7px;
    height: 15px;
    background: var(--core);
    border-radius: 1px;
    animation: kj-blink 1.2s step-end infinite;
  }
</style>
