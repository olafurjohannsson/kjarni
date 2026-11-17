<script>
  import { renderMarkdown } from '../lib/markdown.js';

  // One reply from the model: its reasoning, folded, then its answer.
  let { block } = $props();

  let showReasoning = $state(false);
  let html = $derived(renderMarkdown(block.text));
</script>

<div class="reply">
  {#if block.thinking || block.reasoning}
    <button class="thought" class:live={block.thinking} onclick={() => (showReasoning = !showReasoning)} aria-expanded={showReasoning}>
      <span class="core"></span>
      {block.thinking ? 'Thinking' : 'Thought it through'}
      <span class="hint">{showReasoning ? 'hide' : 'show'}</span>
    </button>
    {#if showReasoning}
      <div class="reasoning">{block.reasoning}</div>
    {/if}
  {/if}
  {#if block.text}
    <div class="prose">{@html html}</div>
  {/if}
</div>

<style>
  .reply {
    display: grid;
    gap: 8px;
    animation: kj-rise 0.18s ease-out;
  }
  .thought {
    justify-self: start;
    display: inline-flex;
    align-items: center;
    gap: 9px;
    font-size: 13.5px;
    color: var(--muted);
    padding: 3px 8px 3px 6px;
    margin-left: -6px;
    border-radius: 8px;
  }
  .thought:hover {
    background: var(--bg2);
    color: var(--text2);
  }
  .core {
    width: 6px;
    height: 13px;
    background: var(--subtle);
    border-radius: 1px;
  }
  .live .core {
    background: var(--core);
    animation: kj-blink 1.2s step-end infinite;
  }
  .hint {
    color: var(--subtle);
    font-size: 12.5px;
  }
  .reasoning {
    white-space: pre-wrap;
    font-size: 14px;
    line-height: 1.6;
    color: var(--muted);
    border-left: 2px solid var(--border-strong);
    padding: 2px 0 2px 14px;
    max-height: 360px;
    overflow: auto;
  }
</style>
