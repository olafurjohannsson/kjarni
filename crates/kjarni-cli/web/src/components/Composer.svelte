<script>
  import Icon from './Icon.svelte';

  // Where the user writes. Enter sends, Shift+Enter breaks the line; on a
  // phone, Enter breaks the line and the button sends. While Kjarni works, a
  // message sent goes to it before its next step.
  let { busy, disabled = false, onsend, onstop, value = $bindable('') } = $props();

  let field;
  const touch = typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches;

  function grow() {
    if (!field) return;
    field.style.height = 'auto';
    field.style.height = Math.min(field.scrollHeight, 240) + 'px';
  }

  // Measured on the next frame, once the textarea holds the new value.
  $effect(() => {
    value;
    requestAnimationFrame(grow);
  });

  function send() {
    const text = value.trim();
    if (!text || disabled) return;
    onsend(text);
    value = '';
  }

  function keydown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !touch && !e.isComposing) {
      e.preventDefault();
      send();
    }
  }

  export function focus() {
    field?.focus();
  }
</script>

<form class="composer" onsubmit={(e) => (e.preventDefault(), send())}>
  <textarea
    bind:this={field}
    bind:value
    onkeydown={keydown}
    rows="1"
    placeholder={busy ? 'Add something while Kjarni works' : 'Ask Kjarni anything'}
    aria-label="Message"
    {disabled}
  ></textarea>
  {#if busy}
    {#if value.trim()}
      <button type="submit" class="action" disabled={disabled} aria-label="Send" title="Send: Kjarni reads it before its next step"><Icon name="up" size={19} /></button>
    {/if}
    <button type="button" class="action stop" onclick={onstop} aria-label="Stop" title="Stop"><Icon name="stop" size={18} /></button>
  {:else}
    <button type="submit" class="action" disabled={!value.trim() || disabled} aria-label="Send" title="Send"><Icon name="up" size={19} /></button>
  {/if}
</form>

<style>
  .composer {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 8px 8px 8px 18px;
    transition: border-color 0.15s, box-shadow 0.15s;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
  }
  .composer:focus-within {
    border-color: var(--border-strong);
  }
  textarea {
    flex: 1;
    resize: none;
    border: none;
    outline: none;
    background: none;
    line-height: 1.5;
    padding: 7px 0;
    max-height: 240px;
    font-size: 16px;
  }
  textarea::placeholder {
    color: var(--subtle);
  }
  .action {
    flex: none;
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    background: var(--core);
    color: var(--ink);
    transition: background 0.12s, opacity 0.12s;
  }
  .action:not(:disabled):hover {
    background: var(--core-hover);
  }
  .action:disabled {
    background: var(--bg4);
    color: var(--subtle);
    cursor: default;
  }
  .stop {
    background: var(--text);
  }
  .stop:hover {
    background: #fff !important;
  }
</style>
