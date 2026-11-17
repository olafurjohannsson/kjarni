<script>
  // The Kjarni mark: a prompt and the orange core. The core blinks while
  // Kjarni is working.
  let { size = 24, blink = false } = $props();
</script>

<svg width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden="true">
  <polyline points="8,15 26,32 8,49" stroke="#F4F2ED" stroke-width="10" stroke-linecap="square" />
  <rect class:blink x="35" y="9" width="22" height="46" fill="#E4572E" />
</svg>

<style>
  .blink {
    animation: kj-blink 1.2s step-end infinite;
  }
  svg {
    flex: none;
    display: block;
  }
</style>
