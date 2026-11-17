<script>
  // Line icons on a 24 grid, drawn for this UI.
  let { name, size = 18 } = $props();

  const paths = {
    menu: 'M4 7h16M4 12h16M4 17h16',
    plus: 'M12 5v14M5 12h14',
    up: 'M12 19V5M5.5 11.5 12 5l6.5 6.5',
    chevron: 'M9.5 6l6 6-6 6',
    down: 'M6 9.5l6 6 6-6',
    check: 'M5 12.5l4.5 4.5L19 7.5',
    x: 'M6.5 6.5l11 11M17.5 6.5l-11 11',
    trash: 'M4.5 7h15M10 11v6M14 11v6M6.5 7l.9 11.2A2 2 0 0 0 9.4 20h5.2a2 2 0 0 0 2-1.8L17.5 7M9.5 7V4.5h5V7',
    file: 'M13.5 3.5H7.5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8.5zM13.5 3.5v5h5',
    folder: 'M3.5 7.5a2 2 0 0 1 2-2h3.8l2 2h7.2a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z',
    search: 'M10.5 17.5a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM20.5 20.5l-5-5',
    terminal: 'M4.5 7.5l5 4.5-5 4.5M12 17.5h7.5',
    globe: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z',
    pencil: 'M4.5 19.5h3.8L19 8.8a2.1 2.1 0 0 0-3-3L5.3 16.5zM14.5 7.5l2.8 2.8',
    bookmark: 'M7 3.5h10a1 1 0 0 1 1 1v16l-6-4-6 4v-16a1 1 0 0 1 1-1z',
    shield: 'M12 3l7.5 3v5.5c0 4.6-3.2 8.2-7.5 9.5-4.3-1.3-7.5-4.9-7.5-9.5V6z',
    chip: 'M8 6.5h8A1.5 1.5 0 0 1 17.5 8v8a1.5 1.5 0 0 1-1.5 1.5H8A1.5 1.5 0 0 1 6.5 16V8A1.5 1.5 0 0 1 8 6.5zM9.5 3v3.5M14.5 3v3.5M9.5 17.5V21M14.5 17.5V21M3 9.5h3.5M3 14.5h3.5M17.5 9.5H21M17.5 14.5H21',
    alert: 'M12 8v5M12 16.5v.5M10.3 4.2 3 17a2 2 0 0 0 1.7 3h14.6A2 2 0 0 0 21 17L13.7 4.2a2 2 0 0 0-3.4 0z',
    arrowdown: 'M12 5v14M5.5 12.5 12 19l6.5-6.5',
    signout: 'M14.5 4.5h3a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2h-3M10 16.5 5.5 12 10 7.5M5.5 12h10',
  };
</script>

{#if name === 'stop'}
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" /></svg>
{:else}
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d={paths[name]} />
  </svg>
{/if}
