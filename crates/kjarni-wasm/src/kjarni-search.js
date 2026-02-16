// kjarni-search.js — Drop-in semantic search widget
// Usage: <script src="kjarni-search.js"></script>
// Then:  KjarniSearch.init({ wasm: '/pkg/kjarni_wasm.js', model: '/model_q8.kjq', index: '/search.idx' })

const KjarniSearch = (() => {
  let search = null;
  let modal = null;
  let debounceTimer = null;
  let opts = {};

  async function init(options = {}) {
    opts = {
      wasm: options.wasm || '/pkg/kjarni_wasm.js',
      model: options.model || '/model_q8.kjq',
      index: options.index || '/kjarni-search.idx',
      limit: options.limit || 8,
      placeholder: options.placeholder || 'Search docs...',
      hotkey: options.hotkey !== false,
      mode: options.mode || 'hybrid', // hybrid | semantic | keyword
      ...options,
    };

    createModal();
    if (opts.hotkey) bindHotkey();
  }

  async function load() {
    if (search) return;

    setStatus('Loading search engine...');

    const wasm = await import(opts.wasm);
    await wasm.default();

    const [modelBuf, indexBuf] = await Promise.all([
      fetch(opts.model).then(r => r.arrayBuffer()),
      fetch(opts.index).then(r => r.arrayBuffer()),
    ]);

    setStatus('Initializing...');
    search = wasm.WasmSearch.load(
      new Uint8Array(modelBuf),
      new Uint8Array(indexBuf),
    );

    setStatus('');
    document.getElementById('kjarni-input').disabled = false;
    document.getElementById('kjarni-input').focus();
  }

  function doSearch(query) {
    if (!search || !query.trim()) {
      renderResults([]);
      return;
    }

    const start = performance.now();
    let results;

    if (opts.mode === 'semantic') {
      results = search.search_semantic(query, opts.limit);
    } else if (opts.mode === 'keyword') {
      results = search.search_keywords(query, opts.limit);
    } else {
      results = search.search(query, opts.limit);
    }

    const elapsed = (performance.now() - start).toFixed(0);
    renderResults(results, elapsed);
  }

  function renderResults(results, elapsed) {
    const container = document.getElementById('kjarni-results');
    if (!results || results.length === 0) {
      container.innerHTML = document.getElementById('kjarni-input')?.value
        ? '<div class="kjarni-empty">No results found</div>'
        : '';
      return;
    }

    container.innerHTML = results.map((r, i) => {
      const url = r.metadata?.source || '#';
      const title = r.metadata?.source?.split('/').pop() || `Result ${i + 1}`;
      const snippet = r.text.length > 200 ? r.text.slice(0, 200) + '...' : r.text;
      const score = (r.score * 100).toFixed(0);

      return `<a class="kjarni-result" href="${url}">
        <div class="kjarni-result-title">${title}<span class="kjarni-score">${score}%</span></div>
        <div class="kjarni-result-text">${snippet}</div>
      </a>`;
    }).join('');

    if (elapsed) {
      container.innerHTML += `<div class="kjarni-meta">${results.length} results in ${elapsed}ms · Powered by <a href="https://kjarni.ai" target="_blank">Kjarni</a></div>`;
    }
  }

  function setStatus(msg) {
    const el = document.getElementById('kjarni-status');
    if (el) el.textContent = msg;
  }

  function createModal() {
    if (document.getElementById('kjarni-modal')) return;

    const style = document.createElement('style');
    style.textContent = `
      .kjarni-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.5); z-index:9999; justify-content:center; align-items:flex-start; padding-top:min(20vh,120px); }
      .kjarni-overlay.open { display:flex; }
      .kjarni-modal { background:#fff; border-radius:12px; width:90%; max-width:620px; max-height:70vh; display:flex; flex-direction:column; box-shadow:0 20px 60px rgba(0,0,0,0.3); }
      .kjarni-header { padding:16px; border-bottom:1px solid #e5e7eb; display:flex; align-items:center; gap:8px; }
      .kjarni-header svg { width:20px; height:20px; color:#9ca3af; flex-shrink:0; }
      #kjarni-input { flex:1; border:none; outline:none; font-size:16px; background:none; }
      #kjarni-input:disabled { opacity:0.5; }
      .kjarni-kbd { font-size:11px; padding:2px 6px; border:1px solid #d1d5db; border-radius:4px; color:#6b7280; }
      #kjarni-results { overflow-y:auto; padding:8px; }
      .kjarni-result { display:block; padding:10px 12px; border-radius:8px; text-decoration:none; color:inherit; cursor:pointer; }
      .kjarni-result:hover { background:#f3f4f6; }
      .kjarni-result-title { font-weight:600; font-size:14px; margin-bottom:2px; display:flex; justify-content:space-between; }
      .kjarni-result-text { font-size:13px; color:#6b7280; line-height:1.4; }
      .kjarni-score { font-size:11px; color:#9ca3af; font-weight:400; }
      .kjarni-empty, #kjarni-status { padding:20px; text-align:center; color:#9ca3af; font-size:14px; }
      .kjarni-meta { padding:8px 12px; font-size:11px; color:#9ca3af; text-align:right; }
      .kjarni-meta a { color:#6b7280; }
      @media(prefers-color-scheme:dark) {
        .kjarni-modal { background:#1f2937; color:#f9fafb; }
        .kjarni-header { border-color:#374151; }
        .kjarni-result:hover { background:#374151; }
        .kjarni-result-text { color:#9ca3af; }
        .kjarni-kbd { border-color:#4b5563; color:#9ca3af; }
      }
    `;
    document.head.appendChild(style);

    const overlay = document.createElement('div');
    overlay.id = 'kjarni-modal';
    overlay.className = 'kjarni-overlay';
    overlay.innerHTML = `
      <div class="kjarni-modal" onclick="event.stopPropagation()">
        <div class="kjarni-header">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
          <input id="kjarni-input" type="text" placeholder="${opts.placeholder}" disabled />
          <span class="kjarni-kbd">esc</span>
        </div>
        <div id="kjarni-status"></div>
        <div id="kjarni-results"></div>
      </div>
    `;

    overlay.addEventListener('click', close);
    document.body.appendChild(overlay);

    document.getElementById('kjarni-input').addEventListener('input', (e) => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => doSearch(e.target.value), 150);
    });

    document.getElementById('kjarni-input').addEventListener('keydown', (e) => {
      if (e.key === 'Escape') close();
    });

    modal = overlay;
  }

  function bindHotkey() {
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        open();
      }
    });
  }

  async function open() {
    if (!modal) return;
    modal.classList.add('open');
    await load();
  }

  function close() {
    if (!modal) return;
    modal.classList.remove('open');
    document.getElementById('kjarni-input').value = '';
    document.getElementById('kjarni-results').innerHTML = '';
  }

  return { init, open, close };
})();

if (typeof window !== 'undefined') window.KjarniSearch = KjarniSearch;
