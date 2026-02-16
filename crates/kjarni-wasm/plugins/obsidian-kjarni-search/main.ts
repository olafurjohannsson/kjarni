import {
    App,
    Modal,
    Notice,
    Plugin,
    PluginSettingTab,
    Setting,
    TFile,
    debounce,
} from "obsidian";
import * as fs from "fs";
import * as path from "path";

// ─── Types ───────────────────────────────────────────────────────

interface KjarniSettings {
    chunkSize: number;
    chunkOverlap: number;
    searchLimit: number;
    rerankerEnabled: boolean;
}

const DEFAULT_SETTINGS: KjarniSettings = {
    chunkSize: 1000,
    chunkOverlap: 200,
    searchLimit: 10,
    rerankerEnabled: true,
};

interface SearchResult {
    score: number;
    text: string;
    metadata: Map<string, string>;
}

interface RerankResult {
    index: number;
    score: number;
    text: string;
}

// ─── WASM Loading ────────────────────────────────────────────────

// wasm-pack --target web generates ES module glue code.
// In Electron we load the .wasm bytes from disk and pass to init().
// The pkg/ folder must be inside the plugin directory.
//
// Plugin directory layout:
//   .obsidian/plugins/kjarni-search/
//   ├── main.js
//   ├── manifest.json
//   ├── styles.css
//   ├── pkg/
//   │   ├── kjarni_wasm.js
//   │   └── kjarni_wasm_bg.wasm
//   └── models/
//       ├── encoder.kjq        (all-MiniLM-L6-v2 quantized)
//       └── reranker.kjq       (ms-marco-MiniLM-L-6-v2 quantized)

let wasmModule: any = null;

function loadWasm(pluginDir: string): any {
    if (wasmModule) return wasmModule;

    const wasmJsPath = path.join(pluginDir, "pkg", "kjarni_wasm.js");
    wasmModule = require(wasmJsPath);
    return wasmModule;
}

// ─── Plugin ──────────────────────────────────────────────────────

export default class KjarniSearchPlugin extends Plugin {
    settings: KjarniSettings = DEFAULT_SETTINGS;
    pluginDir: string = "";
    wasm: any = null;
    search: any = null; // WasmSearch instance
    reranker: any = null; // WasmReranker instance
    indexReady: boolean = false;
    indexing: boolean = false;
    // Track file modification times for incremental indexing
    fileHashes: Map<string, number> = new Map();
    statusBarEl: HTMLElement | null = null;

    async onload() {
        await this.loadSettings();
        // Status bar for indexing progress
        this.statusBarEl = this.addStatusBarItem();
        this.statusBarEl.setText("");
        // Resolve plugin directory
        const basePath = (this.app.vault.adapter as any).basePath;
        this.pluginDir = path.join(
            basePath,
            ".obsidian",
            "plugins",
            "kjarni-search"
        );

        // Register search command (Cmd/Ctrl+K style)
        this.addCommand({
            id: "open-search",
            name: "Search vault",
            callback: () => this.openSearch(),
            hotkeys: [{ modifiers: ["Mod", "Shift"], key: "k" }],
        });

        // Register reindex command
        this.addCommand({
            id: "reindex-vault",
            name: "Reindex vault",
            callback: () => this.reindexVault(),
        });

        // Add ribbon icon
        this.addRibbonIcon("search", "Kjarni Search", () => this.openSearch());

        // Settings tab
        this.addSettingTab(new KjarniSettingTab(this.app, this));

        // Initialize in background
        this.initializeAsync();

        // Watch for file changes for incremental indexing
        this.registerEvent(
            this.app.vault.on("modify", (file) => {
                if (file instanceof TFile && file.extension === "md") {
                    this.debouncedReindexFile(file);
                }
            })
        );
        this.registerEvent(
            this.app.vault.on("create", (file) => {
                if (file instanceof TFile && file.extension === "md") {
                    this.debouncedReindexFile(file);
                }
            })
        );
        this.registerEvent(
            this.app.vault.on("delete", (file) => {
                if (file instanceof TFile && file.extension === "md") {
                    // Full reindex needed on delete (index doesn't support removal)
                    // Could be optimized later with tombstones
                    this.scheduleFullReindex();
                }
            })
        );
    }

    onunload() {
        this.search = null;
        this.reranker = null;
        wasmModule = null;
    }

    // ─── Initialization ──────────────────────────────────────────

    async initializeAsync() {
        try {
            // Check if WASM and models exist
            if (!this.checkFiles()) return;

            new Notice("Kjarni: Loading search engine...");
            this.wasm = loadWasm(this.pluginDir);

            // Load reranker if enabled
            if (this.settings.rerankerEnabled) {
                const rerankerPath = path.join(
                    this.pluginDir,
                    "models",
                    "reranker.kjq"
                );
                if (fs.existsSync(rerankerPath)) {
                    const rerankerBytes = fs.readFileSync(rerankerPath);
                    this.reranker = this.wasm.WasmReranker.load(
                        new Uint8Array(rerankerBytes)
                    );
                }
            }

            // Check for existing index
            const indexPath = path.join(this.pluginDir, "index.idx");
            if (fs.existsSync(indexPath)) {
                await this.loadExistingIndex(indexPath);
            } else {
                await this.buildFullIndex();
            }
        } catch (e) {
            console.error("Kjarni: Initialization failed:", e);
            new Notice(`Kjarni: Failed to initialize — ${e}`);
        }
    }

    checkFiles(): boolean {
        const wasmJs = path.join(this.pluginDir, "pkg", "kjarni_wasm.js");
        const wasmBin = path.join(
            this.pluginDir,
            "pkg",
            "kjarni_wasm_bg.wasm"
        );
        const encoder = path.join(this.pluginDir, "models", "encoder.kjq");

        const missing: string[] = [];
        if (!fs.existsSync(wasmJs)) missing.push("pkg/kjarni_wasm.js");
        if (!fs.existsSync(wasmBin))
            missing.push("pkg/kjarni_wasm_bg.wasm");
        if (!fs.existsSync(encoder)) missing.push("models/encoder.kjq");

        if (missing.length > 0) {
            new Notice(
                `Kjarni: Missing files in plugin directory:\n${missing.join(
                    "\n"
                )}\n\nSee plugin README for setup instructions.`,
                10000
            );
            return false;
        }
        return true;
    }

    async loadExistingIndex(indexPath: string) {
        try {
            const indexBytes = fs.readFileSync(indexPath);
            const encoderBytes = fs.readFileSync(
                path.join(this.pluginDir, "models", "encoder.kjq")
            );
            this.search = this.wasm.WasmSearch.load(
                new Uint8Array(encoderBytes),
                new Uint8Array(indexBytes)
            );
            this.indexReady = true;
            new Notice(
                `Kjarni: Loaded index (${this.search.doc_count()} chunks)`
            );

            // Check for files modified since index was built
            await this.incrementalUpdate(indexPath);
        } catch (e) {
            console.error("Kjarni: Failed to load index, rebuilding:", e);
            await this.buildFullIndex();
        }
    }

    async buildFullIndex() {
        if (this.indexing) return;
        this.indexing = true;

        try {
            const encoderPath = path.join(
                this.pluginDir,
                "models",
                "encoder.kjq"
            );
            const encoderBytes = fs.readFileSync(encoderPath);

            const builder = this.wasm.WasmIndexBuilder.new(
                new Uint8Array(encoderBytes)
            );

            const files = this.app.vault
                .getMarkdownFiles()
                .sort((a, b) => a.path.localeCompare(b.path));

            const total = files.length;
            let indexed = 0;
            let totalChunks = 0;

            this.statusBarEl?.setText(`Kjarni: indexing 0/${total}...`);
            const startTime = Date.now();

            for (const file of files) {
                const text = await this.app.vault.cachedRead(file);
                if (text.trim().length === 0) continue;

                try {
                    const chunks = builder.add_file(text, file.path);
                    totalChunks += chunks;
                } catch (e) {
                    console.warn(`Kjarni: Skipped ${file.path}: ${e}`);
                }
                indexed++;

                // Update status bar every 10 files + yield to UI thread
                if (indexed % 10 === 0) {
                    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
                    const rate = (indexed / (Date.now() - startTime) * 1000).toFixed(1);
                    this.statusBarEl?.setText(
                        `Kjarni: ${indexed}/${total} files (${totalChunks} chunks) · ${elapsed}s · ${rate} files/s`
                    );
                    await sleep(0);
                }
            }

            // Save index
            const indexBytes = builder.finish();
            const indexPath = path.join(this.pluginDir, "index.idx");
            fs.writeFileSync(indexPath, Buffer.from(indexBytes));

            // Load the search instance
            this.search = this.wasm.WasmSearch.load(
                new Uint8Array(encoderBytes),
                new Uint8Array(indexBytes)
            );
            this.indexReady = true;

            // Store file modification times
            for (const file of files) {
                this.fileHashes.set(file.path, file.stat.mtime);
            }

            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            this.statusBarEl?.setText(
                `Kjarni: ${totalChunks} chunks · ${indexed} files · ${elapsed}s`
            );
            new Notice(
                `Kjarni: Indexed ${totalChunks} chunks from ${indexed} files in ${elapsed}s`
            );
        } catch (e) {
            console.error("Kjarni: Indexing failed:", e);
            new Notice(`Kjarni: Indexing failed — ${e}`);
        } finally {
            this.indexing = false;
        }
    }

    // ─── Incremental Indexing ────────────────────────────────────

    async incrementalUpdate(indexPath: string) {
        const indexStat = fs.statSync(indexPath);
        const indexMtime = indexStat.mtimeMs;

        const changedFiles = this.app.vault
            .getMarkdownFiles()
            .filter((f) => f.stat.mtime > indexMtime);

        if (changedFiles.length > 0) {
            // If more than 20% of vault changed, do full reindex
            const total = this.app.vault.getMarkdownFiles().length;
            if (changedFiles.length > total * 0.2) {
                new Notice(
                    `Kjarni: ${changedFiles.length} files changed, rebuilding index...`
                );
                await this.buildFullIndex();
            } else {
                new Notice(
                    `Kjarni: ${changedFiles.length} files changed since last index. Use "Reindex vault" to update.`
                );
            }
        }
    }

    debouncedReindexFile = debounce(
        (file: TFile) => {
            // For now, just note that the index is stale
            // Full incremental per-file update would require index rebuild
            // since SearchIndex doesn't support in-place updates well
            console.log(`Kjarni: File modified: ${file.path} (index may be stale)`);
        },
        5000,
        true
    );

    scheduleFullReindex = debounce(
        () => {
            if (this.indexReady) {
                new Notice(
                    "Kjarni: File deleted. Reindex recommended (Cmd+P → Kjarni: Reindex vault)"
                );
            }
        },
        10000,
        true
    );

    async reindexVault() {
        if (this.indexing) {
            new Notice("Kjarni: Already indexing...");
            return;
        }
        await this.buildFullIndex();
    }

    // ─── Search ──────────────────────────────────────────────────

    openSearch() {
        if (!this.indexReady) {
            new Notice("Kjarni: Search index not ready yet. Please wait for indexing to complete.");
            return;
        }
        new SearchModal(this.app, this).open();
    }

    doSearch(query: string): SearchResult[] {
        if (!this.search || !query.trim()) return [];
        try {
            return this.search.search(query, this.settings.searchLimit * 2);
        } catch (e) {
            console.error("Kjarni: Search failed:", e);
            return [];
        }
    }

    doRerank(query: string, results: SearchResult[]): RerankResult[] {
        if (!this.reranker || results.length === 0) {
            // No reranker — return results as-is in RerankResult format
            return results.map((r, i) => ({
                index: i,
                score: r.score,
                text: r.text,
            }));
        }
        try {
            const docs = results.map((r) => r.text);
            return this.reranker.rerank(query, docs, this.settings.searchLimit);
        } catch (e) {
            console.error("Kjarni: Rerank failed:", e);
            return results.map((r, i) => ({
                index: i,
                score: r.score,
                text: r.text,
            }));
        }
    }

    // ─── Settings ────────────────────────────────────────────────

    async loadSettings() {
        this.settings = Object.assign(
            {},
            DEFAULT_SETTINGS,
            await this.loadData()
        );
    }

    async saveSettings() {
        await this.saveData(this.settings);
    }
}

// ─── Search Modal ────────────────────────────────────────────────

class SearchModal extends Modal {
    plugin: KjarniSearchPlugin;
    inputEl: HTMLInputElement | null = null;
    resultsEl: HTMLElement | null = null;
    debounceTimer: ReturnType<typeof setTimeout> | null = null;
    lastResults: SearchResult[] = [];

    constructor(app: App, plugin: KjarniSearchPlugin) {
        super(app);
        this.plugin = plugin;
    }

    onOpen() {
        const { contentEl, modalEl } = this;

        // Style the modal
        modalEl.addClass("kjarni-search-modal");

        // Header with search input
        const header = contentEl.createDiv({ cls: "kjarni-header" });

        const icon = header.createEl("span", { cls: "kjarni-icon" });
        icon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>`;

        this.inputEl = header.createEl("input", {
            cls: "kjarni-input",
            attr: {
                type: "text",
                placeholder: "Search your vault...",
                spellcheck: "false",
            },
        });

        const kbd = header.createEl("span", { cls: "kjarni-kbd" });
        kbd.setText("esc");

        // Results container
        this.resultsEl = contentEl.createDiv({ cls: "kjarni-results" });

        // Status line
        contentEl.createDiv({ cls: "kjarni-status" });

        // Bind events
        this.inputEl.addEventListener("input", () => {
            if (this.debounceTimer) clearTimeout(this.debounceTimer);
            this.debounceTimer = setTimeout(() => this.runSearch(), 150);
        });

        this.inputEl.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                this.close();
            } else if (e.key === "Enter") {
                // Open first result
                this.openResult(0);
            }
        });

        // Focus input
        this.inputEl.focus();
    }

    runSearch() {
        const query = this.inputEl?.value?.trim();
        if (!query || !this.resultsEl) {
            if (this.resultsEl) this.resultsEl.empty();
            return;
        }

        const t0 = performance.now();
        const results = this.plugin.doSearch(query);
        const searchMs = performance.now() - t0;

        const t1 = performance.now();
        const reranked = this.plugin.doRerank(query, results);
        const rerankMs = performance.now() - t1;

        this.lastResults = results;
        this.renderResults(reranked, results, searchMs, rerankMs);
    }

    renderResults(
        reranked: RerankResult[],
        original: SearchResult[],
        searchMs: number,
        rerankMs: number
    ) {
        if (!this.resultsEl) return;
        this.resultsEl.empty();

        if (reranked.length === 0) {
            this.resultsEl.createDiv({
                cls: "kjarni-empty",
                text: "No results found",
            });
            return;
        }

        reranked.forEach((r, displayIndex) => {
            const origResult = original[r.index];
            const meta =
                origResult?.metadata instanceof Map
                    ? origResult.metadata
                    : new Map();
            const source = meta.get("source") || "unknown";
            const snippet =
                r.text.length > 200 ? r.text.slice(0, 200) + "..." : r.text;

            const resultEl = this.resultsEl!.createDiv({
                cls: "kjarni-result",
            });
            resultEl.addEventListener("click", () =>
                this.openResult(displayIndex)
            );

            const titleRow = resultEl.createDiv({ cls: "kjarni-result-title" });
            titleRow.createEl("span", {
                text: source.replace(/\.md$/, ""),
            });
            titleRow.createEl("span", {
                cls: "kjarni-score",
                text: r.score.toFixed(3),
            });

            resultEl.createDiv({
                cls: "kjarni-result-text",
                text: snippet,
            });
        });

        // Status line
        const statusEl = this.contentEl.querySelector(".kjarni-status");
        if (statusEl) {
            statusEl.textContent = `${reranked.length} results · ${searchMs.toFixed(0)}ms search · ${rerankMs.toFixed(0)}ms rerank`;
        }
    }

    openResult(index: number) {
        const results = this.plugin.doRerank(
            this.inputEl?.value || "",
            this.lastResults
        );
        const result = results[index];
        if (!result) return;

        const origResult = this.lastResults[result.index];
        const meta =
            origResult?.metadata instanceof Map
                ? origResult.metadata
                : new Map();
        const source = meta.get("source");

        if (source) {
            const file = this.app.vault.getAbstractFileByPath(source);
            if (file instanceof TFile) {
                this.app.workspace.openLinkText(source, "", false);
                this.close();
            }
        }
    }

    onClose() {
        if (this.debounceTimer) clearTimeout(this.debounceTimer);
        this.contentEl.empty();
    }
}

// ─── Settings Tab ────────────────────────────────────────────────

class KjarniSettingTab extends PluginSettingTab {
    plugin: KjarniSearchPlugin;

    constructor(app: App, plugin: KjarniSearchPlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        const { containerEl } = this;
        containerEl.empty();
        containerEl.createEl("h2", { text: "Kjarni Search" });

        // Index info
        if (this.plugin.indexReady && this.plugin.search) {
            containerEl.createEl("p", {
                text: `Index: ${this.plugin.search.doc_count()} chunks indexed`,
                cls: "setting-item-description",
            });
        }

        new Setting(containerEl)
            .setName("Search results limit")
            .setDesc("Maximum number of results to show")
            .addSlider((slider) =>
                slider
                    .setLimits(5, 30, 5)
                    .setValue(this.plugin.settings.searchLimit)
                    .setDynamicTooltip()
                    .onChange(async (value) => {
                        this.plugin.settings.searchLimit = value;
                        await this.plugin.saveSettings();
                    })
            );

        new Setting(containerEl)
            .setName("Reranker")
            .setDesc(
                "Use cross-encoder reranking for better result quality (requires reranker.kjq model)"
            )
            .addToggle((toggle) =>
                toggle
                    .setValue(this.plugin.settings.rerankerEnabled)
                    .onChange(async (value) => {
                        this.plugin.settings.rerankerEnabled = value;
                        await this.plugin.saveSettings();
                    })
            );

        new Setting(containerEl)
            .setName("Chunk size")
            .setDesc(
                "Maximum characters per chunk when indexing (default 1000)"
            )
            .addText((text) =>
                text
                    .setValue(String(this.plugin.settings.chunkSize))
                    .onChange(async (value) => {
                        const num = parseInt(value);
                        if (!isNaN(num) && num > 100) {
                            this.plugin.settings.chunkSize = num;
                            await this.plugin.saveSettings();
                        }
                    })
            );

        new Setting(containerEl)
            .setName("Reindex vault")
            .setDesc("Rebuild the search index from scratch")
            .addButton((button) =>
                button.setButtonText("Reindex").onClick(async () => {
                    await this.plugin.reindexVault();
                })
            );
    }
}

// ─── Helpers ─────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}