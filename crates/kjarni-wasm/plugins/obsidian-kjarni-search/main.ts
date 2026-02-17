import {
	App,
	Modal,
	Notice,
	Plugin,
	PluginSettingTab,
	Setting,
	TFile,
	requestUrl,
} from "obsidian";
import * as fs from "fs";
import * as path from "path";

// ─── Config ──────────────────────────────────────────────────────

const MODEL_BASE_URL = "https://kjarni.ai/models";
const MODELS = {
	encoder: {
		filename: "encoder.kjq",
		url: `${MODEL_BASE_URL}/all-MiniLM-L6-v2-q8.kjq`,
		size: "22 MB",
	},
	reranker: {
		filename: "reranker.kjq",
		url: `${MODEL_BASE_URL}/ms-marco-MiniLM-L-6-v2-q8.kjq`,
		size: "22 MB",
	},
};

const ENCODER_WORKER_COUNT = 4;
const FILE_UPDATE_DEBOUNCE_MS = 3000;
const DELETE_DEBOUNCE_MS = 5000;

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

interface SearchResultItem {
	score: number;
	text: string;
	source: string;
	metadata: Record<string, string>;
}

interface RerankResultItem {
	index: number;
	score: number;
	text: string;
}

interface EncodedChunk {
	text: string;
	embedding: number[];
	source: string;
	chunk_index: number;
}

type PendingResolve = (value: any) => void;
type PendingReject = (reason: any) => void;

// ─── Plugin ──────────────────────────────────────────────────────

export default class KjarniSearchPlugin extends Plugin {
	settings: KjarniSettings = DEFAULT_SETTINGS;
	pluginDir: string = "";

	// Search worker (persistent)
	searchWorker: Worker | null = null;
	hasReranker: boolean = false;
	indexReady: boolean = false;
	indexing: boolean = false;
	statusBarEl: HTMLElement | null = null;

	// Message routing
	private nextId: number = 1;
	private searchPending: Map<number, { resolve: PendingResolve; reject: PendingReject }> = new Map();

	// Encoder workers (created for indexing, terminated after)
	private encoderWorkers: Worker[] = [];
	private encoderPending: Map<number, { resolve: PendingResolve; reject: PendingReject }> = new Map();

	// Incremental update queues
	private pendingUpdates: Set<string> = new Set();
	private pendingDeletes: Set<string> = new Set();
	private updateTimer: ReturnType<typeof setTimeout> | null = null;
	private deleteTimer: ReturnType<typeof setTimeout> | null = null;

	// Raw bytes (read once, reused for encoder workers)
	private wasmBytesCache: ArrayBuffer | null = null;
	private encoderBytesCache: ArrayBuffer | null = null;

	async onload() {
		await this.loadSettings();

		this.statusBarEl = this.addStatusBarItem();
		this.statusBarEl.setText("");

		const basePath = (this.app.vault.adapter as any).basePath;
		this.pluginDir = path.join(
			basePath, ".obsidian", "plugins", "kjarni-search"
		);

		this.addCommand({
			id: "open-search",
			name: "Search vault",
			callback: () => this.openSearch(),
			hotkeys: [{ modifiers: ["Mod", "Shift"], key: "k" }],
		});

		this.addCommand({
			id: "reindex-vault",
			name: "Reindex vault",
			callback: () => this.reindexVault(),
		});

		this.addRibbonIcon("search", "Kjarni Search", () => this.openSearch());
		this.addSettingTab(new KjarniSettingTab(this.app, this));

		// File watchers
		this.registerEvent(
			this.app.vault.on("modify", (file) => {
				if (file instanceof TFile && file.extension === "md")
					this.queueFileUpdate(file.path);
			})
		);
		this.registerEvent(
			this.app.vault.on("create", (file) => {
				if (file instanceof TFile && file.extension === "md")
					this.queueFileUpdate(file.path);
			})
		);
		this.registerEvent(
			this.app.vault.on("delete", (file) => {
				if (file instanceof TFile && file.extension === "md")
					this.queueFileDelete(file.path);
			})
		);
		this.registerEvent(
			this.app.vault.on("rename", (file, oldPath) => {
				if (file instanceof TFile && file.extension === "md") {
					this.queueFileDelete(oldPath);
					this.queueFileUpdate(file.path);
				}
			})
		);

		this.initializeAsync();
	}

	onunload() {
		if (this.updateTimer) clearTimeout(this.updateTimer);
		if (this.deleteTimer) clearTimeout(this.deleteTimer);
		this.stopEncoderPool();
		this.searchWorker?.terminate();
		this.searchWorker = null;
	}

	// ─── Worker Helpers ──────────────────────────────────────────

	private createBlobWorker(filename: string): Worker {
		const filePath = path.join(this.pluginDir, filename);
		const code = fs.readFileSync(filePath, "utf8");
		const blob = new Blob([code], { type: "application/javascript" });
		const url = URL.createObjectURL(blob);
		return new Worker(url);
	}

	private postToSearch(msg: any, transfer?: Transferable[]): Promise<any> {
		return new Promise((resolve, reject) => {
			if (!this.searchWorker) {
				reject(new Error("Search worker not started"));
				return;
			}
			const id = this.nextId++;
			msg.id = id;
			this.searchPending.set(id, { resolve, reject });
			if (transfer) {
				this.searchWorker.postMessage(msg, transfer);
			} else {
				this.searchWorker.postMessage(msg);
			}
		});
	}

	private postToEncoder(worker: Worker, msg: any, transfer?: Transferable[]): Promise<any> {
		return new Promise((resolve, reject) => {
			const id = this.nextId++;
			msg.id = id;
			this.encoderPending.set(id, { resolve, reject });
			if (transfer) {
				worker.postMessage(msg, transfer);
			} else {
				worker.postMessage(msg);
			}
		});
	}

	// ─── Search Worker ───────────────────────────────────────────

	private startSearchWorker() {
		this.searchWorker = this.createBlobWorker("worker.js");

		this.searchWorker.onmessage = (e: MessageEvent) => {
			const msg = e.data;
			if (msg.id && this.searchPending.has(msg.id)) {
				const { resolve, reject } = this.searchPending.get(msg.id)!;
				this.searchPending.delete(msg.id);
				if (msg.type === "error") {
					reject(new Error(msg.error));
				} else {
					resolve(msg);
				}
				return;
			}
			if (msg.type === "error") {
				console.error("Kjarni search worker:", msg.error);
			}
		};

		this.searchWorker.onerror = (err: ErrorEvent) => {
			console.error("Kjarni search worker crashed:", err);
			new Notice(`Kjarni: Worker error — ${err.message}`);
		};
	}

	// ─── Encoder Pool ────────────────────────────────────────────

	private async startEncoderPool(): Promise<void> {
		if (!this.wasmBytesCache || !this.encoderBytesCache) {
			throw new Error("WASM/encoder bytes not loaded");
		}

		this.encoderWorkers = [];

		for (let i = 0; i < ENCODER_WORKER_COUNT; i++) {
			const worker = this.createBlobWorker("encoder-worker.js");

			worker.onmessage = (e: MessageEvent) => {
				const msg = e.data;

				// Progress updates (fire-and-forget, no id)
				if (msg.type === "encode_progress") {
					// Handled via progress listener in buildFullIndex
					return;
				}

				if (msg.id && this.encoderPending.has(msg.id)) {
					const { resolve, reject } = this.encoderPending.get(msg.id)!;
					this.encoderPending.delete(msg.id);
					if (msg.type === "error") {
						reject(new Error(msg.error));
					} else {
						resolve(msg);
					}
				}
			};

			worker.onerror = (err: ErrorEvent) => {
				console.error(`Kjarni encoder worker ${i} crashed:`, err);
			};

			// Initialize — each worker gets its own copy of the bytes
			const wasmCopy = this.wasmBytesCache.slice(0);
			const encoderCopy = this.encoderBytesCache.slice(0);

			await this.postToEncoder(worker, {
				type: "init",
				wasmBytes: wasmCopy,
				encoderBytes: encoderCopy,
			}, [wasmCopy, encoderCopy]);

			this.encoderWorkers.push(worker);
		}
	}

	private stopEncoderPool() {
		for (const worker of this.encoderWorkers) {
			worker.terminate();
		}
		this.encoderWorkers = [];
	}

	// ─── Model Download ──────────────────────────────────────────

	async downloadModel(
		modelInfo: { filename: string; url: string; size: string },
		destDir: string
	): Promise<string> {
		const destPath = path.join(destDir, modelInfo.filename);
		if (fs.existsSync(destPath)) return destPath;

		this.statusBarEl?.setText(
			`Kjarni: downloading ${modelInfo.filename} (${modelInfo.size})...`
		);
		new Notice(`Kjarni: Downloading ${modelInfo.filename} (${modelInfo.size})...`);

		try {
			const response = await requestUrl({ url: modelInfo.url, method: "GET" });
			fs.mkdirSync(destDir, { recursive: true });
			fs.writeFileSync(destPath, Buffer.from(response.arrayBuffer));
			return destPath;
		} catch (e) {
			if (fs.existsSync(destPath)) fs.unlinkSync(destPath);
			throw new Error(`Failed to download ${modelInfo.filename}: ${e}`);
		}
	}

	async ensureModels(): Promise<boolean> {
		const modelsDir = path.join(this.pluginDir, "models");
		try {
			await this.downloadModel(MODELS.encoder, modelsDir);
			if (this.settings.rerankerEnabled) {
				await this.downloadModel(MODELS.reranker, modelsDir);
			}
			this.statusBarEl?.setText("");
			return true;
		} catch (e) {
			console.error("Kjarni: Model download failed:", e);
			new Notice(`Kjarni: ${e}`, 10000);
			return false;
		}
	}

	// ─── Initialization ──────────────────────────────────────────

	async initializeAsync() {
		try {
			if (!this.checkFiles()) return;
			if (!(await this.ensureModels())) return;

			this.statusBarEl?.setText("Kjarni: loading engine...");

			// Read bytes once, cache for encoder pool
			const wasmBuf = fs.readFileSync(
				path.join(this.pluginDir, "pkg", "kjarni_wasm_bg.wasm")
			);
			const encoderBuf = fs.readFileSync(
				path.join(this.pluginDir, "models", "encoder.kjq")
			);

			this.wasmBytesCache = wasmBuf.buffer.slice(
				wasmBuf.byteOffset, wasmBuf.byteOffset + wasmBuf.byteLength
			);
			this.encoderBytesCache = encoderBuf.buffer.slice(
				encoderBuf.byteOffset, encoderBuf.byteOffset + encoderBuf.byteLength
			);

			// Start search worker
			this.startSearchWorker();

			let rerankerBytes: ArrayBuffer | null = null;
			if (this.settings.rerankerEnabled) {
				const rerankerPath = path.join(this.pluginDir, "models", "reranker.kjq");
				if (fs.existsSync(rerankerPath)) {
					const buf = fs.readFileSync(rerankerPath);
					rerankerBytes = buf.buffer.slice(
						buf.byteOffset, buf.byteOffset + buf.byteLength
					);
				}
			}

			let indexBytes: ArrayBuffer | null = null;
			let indexMtime = 0;
			const indexPath = path.join(this.pluginDir, "index.idx");
			if (fs.existsSync(indexPath)) {
				const buf = fs.readFileSync(indexPath);
				indexBytes = buf.buffer.slice(
					buf.byteOffset, buf.byteOffset + buf.byteLength
				);
				indexMtime = fs.statSync(indexPath).mtimeMs;
			}

			// Init search worker with all bytes
			const transferList: ArrayBuffer[] = [
				this.wasmBytesCache.slice(0),
				this.encoderBytesCache.slice(0),
			];

			const initMsg: any = {
				type: "init",
				wasmBytes: transferList[0],
				encoderBytes: transferList[1],
			};

			if (rerankerBytes) {
				const copy = rerankerBytes.slice(0);
				initMsg.rerankerBytes = copy;
				transferList.push(copy);
			}
			if (indexBytes) {
				const copy = indexBytes.slice(0);
				initMsg.indexBytes = copy;
				transferList.push(copy);
			}

			const result = await this.postToSearch(initMsg, transferList);
			this.hasReranker = result.hasReranker;

			if (result.hasIndex) {
				this.indexReady = true;
				this.statusBarEl?.setText(`Kjarni: ${result.docCount} chunks`);
				new Notice(`Kjarni: Loaded index (${result.docCount} chunks)`);
				await this.catchUpChangedFiles(indexMtime);
			} else {
				await this.buildFullIndex();
			}
		} catch (e) {
			console.error("Kjarni: Initialization failed:", e);
			new Notice(`Kjarni: Failed to initialize — ${e}`);
			this.statusBarEl?.setText("Kjarni: init failed");
		}
	}

	checkFiles(): boolean {
		const wasmBin = path.join(this.pluginDir, "pkg", "kjarni_wasm_bg.wasm");
		const workerJs = path.join(this.pluginDir, "worker.js");
		const encoderJs = path.join(this.pluginDir, "encoder-worker.js");

		const missing: string[] = [];
		if (!fs.existsSync(wasmBin)) missing.push("pkg/kjarni_wasm_bg.wasm");
		if (!fs.existsSync(workerJs)) missing.push("worker.js");
		if (!fs.existsSync(encoderJs)) missing.push("encoder-worker.js");

		if (missing.length > 0) {
			new Notice(
				`Kjarni: Missing files:\n${missing.join("\n")}\n\nReinstall the plugin.`,
				10000
			);
			return false;
		}
		return true;
	}

	// ─── Full Index Build (Parallel) ─────────────────────────────

	async buildFullIndex() {
		if (this.indexing) return;
		this.indexing = true;

		try {
			const startTime = Date.now();

			// 1. Read all files on main thread
			const files = this.app.vault
				.getMarkdownFiles()
				.sort((a, b) => a.path.localeCompare(b.path));

			this.statusBarEl?.setText(`Kjarni: reading ${files.length} files...`);

			const fileData: { text: string; path: string }[] = [];
			for (const file of files) {
				const text = await this.app.vault.cachedRead(file);
				if (text.trim().length > 0) {
					fileData.push({ text, path: file.path });
				}
			}

			const total = fileData.length;
			if (total === 0) {
				this.statusBarEl?.setText("Kjarni: no files to index");
				return;
			}

			// 2. Start encoder pool
			this.statusBarEl?.setText(`Kjarni: starting ${ENCODER_WORKER_COUNT} encoder workers...`);
			await this.startEncoderPool();

			// 3. Split files across workers
			const batches: { text: string; path: string }[][] = Array.from(
				{ length: ENCODER_WORKER_COUNT },
				() => []
			);
			for (let i = 0; i < fileData.length; i++) {
				batches[i % ENCODER_WORKER_COUNT].push(fileData[i]);
			}

			// 4. Track progress from all workers
			const workerProgress: number[] = new Array(ENCODER_WORKER_COUNT).fill(0);
			let totalChunksEncoded = 0;

			const progressHandler = (e: MessageEvent) => {
				const msg = e.data;
				if (msg.type === "encode_progress") {
					workerProgress[msg.workerId] = msg.indexed;
					const totalIndexed = workerProgress.reduce((a, b) => a + b, 0);
					const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
					const rate = ((totalIndexed / (Date.now() - startTime)) * 1000).toFixed(1);
					this.statusBarEl?.setText(
						`Kjarni: encoding ${totalIndexed}/${total} files · ${elapsed}s · ${rate} files/s`
					);
				}
			};

			for (const worker of this.encoderWorkers) {
				worker.addEventListener("message", progressHandler);
			}

			// 5. Send batches to encoder workers in parallel
			this.statusBarEl?.setText(`Kjarni: encoding 0/${total} files...`);

			const encodePromises = batches.map((batch, i) => {
				if (batch.length === 0) return Promise.resolve({ chunks: [], skipped: 0 });
				return this.postToEncoder(this.encoderWorkers[i], {
					type: "encode_batch",
					files: batch,
					workerId: i,
				});
			});

			const encodeResults = await Promise.all(encodePromises);

			// Remove progress listeners
			for (const worker of this.encoderWorkers) {
				worker.removeEventListener("message", progressHandler);
			}

			// 6. Terminate encoder pool — no longer needed
			this.stopEncoderPool();

			// 7. Collect all encoded chunks
			let allChunks: EncodedChunk[] = [];
			let totalSkipped = 0;

			for (const result of encodeResults) {
				if (result.chunks) {
					allChunks = allChunks.concat(result.chunks);
				}
				totalSkipped += result.skipped || 0;
			}

			totalChunksEncoded = allChunks.length;

			// 8. Send chunks to search worker to build index
			this.statusBarEl?.setText(
				`Kjarni: building index from ${totalChunksEncoded} chunks...`
			);

			await this.postToSearch({ type: "build_start" });

			// Send chunks in batches of 1000 to avoid massive single message
			const CHUNK_BATCH_SIZE = 1000;
			for (let i = 0; i < allChunks.length; i += CHUNK_BATCH_SIZE) {
				const batch = allChunks.slice(i, i + CHUNK_BATCH_SIZE);
				await this.postToSearch({
					type: "add_encoded_chunks",
					chunks: batch,
				});

				this.statusBarEl?.setText(
					`Kjarni: indexing ${Math.min(i + CHUNK_BATCH_SIZE, allChunks.length)}/${totalChunksEncoded} chunks...`
				);
			}

			// 9. Finalize
			const result = await this.postToSearch({ type: "build_finish" });

			if (result.indexBytes) {
				const indexPath = path.join(this.pluginDir, "index.idx");
				fs.writeFileSync(indexPath, Buffer.from(result.indexBytes));
			}

			this.indexReady = true;
			const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
			const skipMsg = totalSkipped > 0 ? ` (${totalSkipped} skipped)` : "";
			this.statusBarEl?.setText(
				`Kjarni: ${result.docCount} chunks · ${total} files`
			);
			new Notice(
				`Kjarni: Indexed ${result.docCount} chunks from ${total} files in ${elapsed}s${skipMsg}`
			);
		} catch (e) {
			console.error("Kjarni: Indexing failed:", e);
			new Notice(`Kjarni: Indexing failed — ${e}`);
			this.statusBarEl?.setText("Kjarni: indexing failed");
			this.stopEncoderPool();
		} finally {
			this.indexing = false;
		}
	}

	// ─── Incremental Indexing ────────────────────────────────────

	async catchUpChangedFiles(indexMtime: number) {
		const allFiles = this.app.vault.getMarkdownFiles();
		const changedFiles = allFiles.filter((f) => f.stat.mtime > indexMtime);

		if (changedFiles.length === 0) return;

		if (changedFiles.length > allFiles.length * 0.5) {
			new Notice(`Kjarni: ${changedFiles.length} files changed — rebuilding...`);
			await this.buildFullIndex();
			return;
		}

		this.statusBarEl?.setText(
			`Kjarni: updating ${changedFiles.length} changed files...`
		);

		let updated = 0;
		for (const file of changedFiles) {
			try {
				const text = await this.app.vault.cachedRead(file);
				await this.postToSearch({
					type: "update_file",
					text,
					path: file.path,
				});
				updated++;
			} catch (e) {
				console.warn(`Kjarni: Failed to update ${file.path}: ${e}`);
			}
		}

		if (updated > 0) {
			await this.requestSaveIndex();
			new Notice(`Kjarni: Updated ${updated} changed files`);
		}
	}

	queueFileUpdate(filePath: string) {
		if (!this.indexReady) return;
		this.pendingDeletes.delete(filePath);
		this.pendingUpdates.add(filePath);
		if (this.updateTimer) clearTimeout(this.updateTimer);
		this.updateTimer = setTimeout(
			() => this.flushUpdates(),
			FILE_UPDATE_DEBOUNCE_MS
		);
	}

	queueFileDelete(filePath: string) {
		if (!this.indexReady) return;
		this.pendingUpdates.delete(filePath);
		this.pendingDeletes.add(filePath);
		if (this.deleteTimer) clearTimeout(this.deleteTimer);
		this.deleteTimer = setTimeout(
			() => this.flushDeletes(),
			DELETE_DEBOUNCE_MS
		);
	}

	async flushUpdates() {
		if (this.pendingUpdates.size === 0) return;

		const paths = [...this.pendingUpdates];
		this.pendingUpdates.clear();

		let updated = 0;
		let lastCount = 0;

		for (const filePath of paths) {
			const file = this.app.vault.getAbstractFileByPath(filePath);
			if (!(file instanceof TFile)) continue;

			try {
				const text = await this.app.vault.cachedRead(file);
				const result = await this.postToSearch({
					type: "update_file",
					text,
					path: filePath,
				});
				lastCount = result.docCount;
				updated++;
			} catch (e) {
				console.warn(`Kjarni: Failed to update ${filePath}: ${e}`);
			}
		}

		if (updated > 0) {
			this.requestSaveIndex();
			this.statusBarEl?.setText(
				`Kjarni: ${lastCount} chunks · updated ${updated} files`
			);
			setTimeout(() => {
				this.statusBarEl?.setText(`Kjarni: ${lastCount} chunks`);
			}, 3000);
		}
	}

	async flushDeletes() {
		if (this.pendingDeletes.size === 0) return;

		const paths = [...this.pendingDeletes];
		this.pendingDeletes.clear();

		let lastCount = 0;

		for (const filePath of paths) {
			try {
				const result = await this.postToSearch({
					type: "remove_file",
					path: filePath,
				});
				lastCount = result.docCount;
			} catch (e) {
				console.warn(`Kjarni: Failed to remove ${filePath}: ${e}`);
			}
		}

		this.requestSaveIndex();
		this.statusBarEl?.setText(`Kjarni: ${lastCount} chunks`);
	}

	async requestSaveIndex() {
		try {
			const result = await this.postToSearch({ type: "save_index" });
			if (result.indexBytes) {
				const indexPath = path.join(this.pluginDir, "index.idx");
				fs.writeFileSync(indexPath, Buffer.from(result.indexBytes));
			}
			this.statusBarEl?.setText(`Kjarni: ${result.docCount} chunks`);
		} catch (e) {
			console.error("Kjarni: Failed to save index:", e);
		}
	}

	async reindexVault() {
		if (this.indexing) {
			new Notice("Kjarni: Already indexing...");
			return;
		}
		this.pendingUpdates.clear();
		this.pendingDeletes.clear();
		if (this.updateTimer) clearTimeout(this.updateTimer);
		if (this.deleteTimer) clearTimeout(this.deleteTimer);
		await this.buildFullIndex();
	}

	// ─── Search ──────────────────────────────────────────────────

	openSearch() {
		if (!this.indexReady) {
			new Notice("Kjarni: Index not ready. Please wait for indexing to complete.");
			return;
		}
		new SearchModal(this.app, this).open();
	}

	async doSearch(query: string, limit: number): Promise<SearchResultItem[]> {
		if (!query.trim()) return [];
		try {
			const result = await this.postToSearch({ type: "search", query, limit });
			return result.results || [];
		} catch (e) {
			console.error("Kjarni: Search failed:", e);
			return [];
		}
	}

	async doRerank(query: string, docs: string[], limit: number): Promise<RerankResultItem[]> {
		if (!this.hasReranker || docs.length === 0) return [];
		try {
			const result = await this.postToSearch({ type: "rerank", query, docs, limit });
			return result.results || [];
		} catch (e) {
			console.error("Kjarni: Rerank failed:", e);
			return [];
		}
	}

	// ─── Settings ────────────────────────────────────────────────

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
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
	statusEl: HTMLElement | null = null;
	debounceTimer: ReturnType<typeof setTimeout> | null = null;
	searchVersion: number = 0;
	lastDisplayed: SearchResultItem[] = [];

	constructor(app: App, plugin: KjarniSearchPlugin) {
		super(app);
		this.plugin = plugin;
	}

	onOpen() {
		const { contentEl, modalEl } = this;
		modalEl.addClass("kjarni-search-modal");

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

		header.createEl("span", { cls: "kjarni-kbd", text: "esc" });

		this.resultsEl = contentEl.createDiv({ cls: "kjarni-results" });
		this.statusEl = contentEl.createDiv({ cls: "kjarni-status" });

		this.inputEl.addEventListener("input", () => {
			if (this.debounceTimer) clearTimeout(this.debounceTimer);
			this.debounceTimer = setTimeout(() => this.runSearch(), 150);
		});

		this.inputEl.addEventListener("keydown", (e) => {
			if (e.key === "Escape") this.close();
			else if (e.key === "Enter") this.openResult(0);
		});

		this.inputEl.focus();
	}

	async runSearch() {
		const query = this.inputEl?.value?.trim();
		if (!query || !this.resultsEl) {
			if (this.resultsEl) this.resultsEl.empty();
			if (this.statusEl) this.statusEl.textContent = "";
			return;
		}

		this.searchVersion++;
		const version = this.searchVersion;

		// Phase 1: Hybrid search
		const t0 = performance.now();
		const results = await this.plugin.doSearch(
			query,
			this.plugin.settings.searchLimit * 2
		);
		const searchMs = performance.now() - t0;

		if (this.searchVersion !== version) return;

		const displayResults = results.slice(0, this.plugin.settings.searchLimit);
		this.lastDisplayed = displayResults;
		this.renderResults(displayResults);

		// Phase 2: Async rerank top 5
		if (this.plugin.hasReranker && displayResults.length > 0) {
			if (this.statusEl) {
				this.statusEl.textContent = `${displayResults.length} results · ${searchMs.toFixed(0)}ms · refining...`;
				this.statusEl.addClass("kjarni-refining");
			}

			const topDocs = displayResults.slice(0, 5).map((r) => r.text);
			const t1 = performance.now();
			const reranked = await this.plugin.doRerank(query, topDocs, 5);
			const rerankMs = performance.now() - t1;

			if (this.searchVersion !== version) return;

			if (reranked.length > 0) {
				const reorderedTop = reranked.map((r) => displayResults[r.index]);
				const rerankedIndices = new Set(reranked.map((r) => r.index));
				const rest = displayResults.filter((_, i) => i >= 5 || !rerankedIndices.has(i));
				this.lastDisplayed = [...reorderedTop, ...rest];
				this.renderResults(this.lastDisplayed);
			}

			if (this.statusEl) {
				this.statusEl.textContent = `${displayResults.length} results · ${searchMs.toFixed(0)}ms search · ${rerankMs.toFixed(0)}ms rerank · refined`;
				this.statusEl.removeClass("kjarni-refining");
				this.statusEl.addClass("kjarni-refined");
				setTimeout(() => this.statusEl?.removeClass("kjarni-refined"), 2000);
			}
		} else {
			if (this.statusEl) {
				this.statusEl.textContent = `${displayResults.length} results · ${searchMs.toFixed(0)}ms`;
				this.statusEl.removeClass("kjarni-refining");
			}
		}
	}

	renderResults(results: SearchResultItem[]) {
		if (!this.resultsEl) return;
		this.resultsEl.empty();

		if (results.length === 0) {
			this.resultsEl.createDiv({ cls: "kjarni-empty", text: "No results found" });
			return;
		}

		results.forEach((r, i) => {
			const snippet = r.text.length > 200 ? r.text.slice(0, 200) + "..." : r.text;

			const el = this.resultsEl!.createDiv({ cls: "kjarni-result" });
			el.addEventListener("click", () => this.openResult(i));

			const title = el.createDiv({ cls: "kjarni-result-title" });
			title.createEl("span", { text: r.source.replace(/\.md$/, "") });
			title.createEl("span", { cls: "kjarni-score", text: r.score.toFixed(3) });

			el.createDiv({ cls: "kjarni-result-text", text: snippet });
		});
	}

	openResult(index: number) {
		const result = this.lastDisplayed[index];
		if (!result) return;

		const file = this.app.vault.getAbstractFileByPath(result.source);
		if (file instanceof TFile) {
			this.app.workspace.openLinkText(result.source, "", false);
			this.close();
		}
	}

	onClose() {
		this.searchVersion++;
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
			.setDesc("Use cross-encoder reranking for better result quality")
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
			.setDesc("Maximum characters per chunk when indexing (default 1000)")
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
