// ─── Similar Notes Sidebar View ──────────────────────────────────
//
// File: similar-view.ts
// Add this as a new file, or paste into main.ts above the SearchModal class.
//
// If using as a separate file, you need to:
// 1. Export SearchResultItem and KjarniSearchPlugin from main.ts
// 2. Import them here
//
// If pasting into main.ts directly, remove the imports below.

import {
	App,
	ItemView,
	TFile,
	WorkspaceLeaf,
	setIcon,
} from "obsidian";

// If in a separate file, uncomment:
import type KjarniSearchPlugin from "./main";
import type { SearchResultItem } from "./main";

export const SIMILAR_VIEW_TYPE = "kjarni-similar-notes";

export class SimilarNotesView extends ItemView {
	plugin: KjarniSearchPlugin;
	private currentFile: TFile | null = null;
	private debounceTimer: ReturnType<typeof setTimeout> | null = null;
	private searchVersion: number = 0;
	private results: SearchResultItem[] = [];

	constructor(leaf: WorkspaceLeaf, plugin: KjarniSearchPlugin) {
		super(leaf);
		this.plugin = plugin;
	}

	getViewType(): string {
		return SIMILAR_VIEW_TYPE;
	}

	getDisplayText(): string {
		return "Similar Notes";
	}

	getIcon(): string {
		return "git-compare";
	}

	async onOpen() {
		const container = this.containerEl.children[1];
		container.empty();
		container.addClass("kjarni-similar-container");

		// Header
		const header = container.createDiv({ cls: "kjarni-similar-header" });
		header.createEl("span", { text: "Similar Notes", cls: "kjarni-similar-title" });

		// Refresh button
		const refreshBtn = header.createEl("button", { cls: "kjarni-similar-refresh clickable-icon" });
		setIcon(refreshBtn, "refresh-cw");
		refreshBtn.setAttribute("aria-label", "Refresh");
		refreshBtn.addEventListener("click", () => {
			if (this.currentFile) {
				this.findSimilar(this.currentFile, true);
			}
		});

		// Results container
		container.createDiv({ cls: "kjarni-similar-results" });

		// Status
		container.createDiv({ cls: "kjarni-similar-status" });

		// Show initial state
		this.renderEmpty("Open a note to see similar notes");

		// Listen for active file changes
		this.registerEvent(
			this.app.workspace.on("active-leaf-change", (leaf) => {
				if (!leaf) return;
				const view = leaf.view;
				if (view.getViewType() === "markdown") {
					const file = (view as any).file;
					if (file instanceof TFile && file.extension === "md") {
						this.onActiveFileChange(file);
					}
				}
			})
		);

		// Also trigger on file modify (so sidebar updates as you type)
		this.registerEvent(
			this.app.vault.on("modify", (file) => {
				if (
					file instanceof TFile &&
					file === this.currentFile &&
					this.plugin.settings.similarAutoUpdate
				) {
					// Longer debounce for typing — don't search on every keystroke
					if (this.debounceTimer) clearTimeout(this.debounceTimer);
					this.debounceTimer = setTimeout(() => {
						this.findSimilar(file, true);
					}, 2000);
				}
			})
		);

		// Check if there's already an active file
		const activeFile = this.app.workspace.getActiveFile();
		if (activeFile) {
			this.onActiveFileChange(activeFile);
		}
	}

	async onClose() {
		this.searchVersion++;
		if (this.debounceTimer) clearTimeout(this.debounceTimer);
	}

	// ─── Core Logic ──────────────────────────────────────────────

	private onActiveFileChange(file: TFile) {
		if (this.currentFile?.path === file.path) {
			// Same file — don't re-search unless explicitly asked
			return;
		}

		this.currentFile = file;
		this.results = []; // Reset so findSimilar knows to fetch

		// Debounce — user might be clicking through files quickly
		if (this.debounceTimer) clearTimeout(this.debounceTimer);
		this.debounceTimer = setTimeout(() => {
			this.findSimilar(file, false);
		}, 500);
	}

	async findSimilar(file: TFile, forceRefresh: boolean = false) {
		if (!this.plugin.indexReady) {
			this.renderEmpty("Index not ready yet...");
			return;
		}

		this.currentFile = file;
		this.searchVersion++;
		const version = this.searchVersion;

		// Show loading
		this.renderLoading(file.basename);

		try {
			// Read current note
			const text = await this.app.vault.cachedRead(file);
			if (text.trim().length === 0) {
				this.renderEmpty("Empty note");
				return;
			}

			// Use the note text as a search query
			// Truncate to first ~500 chars for the query to keep it focused
			const queryText = text.slice(0, 500);

			const t0 = performance.now();
			const results = await this.plugin.doSearch(queryText, 20);
			const searchMs = performance.now() - t0;

			// Check if still current
			if (this.searchVersion !== version) return;

			// Filter out the current file itself
			const filtered = results.filter((r) => r.source !== file.path);

			// Deduplicate by source — keep highest scoring chunk per file
			const bySource = new Map<string, SearchResultItem>();
			for (const r of filtered) {
				const existing = bySource.get(r.source);
				if (!existing || r.score > existing.score) {
					bySource.set(r.source, r);
				}
			}

			this.results = Array.from(bySource.values())
				.sort((a, b) => b.score - a.score)
				.slice(0, this.plugin.settings.searchLimit);

			if (this.searchVersion !== version) return;

			this.renderResults(file.basename, this.results, searchMs);
		} catch (e) {
			console.error("Kjarni: Similar notes search failed:", e);
			this.renderEmpty("Search failed");
		}
	}

	// ─── Rendering ───────────────────────────────────────────────

	private getResultsEl(): HTMLElement | null {
		return this.containerEl.querySelector(".kjarni-similar-results");
	}

	private getStatusEl(): HTMLElement | null {
		return this.containerEl.querySelector(".kjarni-similar-status");
	}

	private renderEmpty(message: string) {
		const el = this.getResultsEl();
		if (!el) return;
		el.empty();

		const emptyEl = el.createDiv({ cls: "kjarni-similar-empty" });
		const iconEl = emptyEl.createDiv({ cls: "kjarni-similar-empty-icon" });
		setIcon(iconEl, "file-search");
		emptyEl.createEl("p", { text: message });

		const statusEl = this.getStatusEl();
		if (statusEl) statusEl.textContent = "";
	}

	private renderLoading(filename: string) {
		const el = this.getResultsEl();
		if (!el) return;
		el.empty();

		const loadingEl = el.createDiv({ cls: "kjarni-similar-loading" });
		loadingEl.createEl("p", { text: `Finding notes similar to "${filename}"...` });

		const statusEl = this.getStatusEl();
		if (statusEl) statusEl.textContent = "Searching...";
	}

	private renderResults(
		currentName: string,
		results: SearchResultItem[],
		searchMs: number
	) {
		const el = this.getResultsEl();
		if (!el) return;
		el.empty();

		if (results.length === 0) {
			this.renderEmpty("No similar notes found");
			return;
		}

		// Normalize scores for display
		const maxScore = Math.max(...results.map((r) => r.score), 0.001);

		for (const result of results) {
			const pct = Math.round((result.score / maxScore) * 100);
			const item = el.createDiv({ cls: "kjarni-similar-item" });

			// Click to open
			item.addEventListener("click", (e) => {
				this.app.workspace.openLinkText(result.source, "", false);
			});

			// Ctrl/Cmd+Click to open in new tab
			item.addEventListener("auxclick", (e) => {
				if (e.button === 1) {
					this.app.workspace.openLinkText(result.source, "", true);
				}
			});

			// Hover preview
			item.addEventListener("mouseover", (e) => {
				this.app.workspace.trigger("hover-link", {
					event: e,
					source: SIMILAR_VIEW_TYPE,
					hoverParent: item,
					targetEl: item,
					linktext: result.source,
				});
			});

			// File name
			const titleRow = item.createDiv({ cls: "kjarni-similar-item-title" });
			const nameEl = titleRow.createEl("span", {
				cls: "kjarni-similar-item-name",
			});
			nameEl.textContent = result.source.replace(/\.md$/, "");

			// Score badge
			const scoreEl = titleRow.createEl("span", {
				cls: "kjarni-similar-item-score",
			});
			scoreEl.textContent = `${pct}%`;

			// Score bar
			const barContainer = item.createDiv({ cls: "kjarni-similar-bar-container" });
			const barFill = barContainer.createDiv({ cls: "kjarni-similar-bar-fill" });
			barFill.style.width = `${pct}%`;

			// Snippet
			const snippet = result.text.length > 120
				? result.text.slice(0, 120) + "…"
				: result.text;
			item.createDiv({
				cls: "kjarni-similar-item-snippet",
				text: snippet,
			});

			// Insert link button
			const linkBtn = item.createEl("button", {
				cls: "kjarni-similar-link-btn clickable-icon",
				attr: { "aria-label": "Insert link to this note" },
			});
			setIcon(linkBtn, "link");
			linkBtn.addEventListener("click", (e) => {
				e.stopPropagation();
				this.insertLink(result.source);
			});
		}

		const statusEl = this.getStatusEl();
		if (statusEl) {
			statusEl.textContent = `${results.length} similar notes · ${searchMs.toFixed(0)}ms`;
		}
	}

	private insertLink(sourcePath: string) {
		const activeView = this.app.workspace.getActiveViewOfType(
			// @ts-ignore
			this.app.workspace.constructor.prototype.constructor
		);
		const editor = this.app.workspace.activeEditor?.editor;
		if (!editor) {
			// Fallback: copy to clipboard
			navigator.clipboard.writeText(`[[${sourcePath.replace(/\.md$/, "")}]]`);
			return;
		}

		const linkText = `[[${sourcePath.replace(/\.md$/, "")}]]`;
		const cursor = editor.getCursor();
		editor.replaceRange(linkText, cursor);
		editor.setCursor({
			line: cursor.line,
			ch: cursor.ch + linkText.length,
		});
	}
}