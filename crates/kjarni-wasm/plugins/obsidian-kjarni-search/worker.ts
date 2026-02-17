/// <reference lib="webworker" />
// worker.ts — search/index worker
// Single instance, stays alive for the plugin's lifetime.
// Builds index from pre-encoded chunks (sent by encoder workers).
// Handles search, rerank, incremental updates.

// @ts-ignore
import init, { WasmIndexBuilder, WasmSearch, WasmReranker, set_debug_logging } from "../../pkg/kjarni_wasm.js";

let search: any = null;
let reranker: any = null;
let builder: any = null;
let encoderBytes: Uint8Array | null = null;

function send(msg: any, transfer?: Transferable[]) {
	if (transfer) {
		self.postMessage(msg, transfer);
	} else {
		self.postMessage(msg);
	}
}

// Sequential message queue
let processing = false;
const queue: MessageEvent[] = [];

self.onmessage = (e: MessageEvent) => {
	queue.push(e);
	processQueue();
};

async function processQueue() {
	if (processing) return;
	processing = true;

	while (queue.length > 0) {
		const e = queue.shift()!;
		const msg = e.data;
		try {
			switch (msg.type) {
				case "init":
					await handleInit(msg);
					break;
				case "build_start":
					handleBuildStart(msg);
					break;
				case "add_encoded_chunks":
					handleAddEncodedChunks(msg);
					break;
				case "build_finish":
					handleBuildFinish(msg);
					break;
				case "search":
					handleSearch(msg);
					break;
				case "rerank":
					handleRerank(msg);
					break;
				case "update_file":
					handleUpdateFile(msg);
					break;
				case "remove_file":
					handleRemoveFile(msg);
					break;
				case "save_index":
					handleSaveIndex(msg);
					break;
				case "set_logging":
					set_debug_logging(msg.enabled);
					send({ type: "logging_set", id: msg.id });
					break;
				default:
					send({ type: "error", id: msg.id, error: `Unknown: ${msg.type}` });
			}
		} catch (e: any) {
			send({ type: "error", id: msg.id, error: e.message || String(e) });
		}
	}

	processing = false;
}

async function handleInit(msg: any) {
	await init(msg.wasmBytes);

	encoderBytes = new Uint8Array(msg.encoderBytes);

	if (msg.rerankerBytes) {
		try {
			reranker = WasmReranker.load(new Uint8Array(msg.rerankerBytes));
		} catch (e: any) {
			console.error("Worker: Failed to load reranker:", e);
		}
	}

	if (msg.indexBytes) {
		try {
			search = WasmSearch.load(encoderBytes, new Uint8Array(msg.indexBytes));
			send({
				type: "init_done",
				id: msg.id,
				hasIndex: true,
				docCount: search.doc_count(),
				hasReranker: reranker !== null,
			});
			return;
		} catch (e: any) {
			console.error("Worker: Failed to load index:", e);
		}
	}

	send({
		type: "init_done",
		id: msg.id,
		hasIndex: false,
		docCount: 0,
		hasReranker: reranker !== null,
	});
}

function handleBuildStart(msg: any) {
	if (!encoderBytes) {
		send({ type: "error", id: msg.id, error: "No encoder loaded" });
		return;
	}
	builder = WasmIndexBuilder.new(encoderBytes);
	send({ type: "build_started", id: msg.id });
}

function handleAddEncodedChunks(msg: any) {
	if (!builder) {
		send({ type: "error", id: msg.id, error: "No builder — call build_start first" });
		return;
	}

	const chunks: {
		text: string;
		embedding: number[];
		source: string;
		chunk_index: number;
	}[] = msg.chunks;

	let added = 0;
	for (const chunk of chunks) {
		try {
			builder.add_chunk(
				chunk.text,
				new Float32Array(chunk.embedding),
				chunk.source,
				chunk.chunk_index
			);
			added++;
		} catch (e: any) {
			console.warn(`Worker: Failed to add chunk from ${chunk.source}: ${e}`);
		}
	}

	send({
		type: "chunks_added",
		id: msg.id,
		added,
	});
}

function handleBuildFinish(msg: any) {
	if (!builder || !encoderBytes) {
		send({ type: "error", id: msg.id, error: "No builder or encoder" });
		return;
	}

	const indexBytes = builder.finish();
	search = WasmSearch.load(encoderBytes, new Uint8Array(indexBytes));
	builder = null;

	const transferable = indexBytes.buffer || indexBytes;
	send(
		{
			type: "build_done",
			id: msg.id,
			docCount: search.doc_count(),
			indexBytes: indexBytes,
		},
		[transferable]
	);
}

function handleSearch(msg: any) {
	if (!search) {
		send({ type: "search_result", id: msg.id, results: [] });
		return;
	}

	const rawResults = search.search(msg.query, msg.limit || 20);

	const results = rawResults.map((r: any) => {
		const meta =
			r.metadata instanceof Map
				? Object.fromEntries(r.metadata)
				: r.metadata || {};
		return {
			score: r.score,
			text: r.text,
			source: meta.source || "unknown",
			metadata: meta,
		};
	});

	send({ type: "search_result", id: msg.id, results });
}

function handleRerank(msg: any) {
	if (!reranker) {
		send({ type: "rerank_result", id: msg.id, results: [] });
		return;
	}

	const rawResults = reranker.rerank(msg.query, msg.docs, msg.limit || 5);

	const results = rawResults.map((r: any) => ({
		index: r.index,
		score: r.score,
		text: r.text,
	}));

	send({ type: "rerank_result", id: msg.id, results });
}

function handleUpdateFile(msg: any) {
	if (!search) {
		send({ type: "error", id: msg.id, error: "No search index loaded" });
		return;
	}

	let chunks = 0;
	try {
		chunks = search.update_file(msg.text, msg.path);
	} catch (e: any) {
		send({ type: "error", id: msg.id, error: `Update failed: ${e}` });
		return;
	}

	send({
		type: "file_updated",
		id: msg.id,
		path: msg.path,
		chunks,
		docCount: search.doc_count(),
	});
}

function handleRemoveFile(msg: any) {
	if (!search) {
		send({ type: "error", id: msg.id, error: "No search index loaded" });
		return;
	}

	const removed = search.remove_file(msg.path);

	send({
		type: "file_removed",
		id: msg.id,
		path: msg.path,
		removed,
		docCount: search.doc_count(),
	});
}

function handleSaveIndex(msg: any) {
	if (!search) {
		send({ type: "error", id: msg.id, error: "No search index to save" });
		return;
	}

	const indexBytes = search.save_index();
	const transferable = indexBytes.buffer || indexBytes;

	send(
		{
			type: "index_saved",
			id: msg.id,
			docCount: search.doc_count(),
			indexBytes: indexBytes,
		},
		[transferable]
	);
}
