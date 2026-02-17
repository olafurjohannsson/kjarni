/// <reference lib="webworker" />

// encoder-worker.ts — lightweight encoding worker
// Multiple instances run in parallel during indexing.
// Only does: split text → encode chunks → return embeddings.
// Terminated after indexing completes.

// @ts-ignore
import init, { WasmEncoder } from "../../pkg/kjarni_wasm.js";

let encoder: any = null;

function send(msg: any, transfer?: Transferable[]) {
	if (transfer) {
		self.postMessage(msg, transfer);
	} else {
		self.postMessage(msg);
	}
}

// Sequential message queue (prevent interleaving on async init)
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
				case "encode_batch":
					handleEncodeBatch(msg);
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

	encoder = WasmEncoder.new(new Uint8Array(msg.encoderBytes));

	send({ type: "init_done", id: msg.id });
}

function handleEncodeBatch(msg: any) {
	if (!encoder) {
		send({ type: "error", id: msg.id, error: "Encoder not initialized" });
		return;
	}

	const files: { text: string; path: string }[] = msg.files;
	const allChunks: any[] = [];
	let skipped = 0;

	for (let i = 0; i < files.length; i++) {
		const file = files[i];

		try {
			const encoded = encoder.encode_file(file.text, file.path);
			// encoded is array of {text, embedding, source, chunk_index}
			for (const chunk of encoded) {
				allChunks.push({
					text: chunk.text,
					embedding: Array.from(chunk.embedding),
					source: chunk.source,
					chunk_index: chunk.chunk_index,
				});
			}
		} catch (e: any) {
			skipped++;
		}

		// Progress every 10 files
		if ((i + 1) % 10 === 0) {
			send({
				type: "encode_progress",
				workerId: msg.workerId,
				indexed: i + 1,
				total: files.length,
				chunks: allChunks.length,
			});
		}
	}

	send({
		type: "encode_done",
		id: msg.id,
		workerId: msg.workerId,
		chunks: allChunks,
		fileCount: files.length,
		skipped,
	});
}
