// Local AI inference in the browser, off the main thread.
//
//   const kjarni = await Kjarni.load({ encoder: "/models/minilm.kjq" });
//   const [a, b] = await kjarni.encode(["doctor", "physician"]);
//   console.log(Kjarni.cosine(a, b));   // 0.8598
//
// Every method returns a promise and the work happens in a worker, so the page
// keeps painting while a model loads or a reply generates.

import type {
  Capability,
  ClassifyResult,
  ModelSource,
  RankedDocument,
  Request,
  Response,
} from "./protocol.js";

export type { ClassifyResult, ModelSource, RankedDocument };

export interface LoadOptions {
  /** Embeddings and semantic similarity. A `.kjq` file. */
  encoder?: ModelSource;
  /** Sentiment, emotion, toxicity. A `.kjq` file. */
  classifier?: ModelSource;
  /** Cross-encoder reranking. A `.kjq` file. */
  reranker?: ModelSource;
  /** A local language model. A `.kjq` file. */
  chat?: ModelSource;
  /** Chat only: the registry id whose prompt template should be used. */
  chatModelId?: string;
  /** Called as each model is fetched, for progress UI. */
  onProgress?: (capability: Capability, loaded: number, total: number) => void;
}

export interface ChatOptions {
  maxNewTokens?: number;
  /** 0 selects greedy decoding, which is deterministic. */
  temperature?: number;
}

async function toBytes(
  src: ModelSource,
  capability: Capability,
  onProgress?: LoadOptions["onProgress"],
): Promise<Uint8Array> {
  if (src instanceof Uint8Array) return src;
  if (src instanceof ArrayBuffer) return new Uint8Array(src);

  const response = await fetch(String(src));
  if (!response.ok) {
    throw new Error(`fetching ${src} failed: HTTP ${response.status}`);
  }

  // Models run to hundreds of megabytes, so report progress rather than leaving
  // the page silent for a minute. Falls back to a plain read when the server
  // sends no length or the body cannot be streamed.
  const total = Number(response.headers.get("content-length") ?? 0);
  if (!onProgress || !total || !response.body) {
    return new Uint8Array(await response.arrayBuffer());
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(capability, loaded, total);
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

/**
 * `Omit` over a union collapses it to the keys every member shares, which for
 * `Request` is just `kind` and `id`. Distributing over the members first keeps
 * each variant's own fields, so a malformed request is still a type error.
 */
type PendingRequest = Request extends infer R
  ? R extends { id: number }
    ? Omit<R, "id">
    : never
  : never;

export class Kjarni {
  #worker: Worker;
  #next = 1;
  #pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void; onToken?: (t: string) => void }
  >();

  private constructor(worker: Worker) {
    this.#worker = worker;
    this.#worker.onmessage = (event: MessageEvent<Response>) => {
      const msg = event.data;
      const entry = this.#pending.get(msg.id);
      if (!entry) return;

      if (msg.kind === "token") {
        entry.onToken?.(msg.text);
        return; // more to come; the promise settles on ok or err
      }
      this.#pending.delete(msg.id);
      if (msg.kind === "err") entry.reject(new Error(msg.message));
      else entry.resolve(msg.value);
    };
  }

  /** Starts a worker and loads whichever models you name. */
  static async load(options: LoadOptions): Promise<Kjarni> {
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const kjarni = new Kjarni(worker);

    const capabilities: Capability[] = ["encoder", "classifier", "reranker", "chat"];
    for (const capability of capabilities) {
      const source = options[capability];
      if (!source) continue;
      const bytes = await toBytes(source, capability, options.onProgress);
      await kjarni.#send(
        { kind: "load", capability, bytes, modelId: options.chatModelId },
        // The buffer is handed to the worker rather than copied. A 500MB model
        // copied per load is a real pause and a doubled peak memory footprint.
        [bytes.buffer],
      );
    }
    return kjarni;
  }

  #send(req: PendingRequest, transfer: Transferable[] = [], onToken?: (t: string) => void) {
    const id = this.#next++;
    return new Promise<unknown>((resolve, reject) => {
      this.#pending.set(id, { resolve, reject, onToken });
      this.#worker.postMessage({ ...req, id } as Request, transfer);
    });
  }

  /** One vector per input, in the order given. */
  async encode(texts: string[], normalize = true): Promise<Float32Array[]> {
    return (await this.#send({ kind: "encode", texts, normalize })) as Float32Array[];
  }

  /** Cosine similarity between two texts, from -1 to 1. */
  async similarity(a: string, b: string): Promise<number> {
    const [va, vb] = await this.encode([a, b]);
    return Kjarni.cosine(va, vb);
  }

  /** The best label for each input. */
  async classify(texts: string | string[]): Promise<ClassifyResult[]> {
    const batch = Array.isArray(texts) ? texts : [texts];
    return (await this.#send({ kind: "classify", texts: batch })) as ClassifyResult[];
  }

  /** Every label this classifier can predict, in index order. */
  async labels(): Promise<string[]> {
    return (await this.#send({ kind: "labels" })) as string[];
  }

  /** Documents reordered by relevance, best first. Indices point into your input. */
  async rerank(query: string, documents: string[], limit = documents.length): Promise<RankedDocument[]> {
    return (await this.#send({ kind: "rerank", query, documents, limit })) as RankedDocument[];
  }

  /** The chat model's context window, in tokens. */
  async contextSize(): Promise<number> {
    return (await this.#send({ kind: "contextSize" })) as number;
  }

  /**
   * Generates a reply, yielding text as it arrives.
   *
   *   for await (const piece of kjarni.chat("Explain RAG")) out.textContent += piece;
   */
  async *chat(prompt: string, options: ChatOptions = {}): AsyncGenerator<string, void, unknown> {
    const queue: string[] = [];
    let wake: (() => void) | null = null;

    const done = this.#send(
      {
        kind: "chat",
        prompt,
        maxNewTokens: options.maxNewTokens ?? 256,
        temperature: options.temperature ?? 0,
      },
      [],
      (text) => {
        queue.push(text);
        wake?.();
      },
    );

    let finished = false;
    const settled = done.then(
      () => { finished = true; wake?.(); },
      (e) => { finished = true; wake?.(); throw e; },
    );

    while (!finished || queue.length) {
      if (queue.length) {
        yield queue.shift()!;
        continue;
      }
      await new Promise<void>((r) => (wake = r));
    }
    await settled;   // surfaces a generation error rather than ending quietly
  }

  /** Releases the models and stops the worker. */
  async close(): Promise<void> {
    try {
      await this.#send({ kind: "close" });
    } finally {
      this.#worker.terminate();
    }
  }

  /** Cosine similarity between two vectors. */
  static cosine(a: Float32Array, b: Float32Array): number {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1e-9);
  }
}

export default Kjarni;
