// The worker. Everything expensive happens here so the page never janks.
//
// Inference in this build is synchronous: `encode` and `generate` occupy the
// thread until they return. On the main thread that is a frozen tab for the
// duration, which for a chat reply is seconds. Here it costs nothing, because
// this thread has nothing else to do.

import init, {
  WasmModel,
  WasmClassifier,
  WasmReranker,
  WasmChat,
} from "../pkg/kjarni_wasm.js";

import type { Request, Response } from "./protocol.js";

let model: WasmModel | null = null;
let classifier: WasmClassifier | null = null;
let reranker: WasmReranker | null = null;
let chat: WasmChat | null = null;
let ready: Promise<unknown> | null = null;

const post = (msg: Response) => (self as unknown as Worker).postMessage(msg);

/** Loads the wasm binary once, however many times this is called. */
function ensureWasm(): Promise<unknown> {
  ready ??= init();
  return ready;
}

function handle(req: Request): unknown {
  switch (req.kind) {
    case "load": {
      // Loading a second model of the same kind replaces the first, and the old
      // one has to be freed explicitly: wasm-bindgen handles cannot be collected
      // by the JS garbage collector, so dropping the reference leaks the weights.
      switch (req.capability) {
        case "encoder":
          model?.free();
          model = WasmModel.from_quantized(req.bytes);
          return null;
        case "classifier":
          classifier?.free();
          classifier = WasmClassifier.load(req.bytes);
          return null;
        case "reranker":
          reranker?.free();
          reranker = WasmReranker.load(req.bytes);
          return null;
        case "chat":
          chat?.free();
          chat = WasmChat.load(req.bytes, req.modelId ?? null);
          return null;
      }
    }

    case "encode": {
      if (!model) throw new Error("no encoder loaded; call load({ encoder }) first");
      // The binding returns every vector concatenated; split it back out here so
      // callers get one array per input rather than doing the arithmetic.
      const flat = model.encode(req.texts, req.normalize);
      const dim = flat.length / req.texts.length;
      const out: Float32Array[] = [];
      for (let i = 0; i < req.texts.length; i++) {
        out.push(flat.slice(i * dim, (i + 1) * dim));
      }
      return out;
    }

    case "classify": {
      if (!classifier) throw new Error("no classifier loaded; call load({ classifier }) first");
      return req.texts.length === 1
        ? [classifier.classify(req.texts[0])]
        : classifier.classify_batch(req.texts);
    }

    case "labels":
      if (!classifier) throw new Error("no classifier loaded");
      return classifier.labels();

    case "rerank":
      if (!reranker) throw new Error("no reranker loaded; call load({ reranker }) first");
      return reranker.rerank(req.query, req.documents, req.limit);

    case "contextSize":
      if (!chat) throw new Error("no chat model loaded");
      return chat.context_size();

    case "chat": {
      if (!chat) throw new Error("no chat model loaded; call load({ chat }) first");
      // Genuinely incremental: the callback fires between decoding steps, and
      // each postMessage reaches the page while the model is still generating.
      // This is the reason the whole wrapper is worth having, since the same
      // call on the main thread would freeze the tab until the reply finished.
      chat.generate_stream(req.prompt, req.maxNewTokens, req.temperature, (piece: string) => {
        post({ id: req.id, kind: "token", text: piece });
      });
      return null;
    }

    case "close": {
      model?.free();
      classifier?.free();
      reranker?.free();
      chat?.free();
      model = classifier = reranker = chat = null;
      return null;
    }
  }
}

self.onmessage = async (event: MessageEvent<Request>) => {
  const req = event.data;
  try {
    await ensureWasm();
    const value = handle(req);
    post({ id: req.id, kind: "ok", value });
  } catch (e) {
    post({ id: req.id, kind: "err", message: e instanceof Error ? e.message : String(e) });
  }
};
