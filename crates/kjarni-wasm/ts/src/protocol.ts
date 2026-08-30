// Messages between the page and the worker.
//
// Every request carries an id and is answered by exactly one Ok or Err with the
// same id, except `chat`, which streams any number of Token messages before its
// Ok. Keeping that contract in one file is what lets both sides be type-checked
// against the same shapes rather than agreeing by convention.

/** A model to load, either already in memory or somewhere to fetch it from. */
export type ModelSource = string | URL | ArrayBuffer | Uint8Array;

export type Capability = "encoder" | "classifier" | "reranker" | "chat";

export interface ClassifyResult {
  label: string;
  score: number;
  index: number;
}

export interface RankedDocument {
  index: number;
  score: number;
  text: string;
}

export type Request =
  | { id: number; kind: "load"; capability: Capability; bytes: Uint8Array; modelId?: string }
  | { id: number; kind: "encode"; texts: string[]; normalize: boolean }
  | { id: number; kind: "classify"; texts: string[] }
  | { id: number; kind: "labels" }
  | { id: number; kind: "rerank"; query: string; documents: string[]; limit: number }
  | { id: number; kind: "chat"; prompt: string; maxNewTokens: number; temperature: number }
  | { id: number; kind: "contextSize" }
  | { id: number; kind: "close" };

export type Response =
  | { id: number; kind: "ok"; value: unknown }
  | { id: number; kind: "token"; text: string }
  | { id: number; kind: "err"; message: string };
