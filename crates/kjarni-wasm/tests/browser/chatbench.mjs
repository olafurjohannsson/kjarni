// Times greedy generation in Chromium and prints the text, which doubles as the
// correctness check for the Q8_0 kernel: greedy output must not change.
//
//   KJARNI_KJQ_DIR=/tmp/kjq node chatbench.mjs [model.kjq] [tokens]
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { join, extname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = fileURLToPath(new URL(".", import.meta.url));
const pkgDir = process.env.KJARNI_PKG_DIR ?? join(here, "../../pkg");
const modelDir = process.env.KJARNI_KJQ_DIR;
const modelFile = process.argv[2] ?? "qwen05b-kjq8.kjq";
const tokens = Number(process.argv[3] ?? 16);
if (!modelDir) { console.error("KJARNI_KJQ_DIR must be set"); process.exit(2); }

const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
               ".wasm": "application/wasm", ".kjq": "application/octet-stream" };
const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  let file;
  if (url.pathname.startsWith("/pkg/")) file = join(pkgDir, basename(url.pathname));
  else if (url.pathname.startsWith("/models/")) file = join(modelDir, basename(url.pathname));
  else file = join(here, "chatbench.html");
  try {
    const body = await readFile(file);
    res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
    res.end(body);
  } catch (e) { res.writeHead(404).end(String(e)); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
try {
  await page.goto(`http://127.0.0.1:${port}/chatbench.html`);
  const o = await page.evaluate(([m, t]) => window.__chat(m, t), [modelFile, tokens]);
  console.log(`  load ${o.loadMs.toFixed(0)} ms`);
  console.log(`  ${o.tokens} tokens in ${o.genMs.toFixed(0)} ms  =  ${(o.tokens / (o.genMs / 1000)).toFixed(2)} tok/s`);
  console.log(`  text: ${JSON.stringify(o.text)}`);
} catch (e) {
  console.log(`  FAILED: ${e.message}`);
  if (errors.length) console.log(`  page errors: ${errors.join("; ")}`);
  process.exitCode = 1;
} finally { await browser.close(); server.close(); }
