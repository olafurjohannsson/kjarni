// Times the wasm bundle in Chromium. Same serving setup as run.mjs.
//
//   KJARNI_KJQ_DIR=/tmp/kjq node bench.mjs [model.kjq]
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { join, extname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = fileURLToPath(new URL(".", import.meta.url));
const pkgDir = process.env.KJARNI_PKG_DIR ?? join(here, "../../pkg");
const modelDir = process.env.KJARNI_KJQ_DIR;
const modelFile = process.argv[2] ?? "all-MiniLM-L6-v2-q8.kjq";
if (!modelDir) { console.error("KJARNI_KJQ_DIR must be set"); process.exit(2); }

const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
               ".wasm": "application/wasm", ".kjq": "application/octet-stream" };

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  let file;
  if (url.pathname.startsWith("/pkg/")) file = join(pkgDir, basename(url.pathname));
  else if (url.pathname.startsWith("/models/")) file = join(modelDir, basename(url.pathname));
  else file = join(here, "bench.html");
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
  await page.goto(`http://127.0.0.1:${port}/bench.html`);
  const out = await page.evaluate((m) => window.__bench(m), modelFile);
  console.log(`\n  ${modelFile}   load ${out.loadMs.toFixed(0)} ms\n`);
  console.log(`  ${"shape".padEnd(26)}${"ms".padStart(10)}`);
  for (const r of out.rows) console.log(`  ${r.label.padEnd(26)}${r.ms.toFixed(2).padStart(10)}`);
  console.log(`  fp ${out.fp.map((x) => x.toFixed(6)).join(" ")}`);
} catch (e) {
  console.log(`  FAILED: ${e.message}`);
  if (errors.length) console.log(`  page errors: ${errors.join("; ")}`);
  process.exitCode = 1;
} finally { await browser.close(); server.close(); }
