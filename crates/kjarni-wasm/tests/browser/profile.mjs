// Samples a wasm encode in Chromium and reports self time by function.
//
// Chrome's profiler resolves wasm frames by name, so this says which kernel the
// browser actually spends its time in, rather than which one we assume it uses.
//
//   KJARNI_KJQ_DIR=/tmp/kjq node profile.mjs [model.kjq]
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
try {
  await page.goto(`http://127.0.0.1:${port}/bench.html`);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Profiler.enable");
  await cdp.send("Profiler.setSamplingInterval", { interval: 100 });
  await cdp.send("Profiler.start");
  await page.evaluate((m) => window.__bench(m), modelFile);
  const { profile } = await cdp.send("Profiler.stop");

  const byId = new Map(profile.nodes.map((n) => [n.id, n]));
  const self = new Map();
  for (const n of profile.nodes) {
    const name = n.callFrame.functionName || "(anonymous)";
    self.set(name, (self.get(name) ?? 0) + (n.hitCount ?? 0));
  }
  const total = [...self.values()].reduce((a, b) => a + b, 0);
  const rows = [...self.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18);
  console.log(`\n  ${profile.samples.length} samples, ${total} hits\n`);
  console.log(`  ${"self%".padStart(7)}  function`);
  for (const [name, hits] of rows) {
    if (hits / total < 0.004) continue;
    console.log(`  ${((hits / total) * 100).toFixed(2).padStart(6)}%  ${name.slice(0, 110)}`);
  }
} catch (e) {
  console.log(`  FAILED: ${e.message}`);
  process.exitCode = 1;
} finally { await browser.close(); server.close(); }
