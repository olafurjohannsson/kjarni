// Loads the built WASM bundle in a real browser and checks it works.
//
// Everything else that tests kjarni-wasm runs the Rust natively. That covers the
// engine and misses the artifact: a broken import, a bundle built from stale
// sources, a binding that throws only under wasm-bindgen's glue. All of those
// ship green today, which is how the live site served a bundle without
// classification or chat for days.
//
//   KJARNI_KJQ_DIR=/tmp/kjq node run.mjs
//
// Needs `crates/kjarni-wasm/pkg` to exist (wasm-pack build --target web) and a
// directory of .kjq fixtures.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { join, extname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = fileURLToPath(new URL(".", import.meta.url));
const pkgDir = process.env.KJARNI_PKG_DIR ?? join(here, "../../pkg");
const modelDir = process.env.KJARNI_KJQ_DIR;

if (!modelDir) {
  console.error("KJARNI_KJQ_DIR must point at a directory of .kjq fixtures.");
  console.error("Build one with scripts/quantize_model.py; see tests/browser/README.md.");
  process.exit(2);
}

// instantiateStreaming refuses anything that is not application/wasm, and the
// module graph will not load without the right JS type either.
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".kjq": "application/octet-stream",
};

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  let file;
  if (url.pathname.startsWith("/pkg/")) file = join(pkgDir, basename(url.pathname));
  else if (url.pathname.startsWith("/models/")) file = join(modelDir, basename(url.pathname));
  else file = join(here, "harness.html");

  try {
    const body = await readFile(file);
    res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
    res.end(body);
  } catch (e) {
    res.writeHead(404).end(String(e));
  }
});

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

let failures = 0;
const check = (ok, what, detail = "") => {
  if (ok) {
    console.log(`  ok    ${what}`);
  } else {
    failures++;
    console.log(`  FAIL  ${what}${detail ? `  (${detail})` : ""}`);
  }
};

const browser = await chromium.launch();
const page = await browser.newPage();

// A page error would otherwise surface as an unexplained timeout.
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(e.message));
page.on("console", (m) => m.type() === "error" && pageErrors.push(m.text()));

try {
  await page.goto(`http://127.0.0.1:${port}/harness.html`);
  const result = await page.evaluate(() => window.__run());

  check(pageErrors.length === 0, "page loads without errors", pageErrors.join("; "));
  check(result.missing.length === 0, "every expected class is exported", `missing: ${result.missing}`);
  check(result.dim === 384, "MiniLM returns 384 dimensions", `got ${result.dim}`);
  check(result.related > 0.7, "doctor ~ physician", `cosine ${result.related.toFixed(4)}`);
  check(result.unrelated < 0.4, "doctor !~ banana", `cosine ${result.unrelated.toFixed(4)}`);
  check(
    result.related > result.unrelated + 0.2,
    "the related pair ranks well above the unrelated one",
    `${result.related.toFixed(4)} vs ${result.unrelated.toFixed(4)}`,
  );
} catch (e) {
  failures++;
  console.log(`  FAIL  harness threw: ${e.message}`);
  if (pageErrors.length) console.log(`        page errors: ${pageErrors.join("; ")}`);
} finally {
  await browser.close();
  server.close();
}

console.log(`\n${failures === 0 ? "all checks passed" : `${failures} failed`}`);
process.exit(failures === 0 ? 0 : 1);
