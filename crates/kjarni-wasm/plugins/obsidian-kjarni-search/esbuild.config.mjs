import esbuild from "esbuild";
import fs from "fs";
import process from "process";

const prod = process.argv[2] === "production";

// Build workers FIRST, then inline them into main.js
const workerShared = {
    bundle: true,
    platform: "browser",
    format: "iife",
    target: "es2018",
    logLevel: "info",
    sourcemap: false,  // no sourcemaps in inlined workers
    treeShaking: true,
    minify: prod,
    define: {
        "import.meta.url": "''",
    },
};

// Build workers to temporary files
await esbuild.build({
    ...workerShared,
    entryPoints: ["worker.ts"],
    outfile: ".build/worker.js",
});

await esbuild.build({
    ...workerShared,
    entryPoints: ["encoder-worker.ts"],
    outfile: ".build/encoder-worker.js",
});

// Plugin that replaces WORKER_SOURCE / ENCODER_WORKER_SOURCE with actual code
const inlineWorkersPlugin = {
    name: "inline-workers",
    setup(build) {
        build.onResolve({ filter: /^inline:/ }, args => ({
            path: args.path,
            namespace: "inline-workers",
        }));
        build.onLoad({ filter: /.*/, namespace: "inline-workers" }, args => {
            const file = args.path.replace("inline:", "");
            const code = fs.readFileSync(`.build/${file}`, "utf8");
            return {
                contents: `export default ${JSON.stringify(code)};`,
                loader: "js",
            };
        });
    },
};

const mainShared = {
    bundle: true,
    platform: "node",
    external: [
        "obsidian",
        "electron",
        "@codemirror/autocomplete",
        "@codemirror/collab",
        "@codemirror/commands",
        "@codemirror/language",
        "@codemirror/lint",
        "@codemirror/search",
        "@codemirror/state",
        "@codemirror/view",
        "@lezer/common",
        "@lezer/highlight",
        "@lezer/lr",
    ],
    format: "cjs",
    target: "es2018",
    logLevel: "info",
    sourcemap: prod ? false : "inline",
    treeShaking: true,
    minify: prod,
    plugins: [inlineWorkersPlugin],
};

await esbuild.build({
    ...mainShared,
    entryPoints: ["main.ts"],
    outfile: "main.js",
});

process.exit(0);