import esbuild from "esbuild";
import process from "process";

const prod = process.argv[2] === "production";

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
};

const workerShared = {
	bundle: true,
	platform: "browser",
	format: "iife",
	target: "es2018",
	logLevel: "info",
	sourcemap: prod ? false : "inline",
	treeShaking: true,
	minify: prod,
	define: {
		"import.meta.url": "''",
	},
};

const mainContext = await esbuild.context({
	...mainShared,
	entryPoints: ["main.ts"],
	outfile: "main.js",
});

const workerContext = await esbuild.context({
	...workerShared,
	entryPoints: ["worker.ts"],
	outfile: "worker.js",
});

const encoderContext = await esbuild.context({
	...workerShared,
	entryPoints: ["encoder-worker.ts"],
	outfile: "encoder-worker.js",
});

if (prod) {
	await mainContext.rebuild();
	await workerContext.rebuild();
	await encoderContext.rebuild();
	process.exit(0);
} else {
	await mainContext.watch();
	await workerContext.watch();
	await encoderContext.watch();
}
