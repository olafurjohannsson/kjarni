import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// Fixed file names: the daemon embeds dist/ by these paths.
export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    assetsInlineLimit: 0,
    rollupOptions: {
      output: {
        entryFileNames: 'assets/app.js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: (asset) =>
          asset.names?.some((n) => n.endsWith('.css')) ? 'assets/app.css' : 'assets/[name][extname]',
      },
    },
  },
  server: {
    // `npm run dev` talks to a daemon on the default port, or $KJARNI_URL.
    // The Host header passes through unchanged: the daemon checks that a
    // browser's Origin matches it.
    proxy: {
      '/api': { target: process.env.KJARNI_URL ?? 'http://127.0.0.1:7710', changeOrigin: false },
      '/login': { target: process.env.KJARNI_URL ?? 'http://127.0.0.1:7710', changeOrigin: false },
    },
  },
});
