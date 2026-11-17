# The daemon's web UI

Svelte 5, built with Vite into `dist/`. The `kjarni` binary embeds `dist/`
(see `src/commands/daemon/web.rs`), so building Kjarni needs no Node. After
changing anything here, rebuild and commit `dist/` with it:

```sh
npm ci
npm test        # the transcript logic
npm run build   # writes dist/
```

To work on the UI against a running daemon, with reloads on save:

```sh
kjarni daemon run                             # or any daemon, on any port
KJARNI_URL=http://127.0.0.1:7710 npm run dev  # then open the printed address
```

Sign the dev page in with a link from `kjarni daemon open`, changing its
address to the dev server's.
