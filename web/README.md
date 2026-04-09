# web

Next.js 16 static site for the watch-a-model-grok visualization. See the root [README](../README.md) for project context and deploy notes.

```bash
pnpm install
pnpm dev        # http://localhost:3000
pnpm build      # static export to out/
```

Data bundle (`public/data/meta.json` + `*.bin` float32 arrays) is produced by `training/export_web.py` in the parent project.
