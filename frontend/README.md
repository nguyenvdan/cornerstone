# Cornerstone — frontend

Clean, fast, mobile-friendly React (Vite + TypeScript) dashboard for the
Cornerstone projection system. It is fully **static**: it reads precomputed JSON
artifacts produced by the Python pipeline (no backend required), so it deploys
anywhere as a static site.

## Views
- **Projection** — tier-probability distribution, expected career VORP with an
  80% interval, season-by-season development curve, and swing factors.
- **Comparables** — explained historical analogs.
- **Wizards build** — roster fit, per-skill need vs. supply gaps, recommended
  archetypes, current rotation.
- **Ask the agent** — the agent's reasoning trace + synthesized, cited report.
- **Methodology** — the leakage-free back-test, calibration plot, honest limits.

## Develop / build

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # -> dist/
```

The data files in `public/data/` and `public/calibration.png` are regenerated
from the models with `make frontend-data` (run from the repo root). Re-run that
whenever the models change, then rebuild.

## Deploy to Vercel

The app is a standard static Vite build — Vercel auto-detects it.

1. Push the repo to GitHub (already done).
2. In Vercel, **New Project → import the repo**.
3. Set **Root Directory** to `frontend`.
4. Framework preset: **Vite** (auto-detected). Build command `npm run build`,
   output dir `dist`.
5. Deploy → you get a public URL.

No environment variables are needed (the site is static). To refresh the data,
run `make frontend-data` locally, commit, and Vercel redeploys.
