#!/usr/bin/env bash
# One-time setup + deploy to Railway.
# Checkpoints are gitignored but need to be included in the upload.
# This script temporarily lifts that exclusion for the duration of railway up.
set -e

echo "→ Checking Railway login…"
railway whoami 2>/dev/null || railway login

echo "→ Linking Railway project…"
# Init a new project if none is linked yet
railway status 2>/dev/null | grep -q "Project:" || railway init

echo "→ Temporarily including checkpoints in upload…"
cp .gitignore .gitignore.bak
grep -v "model/checkpoints" .gitignore > .gitignore.tmp && mv .gitignore.tmp .gitignore

echo "→ Deploying to Railway (this uploads ~40 MB of checkpoints)…"
railway up --detach || { mv .gitignore.bak .gitignore; echo "Deploy failed — .gitignore restored."; exit 1; }

echo "→ Restoring .gitignore…"
mv .gitignore.bak .gitignore

echo ""
echo "✓ Deployed. Next steps:"
echo "  1. Set env vars in Railway dashboard:"
echo "     ANTHROPIC_API_KEY=<your key>"
echo "     (HF_REPO_ID is optional — checkpoints are baked in)"
echo "  2. Copy your Railway service URL (e.g. https://tactical-world-model-xyz.up.railway.app)"
echo "  3. Add to Pelada's Vercel env vars:"
echo "     VITE_TACTICAL_MODEL_URL=<your Railway URL>"
echo "  4. Redeploy Pelada on Vercel."
