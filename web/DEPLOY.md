# Deployment Guide for NutriTaste ML Service

This guide explains how to deploy the **NutriTaste ML Service** so it runs 24/7 with a public URL.

---

## Why GitHub Pages or Vercel are not suitable

1. **GitHub Pages:**
   * Only hosts static HTML/CSS/JS. It has no Python runtime and cannot execute Scikit-Learn or load `recipe_model.joblib`.
2. **Vercel:**
   * Designed for small JavaScript serverless functions (max uncompressed bundle 250 MB).
   * Our 100+ MB model file plus `scikit-learn` and `pandas` exceeds standard serverless bundle limits and triggers cold-start memory/timeout crashes.

---

## Recommended Free & Always-On Platforms

### Option 1: Hugging Face Spaces (Best for ML Portfolios, 100% Free)

* **Specs:** 16 GB RAM, 2 vCPU, persistent Docker container, free HTTPS domain.
* **Steps to deploy:**
  1. Create a free account at [huggingface.co](https://huggingface.co).
  2. Click **New Space** -> Choose **Docker** as Space SDK (Blank template).
  3. Set visibility to **Public**.
  4. Clone your new Space repository locally or push your files:
     ```bash
     git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
     ```
  5. Place the `Dockerfile` from `web/Dockerfile` in the root of the Space, along with `web/` and `src/` directories.
  6. Commit and push:
     ```bash
     git add .
     git commit -m "Deploy NutriTaste ML"
     git push space main
     ```
  7. Hugging Face builds the Docker container automatically in ~2 minutes and gives you a live public URL: `https://<username>-<space-name>.hf.space`.

---

### Option 2: Render.com (Web Service, Free Tier)

* **Steps to deploy:**
  1. Sign up at [render.com](https://render.com).
  2. Click **New +** -> **Web Service**.
  3. Connect your GitHub repository.
  4. Settings:
     * **Environment:** Python or Docker
     * If Docker: Point Dockerfile path to `web/Dockerfile` and Docker Context to `.`
     * **Plan:** Free
  5. Click **Create Web Service**.
  * *Tip:* To prevent free-tier sleep after 15 min of inactivity, add your Render URL to a free uptime monitor like [cron-job.org](https://cron-job.org) or [uptimerobot.com](https://uptimerobot.com) to ping `https://your-app.onrender.com/api/health` every 10 minutes.

---

### Option 3: Cheap VPS (Timeweb Cloud, Aeza, Hetzner, etc.)

* Run directly with Docker on your own server ($2-$3/month):
  ```bash
  docker build -f web/Dockerfile -t nutritaste-ml .
  docker run -d -p 80:7860 --restart always --name nutritaste nutritaste-ml
  ```

