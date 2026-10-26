# MLP vs Teachable Machine · Animal Classifier

A static website (no server, no build step). Both models run in the browser with TensorFlow.js.

## Run it on your computer

Browsers block model loading from `file://`, so serve the folder:

```bash
cd animal-classifier
python -m http.server 8000
```

Open http://localhost:8000

## Publish it for free (GitHub Pages)

1. Create a new public repository on GitHub, e.g. `animal-classifier`.
2. Upload **the contents of this folder** (`index.html`, `app.js`, `style.css`, `vendor/`, `models/`, `gallery/`, `assets/`) to the repository root.
   - Drag and drop on github.com works for up to 100 files at a time, so use GitHub Desktop or `git` for the `gallery/img` folder (1,500 images):
     ```bash
     git init && git add . && git commit -m "Animal classifier demo"
     git branch -M main
     git remote add origin https://github.com/<your-username>/animal-classifier.git
     git push -u origin main
     ```
3. Repository → **Settings → Pages** → Source: *Deploy from a branch* → Branch: `main`, folder `/ (root)` → Save.
4. After a minute your site is live at `https://<your-username>.github.io/animal-classifier/`. The webcam works there because GitHub Pages uses https.

Total size is about 30 MB, well under GitHub's limits.

## What is in the folder

| Path | What it is |
|---|---|
| `index.html`, `style.css`, `app.js` | The website |
| `models/mlp/` | MLP weights (`mlp_weights.bin`) and layer description (`mlp_manifest.json`), exported from the Keras model |
| `models/tm/` | Teachable Machine TensorFlow.js export (`model.json`, `metadata.json`, `weights.bin`) |
| `fonts/` | Fraunces and Nunito (SIL Open Font License 1.1), self-hosted so the site works offline |
| `vendor/` | TensorFlow.js 1.7.4 and the Teachable Machine image library 0.8.5 (bundled, so no CDN is needed) |
| `gallery/` | The 1,500 held-out test images and `gallery.json` with both models' predictions on them |
| `assets/` | Confusion-matrix figures for the Results tab |

## Adding another model later (CNN, transfer learning, ...)

In `app.js`, every model is an object with `id`, `name`, `cls`, `info` and an async `predict(canvas224)` that returns 10 probabilities in the class order of `mlp_manifest.json`. Load it in `boot()` and `state.models.push(model)`. Result cards, heatmaps, stress tests and sweeps pick it up automatically (add a colour for its `cls` in `style.css`).

## Notes

- Images are centre-cropped to a square. The MLP receives them as 64 × 64 pixels, Teachable Machine as 224 × 224.
- Predictions in `gallery/gallery.json` were produced by this same page's code, so the Test explorer matches what you see when you open an image in "Try it".
- The footer says "Models trained by Alina Irshad"; edit it in `index.html` if you prefer something else.
