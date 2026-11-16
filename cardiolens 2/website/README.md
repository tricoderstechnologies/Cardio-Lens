# CardioLens AI website (static)
Plain HTML/CSS/JS, no build step, no external requests. Open `index.html` directly or host the folder.
All numbers come from `assets/data.js`, generated from the executed project outputs by `build_data.py`
(run `python website/build_data.py` from the project root after re-running the pipeline).
The site shows **precomputed** results; it does not run the Python models. Live predictions would need a separate
Python service (the Streamlit app in `src/app.py` is that service) and are not implemented here.
Not medical advice; not clinically validated.

## Deploy (all paths are relative)
**GitHub Pages:** push this folder's contents to a repo (or put it in `/docs`), then Settings > Pages > Deploy from branch > pick the branch and `/ (root)` or `/docs`. URL: `https://<user>.github.io/<repo>/`.
**Netlify:** app.netlify.com > Add new site > Deploy manually, drag the folder (or the ZIP contents); or connect the repo with publish directory set to this folder and no build command.
**Vercel:** `npm i -g vercel`, then `vercel --prod` inside this folder (framework "Other", no build command, output directory `.`), or import the repo and set the root directory to this folder.

When served by `uvicorn src.api:app` the site also gets live inference (see the project README). On static hosts it runs in saved-results mode automatically.
