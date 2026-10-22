# Uploading IntelliBot to GitHub from an Android phone

IntelliBot needs its **own new repository** (do not use your SmartAssist repository).
Suggested name: `intellibot-context-aware-rag-chatbot`

> These steps were written carefully but **not tested on a phone** by me. If one step
> looks different on your screen, the idea is the same.

## One-time: create the repository
1. Open https://github.com in Chrome and sign in.
2. Tap **+ → New repository**.
3. Name: `intellibot-context-aware-rag-chatbot`. Choose **Public** (needed for free Streamlit Cloud) or Private.
4. Tick **Add a README**? **No** - leave it unticked. Tap **Create repository**.

## Every day: upload the new ZIP
GitHub's phone upload page cannot upload folders, so the easiest way is **GitHub Codespaces**
(a free online code editor in your browser; it has a monthly free allowance - check your GitHub billing page).

1. Download the day's ZIP (for example `intellibot-day1.zip`) to your phone.
2. Open your repository → tap **Code → Codespaces → Create codespace on main**. Wait for the editor to open.
3. In the left file list, tap the **⋮ / ...** menu (or long-press empty space) → **Upload...** and choose the ZIP.
4. Open the **Terminal** (menu → Terminal → New Terminal) and run these three lines:
   ```
   unzip -o intellibot-day1.zip
   cp -r intellibot/. . && rm -rf intellibot intellibot-day1.zip
   git status
   ```
5. Save the work online:
   ```
   git add .
   git commit -m "Day 1: project setup and document loader"
   git push
   ```
6. Open your repository page and check that the files are there. The commit date proves your daily progress.
7. Stop the codespace when done (menu → **Stop Current Codespace**) to save your free hours.

Each ZIP contains the **whole project so far**, so step 4 simply updates older files.
(Replace `intellibot-day1.zip` with the name of the ZIP you are uploading. No file has ever been removed between versions, so copying over the old files is safe.)

## Final upload (Day 7) to your EXISTING repository - checklist
1. ☐ Download **`intellibot-day7-final.zip`** to your phone (it is the complete project; earlier ZIPs are not needed).
2. ☐ GitHub app/Chrome → your repository → **Code → Codespaces** → open your existing codespace (or **Create codespace on main**).
3. ☐ Left file list → **⋮ → Upload...** → choose the ZIP.
4. ☐ Terminal:
   ```
   unzip -o intellibot-day7-final.zip
   cp -r intellibot/. . && rm -rf intellibot intellibot-day7-final.zip
   git status
   ```
5. ☐ Read the `git status` list: it must **not** show `.env`. (If you created a `.env` for your key it is ignored automatically and will not appear.)
6. ☐ Commit and push:
   ```
   git add .
   git commit -m "Day 7: final evaluation, regression testing and documentation"
   git push
   ```
7. ☐ Open the repository page and check that `README.md`, `app.py`, `core/`, `tests/`, `docs/` and `data/` are there and the commit is new.
8. ☐ **Do the real checks now** (still in the codespace; this is what turns "tested with stand-ins" into "tested for real"):
   ```
   pip install -r requirements-dev.txt
   python -m pytest -q                      # the real-ChromaDB / real-Streamlit tests now run instead of being skipped
   cp .env.example .env                     # then open .env and paste YOUR key after OPENAI_API_KEY=
   streamlit run app.py                     # Codespaces shows "Open in Browser"
   python scripts/run_evaluation.py --mode local
   ```
   Write down every error message you see; they are the next things to fix.
9. ☐ Stop the codespace (menu → **Stop Current Codespace**).

## Suggested commit messages
| Day | Message |
|---|---|
| 1 | `Day 1: project setup, config and PDF/TXT document loader` |
| 2 | `Day 2: text chunking and embedding pipeline` |
| 3 | `Day 3: persistent vector store and semantic retrieval` |
| 4 | `Day 4: grounded LLM answers and source attribution` |
| 5 | `Day 5: conversation memory and session isolation` |
| 6 | `Day 6: polished Streamlit chat interface and mobile UX` |
| 7 | `Day 7: final evaluation, regression testing and documentation` |

## Never upload
- `.env` (your real keys) - it is already blocked by `.gitignore`
- API keys, tokens or passwords in any file
- `.venv/`, `__pycache__/`, `chroma_db/` folders
- the old ZIP files (the `.gitignore` already skips `*.zip`)

## If a key is ever uploaded by mistake
Delete/rotate the key in your OpenAI or Hugging Face account immediately. Removing it from GitHub is not enough.

## Alternative: Termux (advanced)
Install Termux from F-Droid, then `pkg install git unzip`, unzip the project, and use `git push`
with a GitHub personal access token (enter it only when asked; never save it in a file).
