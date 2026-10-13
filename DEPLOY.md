# Deploying to Google Cloud Run

## Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated (`gcloud init`)
- A Gemini API key

## One-time setup
```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

## Deploy
From the project root (where the Dockerfile lives):
```bash
gcloud run deploy priority-agent \
  --source . \
  --region australia-southeast1 \
  --allow-unauthenticated \
  --max-instances=1 \
  --set-env-vars GOOGLE_API_KEY=your_key_here,GEMINI_MODEL=gemini-flash-latest
```

`--source .` tells Cloud Run to build the container from the Dockerfile via Cloud Build automatically — no separate `docker build`/`push` step needed.

`--max-instances=1` matters here: `AgentState` is a single in-memory object living inside the running process (see `app/state.py`). Cloud Run can otherwise scale to multiple instances under load, each with its own separate copy of that state — so `/api/agent/state` could show different history depending on which instance handled the request. Capping at one instance keeps a single consistent state. It also means state is not guaranteed to survive a redeploy or a cold restart — this is a known limitation worth mentioning in your technical report's "State Management" or "Reflection" section; a production version would persist state to something like SQLite or Redis instead of an in-memory dataclass.

Note the env var name here is `GOOGLE_API_KEY`, matching `app/config.py` — not `GEMINI_API_KEY`. If you use the wrong name, the app will build and deploy fine but every classification will fail over to the safe fallback response, since `get_llm()` will raise "GOOGLE_API_KEY is not set."

## After deploying
`gcloud run deploy` prints a service URL like:
```
https://priority-agent-xxxxxxxxxx-ts.a.run.app
```
Open that URL directly to reach the frontend (the FastAPI app serves it at `/`) — that's the URL to submit and to show in the demo video.

## Updating after code changes
Re-run the same `gcloud run deploy` command — it rebuilds and creates a new revision automatically.

## Running locally instead (for comparison/demo)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key_here   # or use a .env file
uvicorn app.main:app --reload
```
Visit http://localhost:8000

## Running locally via Docker
```bash
docker build -t priority-agent .
docker run -p 8080:8080 -e GOOGLE_API_KEY=your_key_here priority-agent
```
Visit http://localhost:8080

This is also a good way to sanity-check the container before pushing it to Cloud Run, since it runs the exact same Dockerfile.
