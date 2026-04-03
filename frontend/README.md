# React Visualization

This frontend is a Vite + React + TypeScript client for the sustainability-aware
workload scheduling simulator.

## Local development

Start the FastAPI backend from the repository root:

```bash
python3 -m uvicorn src.api:app --reload
```

Then start the frontend:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api/*` requests to `http://127.0.0.1:8000`.

## Environment override

If you want to point the frontend at a different backend, create a local
`.env` file in this directory and set:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```
