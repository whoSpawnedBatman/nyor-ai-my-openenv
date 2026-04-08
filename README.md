# Hospital Quotation OpenEnv

An OpenEnv-compatible benchmark environment that simulates hospital medicine quotation preparation using an LLM agent.

## Setup

```bash
pip install -r requirements.txt
```

Copy `env/.env.example` to `env/.env` and fill in your keys:
```
HF_TOKEN=your_openai_or_hf_api_key
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

## Run Inference

```bash
python inference.py
```

## Start the OpenEnv Server (HF Space)

```bash
python app.py
# Server at http://localhost:7860
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Reset environment, returns initial observation |
| `POST` | `/step` | Apply action, returns reward + observation |
| `GET`  | `/state` | Get current state |
| `GET`  | `/tasks` | List all available tasks |

## Tasks

| Task | Description |
|------|-------------|
| `quotation` | Full workflow: search → brand → supplier → confirm → price → finalize |
| `brand-selection` | Select cheapest valid brand + supplier |
| `margin-check` | Calculate sell price meeting 8% margin and finalize |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | API key |
