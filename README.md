# FinClaw

A lightweight financial signal pipeline that:
- pulls market data (yfinance)
- fetches news (RSS)
- computes simple signals & alerts
- generates a summary via an LLM
- emails a daily report using SendGrid

## 📦 What’s included

- `utils.py` – all reusable functions (data fetch, signals, prompt, summarization, send email)
- `main.py` – single entrypoint that runs everything end-to-end
- `Dockerfile` + `cron/finclaw.cron` – builds a container that runs once at startup and then runs daily at 6am Mon–Fri
- `requirements.txt` – Python dependencies

## 🚀 Quick start (local)

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Set required environment variables:

```bash
export SENDGRID_API=os.environ.get('SENDGRID_API')
export FROM_EMAIL=os.environ.get('FROM_EMAIL')
export TO_EMAIL=os.environ.get('TO_EMAIL')
```

3) Run:

```bash
python main.py
```

## 🐳 Run with Docker (recommended for scheduled runs)

Build:

```bash
docker build -t finclaw .
```

Run with Docker:

```bash
docker run --rm \
  -e SENDGRID_API="$SENDGRID_API" \
  -e FROM_EMAIL="$FROM_EMAIL" \
  -e TO_EMAIL="$TO_EMAIL" \
  finclaw
```

Run with Docker Compose (recommended):

```bash
# create a .env file containing SENDGRID_API, FROM_EMAIL, TO_EMAIL

docker compose up --build
```

The container will:
1. Run once at startup
2. Then run daily at 6:00 AM Monday–Friday (via cron)

## ✅ Notes

- Ensure the `FROM_EMAIL` is verified in your SendGrid account (required for sending).
- Cron logs are written to `/app/logs/cron.log` inside the container.
- If you need to change the schedule, edit `cron/finclaw.cron`.
