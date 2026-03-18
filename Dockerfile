FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cron \
        build-essential \
        gcc \
        g++ \
        python3-dev \
        python3-venv \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
# Base image already includes PyTorch; installing it again can be slow or mismatch CUDA.
RUN python - <<'PY'
import pathlib, re

req_path = pathlib.Path("requirements.txt")
lines = req_path.read_text().splitlines()
filtered = [line for line in lines if line.strip() != "torch"]

out = pathlib.Path("/tmp/requirements.filtered.txt")
out.write_text("\n".join(filtered).strip() + "\n")
print(f"Filtered requirements written to {out}")
PY

RUN pip install --no-cache-dir -r /tmp/requirements.filtered.txt

COPY . /app

# Ensure logs directory exists
RUN mkdir -p /app/logs

# Install cron schedule
COPY cron/finclaw.cron /etc/cron.d/finclaw
RUN chmod 0644 /etc/cron.d/finclaw && \
    crontab /etc/cron.d/finclaw

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
