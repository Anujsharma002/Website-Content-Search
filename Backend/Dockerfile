FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}"

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python dependency installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies using uv (way faster than pip)
RUN uv pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY . .

EXPOSE 8000

# Run FastAPI backend using uv
CMD ["uv", "run", "main.py"]
