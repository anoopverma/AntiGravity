FROM python:3.11-slim

WORKDIR /app

# Build/runtime packages needed by scientific and native Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	gcc \
	g++ \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5002
EXPOSE 5002

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
