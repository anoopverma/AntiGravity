FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5002
EXPOSE 5002

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
