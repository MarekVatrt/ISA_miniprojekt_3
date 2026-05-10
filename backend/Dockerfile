FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt

# Nastavenie timeoutov pre pip
ENV PIP_DEFAULT_TIMEOUT=300

# Inštalácia balíčkov s retry mechanikou
RUN pip install --no-cache-dir \
    --timeout 300 \
    --retries 5 \
    -r requirements.txt

COPY backend/app /app/app
COPY data /app/data
COPY results /app/results

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]