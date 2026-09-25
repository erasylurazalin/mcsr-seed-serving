FROM python:3.13-slim

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY serving/ serving/
COPY model/ model/

RUN useradd --create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
