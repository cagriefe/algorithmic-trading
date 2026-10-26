FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# TA-Lib wheels bundle the C library, so no compiler or source build is needed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY algotrading ./algotrading
COPY main.py .

ENTRYPOINT ["python", "main.py"]
