FROM python:3.10.6-slim-buster

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip && \
  pip install -r requirements.txt

COPY api /api
COPY trainer /trainer
COPY assets /assets

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
