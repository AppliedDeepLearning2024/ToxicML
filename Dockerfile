FROM python:3.12-slim

RUN pip install poetry

RUN mkdir -p /saved_models/deep
run mkdir ToxicMl
COPY ToxicMl ToxicMl/
COPY saved_models/deep saved_models/deep/
run ls -la

COPY saved_models/deep saved_models/
COPY poetry.lock .
COPY poetry.toml .
COPY pyproject.toml .

RUN poetry install --no-root

ENV PYTHONPATH /

CMD ["poetry", "run", "streamlit", "run", "ToxicMl/Api/main.py"]
