FROM python:3.12-slim

RUN pip install poetry

COPY . .

RUN poetry install

CMD ["poetry", "run", "fastapi", "run", "ToxicMl/Api/main.py", "--port", "8081"]
