FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    openjdk-17-jdk \
    git && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml .

RUN poetry config virtualenvs.create false

RUN poetry install --no-root

COPY . .

EXPOSE 8501

CMD ["python","main.py"]