FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY dummy_requirements.txt dummy_requirements.txt
COPY main.py main.py
WORKDIR /
RUN pip install -r dummy_requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "main.py"]