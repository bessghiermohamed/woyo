FROM python:3.12-slim

# git is needed for pip to install woyo straight from GitHub
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin the release so the Space is reproducible (override to upgrade)
ARG WOYO_REF=v0.3.0
RUN pip install --no-cache-dir "woyo[search,extract] @ git+https://github.com/bessghiermohamed/woyo.git@${WOYO_REF}"

COPY app.py .

ENV PORT=7860 PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["python", "app.py"]
