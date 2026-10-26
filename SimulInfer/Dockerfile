FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE NOTICE /app/
COPY simulinfer /app/simulinfer
ARG WITH_VAD=1
RUN if [ "$WITH_VAD" = "1" ]; then \
      pip install --no-cache-dir 'torch>=2.2' --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir '.[vad]'; \
    else pip install --no-cache-dir .; fi
EXPOSE 8080
ENTRYPOINT ["simulinfer"]
CMD ["--host", "0.0.0.0", "--config", "/app/config.toml"]
