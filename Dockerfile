# SafeNet live sensor for Linux hosts, gateways and SPAN/mirror-port boxes.
#   docker build -t safenet-sensor .
#   docker run -d --name safenet-sensor --restart unless-stopped --net=host \
#       --cap-add NET_RAW --cap-add NET_ADMIN -e DATABASE_URL='postgresql://...' safenet-sensor [--iface eth0]
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libpcap0.8 tcpdump \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-sensor.txt .
# CPU-only torch first, from PyTorch's CPU index alone (PyPI's aarch64/x86 wheels pull in CUDA libraries)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==$(grep '^torch==' requirements-sensor.txt | cut -d= -f3)" \
    && pip install --no-cache-dir -r requirements-sensor.txt
COPY safenet/ safenet/
COPY scripts/ scripts/
COPY sql/ sql/
# Docker caches this layer: pass --build-arg MODELS_REFRESH=$(date +%s) to pull models re-published under the same tag
ARG MODELS_REFRESH=0
RUN echo "models refresh $MODELS_REFRESH" && python -m scripts.fetch_models
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "scripts.sensor"]
