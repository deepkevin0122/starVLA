FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# make python/pip available as `python`/`pip`
RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

# pip base tools
RUN pip install -r requirements.txt

RUN pip install flash-attn==2.7.4.post1 --no-build-isolation

# default
CMD ["/bin/bash"]
