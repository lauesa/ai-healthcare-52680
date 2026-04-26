FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    git \
    curl \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

RUN pip install --no-cache-dir --break-system-packages \
    "numpy>=1.26.4,<2.0.0" \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir --break-system-packages \
    spacy \
    scispacy \
    thinc

RUN pip install --no-cache-dir --break-system-packages \
    vllm \
    pandas \
    psycopg2-binary \
    scikit-learn \
    rouge-score \
    meteor \
    nvidia-ml-py 

RUN pip install --no-cache-dir --break-system-packages \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

RUN mkdir -p /root/.local/share/opencode
ENV CUDA_VISIBLE_DEVICES=0,1

CMD ["/bin/bash"]