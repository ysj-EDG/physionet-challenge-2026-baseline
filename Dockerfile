FROM python:3.10.1-buster

WORKDIR /challenge

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=300

COPY requirements.txt /challenge/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel

# Install the official CUDA PyTorch wheel first.
# Do not use CPU torch, local wheel caches, or regional mirrors for official builds.
RUN python -m pip install --no-cache-dir --retries 10 --timeout 300 \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121

RUN python -m pip install --no-cache-dir --retries 10 --timeout 300 \
    -r /challenge/requirements.txt

COPY . /challenge
