FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=300

RUN python -m pip install --upgrade pip setuptools wheel

# Install the official CUDA PyTorch wheel first.
# Do not use CPU torch, local wheel caches, or regional mirrors for official builds.
RUN python -m pip install --no-cache-dir --retries 10 --timeout 300 \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121

## Include the following line if you have a requirements.txt file.
RUN python -m pip install --no-cache-dir --retries 10 --timeout 300 \
    -r /challenge/requirements.txt
