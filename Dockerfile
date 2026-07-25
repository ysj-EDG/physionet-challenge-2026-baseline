FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=300

RUN python -m pip install --upgrade \
    pip \
    "setuptools<82" \
    wheel

# Install the CUDA-enabled PyTorch wheel.
RUN python -m pip install \
    --no-cache-dir \
    --retries 10 \
    --timeout 300 \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121

# Install project dependencies.
RUN python -m pip install \
    --no-cache-dir \
    --retries 10 \
    --timeout 300 \
    -r /challenge/requirements.txt

# Verify package dependency metadata.
RUN python -m pip check

# Verify imports required by the official execution path.
RUN python -c "import numpy, pandas, scipy, sklearn, edfio, neurokit2, torch; \
from per_epoch_features.per_epoch_extractor import PerEpochExtractor; \
import team_code; \
print('Dependency and project import checks passed'); \
print('numpy:', numpy.__version__); \
print('neurokit2:', neurokit2.__version__); \
print('torch:', torch.__version__); \
print('torch CUDA build:', torch.version.cuda)"
