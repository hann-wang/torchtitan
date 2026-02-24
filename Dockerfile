FROM rocm/pytorch-nightly:2026-02-17-rocm7.2

RUN apt-get update && apt-get install -y \
    git-lfs \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-pciids

RUN pip install --no-cache-dir huggingface_hub "datasets>=3.6.0" \
    transformers tabulate wandb fsspec tyro "tokenizers>=0.15.0" safetensors \
    tensorboard pre-commit yapf pybind11 meson-python torchdata pytablewriter \
    "antlr4-python3-runtime==4.11.0" sympy math_verify more_itertools peft \
    accelerate pillow

RUN git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness && \
    cd lm-evaluation-harness && \
    pip install -e .

# COPY . /usr/local/src/torchtitan

# RUN cd /usr/local/src/torchtitan && \
#     pip install --no-build-isolation --no-deps -e .
