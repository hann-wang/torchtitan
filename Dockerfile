FROM rocm/pytorch-nightly:2025-09-21-rocm6.4.2

RUN apt-get update && apt-get install -y \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "tomli>=1.1.0" huggingface_hub datasets transformers tabulate \
    wandb fsspec tyro "tokenizers>=0.15.0" safetensors tensorboard pre-commit yapf

COPY . /usr/local/src/torchtitan

RUN cd /usr/local/src/torchtitan && \
    pip install -e .
