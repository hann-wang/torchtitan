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

RUN pip install /opt/rocm/share/amd_smi
RUN cd /var/lib/jenkins && \
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness && \
    cd lm-evaluation-harness && \
    pip install -e .

ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950"

RUN cd /var/lib/jenkins && \
    git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git checkout v0.17.0 && \
    pip install --upgrade numba \
        scipy \
        huggingface-hub[cli,hf_transfer] \
        setuptools_scm && \
    pip install -r requirements/rocm.txt && \
    python3 setup.py develop

RUN rm -rf /root/.rustup && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . $HOME/.cargo/env

RUN apt-get update && apt-get install -y \
    pkg-config clang libclang-dev libunwind-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install protobuf-protoc-bin fmt

RUN cd /var/lib/jenkins && \
    git clone --recursive https://github.com/linux-rdma/rdma-core.git && \
    cd rdma-core && \
    git checkout v62.0 && \
    mkdir build && \
    cd build && \
    cmake -DNO_MAN_PAGES=1 .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

RUN cd /var/lib/jenkins && \
    . $HOME/.cargo/env && \
    git clone --recursive https://github.com/hann-wang/monarch.git && \
    cd monarch && \
    git checkout han/rocm && \
    pip install -r build-requirements.txt && \
    USE_TENSOR_ENGINE=1 \
    pip install --no-build-isolation -e .

# COPY . /usr/local/src/torchtitan

# RUN cd /usr/local/src/torchtitan && \
#     pip install --no-build-isolation --no-deps -e .
