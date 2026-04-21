FROM rocm/pytorch-nightly:2026-02-17-rocm7.2

RUN apt-get update && apt-get install -y \
    git-lfs \
    pkg-config \
    clang \
    libclang-dev \
    libunwind-dev \
    libnl-3-dev \
    libnl-route-3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-pciids

RUN pip install --no-cache-dir huggingface_hub "datasets>=3.6.0" \
    transformers tabulate wandb fsspec tyro "tokenizers>=0.15.0" safetensors \
    tensorboard pre-commit yapf pybind11 meson-python torchdata pytablewriter \
    "antlr4-python3-runtime==4.11.0" sympy math_verify more_itertools peft \
    accelerate pillow "numpy<2" opencv-python-headless scipy \
    numba huggingface-hub[cli,hf_transfer] "packaging>=24.2" \
    "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8" \
    protobuf-protoc-bin fmt && \
    pip install --no-cache-dir /opt/rocm/share/amd_smi

RUN cd /var/lib/jenkins && \
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness && \
    cd lm-evaluation-harness && \
    pip install -e .

ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950"
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo

RUN rm -rf /root/.rustup && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

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
    . $CARGO_HOME/env && \
    git clone --recursive https://github.com/hann-wang/monarch.git && \
    cd monarch && \
    git checkout han/rocm && \
    pip install -r build-requirements.txt && \
    USE_TENSOR_ENGINE=1 \
    pip install --no-build-isolation -e .

RUN cd /var/lib/jenkins && \
    git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git checkout v0.17.0 && \
    sed -i 's/^opencv-python-headless.*//' requirements/common.txt && \
    sed -i 's/^ray.*//' requirements/common.txt && \
    pip install -r requirements/rocm.txt "numpy<2" && \
    python3 setup.py develop

# COPY . /usr/local/src/torchtitan

# RUN cd /usr/local/src/torchtitan && \
#     pip install --no-build-isolation --no-deps -e .

RUN sed -i 's/self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param))/self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param), requires_grad=param.requires_grad)/' /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py && \
    sed -i 's/        self.sharded_param.requires_grad_(param.requires_grad)//' /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py
