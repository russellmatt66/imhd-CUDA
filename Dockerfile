FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libeigen3-dev \
        libhdf5-openmpi-dev \
        libopenmpi-dev \
        openmpi-bin \
        python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/imhd-cuda
COPY src/ src/
COPY include/ include/
COPY lib/ lib/
COPY build/simulation_launcher.py build/
COPY build/on-device/input.inp build/on-device/

ARG CUDA_ARCHITECTURES="60;61;75"
RUN cmake -S src -B build -DARCH_ONDEV=ON -DHDF5_PREFER_PARALLEL=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    && cmake --build build --parallel 2 \
    && useradd --create-home --user-group imhd \
    && mkdir -p data/on-device \
    && chown imhd:imhd data/on-device

USER imhd
WORKDIR /opt/imhd-cuda/build
CMD ["python3", "simulation_launcher.py", "nodiff"]

