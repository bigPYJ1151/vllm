FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ENV http_proxy="http://child-prc.intel.com:913"

ENV https_proxy="http://child-prc.intel.com:913"

ENV TZ=Asia/Shanghai

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y \
    && apt-get install -y git ninja-build make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl vim cmake systemd numactl

RUN apt-get install -y gcc-12 g++-12 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ENV PYENV_ROOT="$HOME/.pyenv"

ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

RUN set -ex && curl https://pyenv.run | bash

RUN pyenv install 3.10.0 && pyenv global 3.10.0 && python -m sysconfig

WORKDIR /root/

RUN git clone https://github.com/llvm/llvm-project.git && cd llvm-project && git checkout b8b2a279d002 && cmake -G Ninja -S ./llvm -B ./build -DCMAKE_BUILD_TYPE=Release -DLLVM_INSTALL_UTILS=ON -DCMAKE_INSTALL_PREFIX=/usr/lib/llvm && ninja -C ./build install && cmake -G Ninja -S ./openmp -B ./build-omp -DLLVM_ROOT=/usr/lib/llvm  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ && ninja -C ./build-omp

RUN pip install --upgrade pip

RUN pip install wheel packaging ninja setuptools>=49.4.0

RUN pip install torch==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu

ENV LD_LIBRARY_PATH="/.pyenv/versions/3.10.0/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH"

RUN git clone -b perf_data  https://github.com/bigPYJ1151/vllm.git && cd vllm && make py_install_cpu

RUN rm /.pyenv/versions/3.10.0/lib/python3.10/site-packages/torch/lib/libgomp-a34b3233.so.1 && cp llvm-project/build-omp/runtime/src/libgomp.so /.pyenv/versions/3.10.0/lib/python3.10/site-packages/torch/lib/libgomp-a34b3233.so.1

