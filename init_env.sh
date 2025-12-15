set -ex

echo 'export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"' >> ~/.bashrc
echo 'export CCACHE_DIR=~/.cache/ccache' >> ~/.bashrc
echo 'export CMAKE_CXX_COMPILER_LAUNCHER=ccache' >> ~/.bashrc

export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends ccache git curl wget ca-certificates python3-pip python3-venv pre-commit \
        gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 jq lsof vim numactl xz-utils make clangd-14 \
    && sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

sudo ln -sf /usr/bin/clangd-14 /usr/bin/clangd

pip install --upgrade pip --break-system-packages
pip install -r requirements/cpu.txt --break-system-packages

echo 'export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/home/ubuntu/.local/lib/libiomp5.so"' >> ~/.bashrc
source ~/.bashrc

pip install -r requirements/cpu-build.txt --break-system-packages
pip install -e tests/vllm_test_utils --break-system-packages
pip install -r requirements/lint.txt --break-system-packages

pre-commit install --hook-type pre-commit --hook-type commit-msg

git config --global user.name "jiang1.li"
git config --global user.email "jiang1.li@intel.com"
git config --global credential.helper store
git config --global core.editor "vim"
