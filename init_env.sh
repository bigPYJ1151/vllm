set -ex

PYTHON_VERSION="3.12"

echo 'export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"' >> ~/.bashrc
echo 'export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"' >> ~/.bashrc
echo 'export UV_INDEX_STRATEGY="unsafe-best-match"' >> ~/.bashrc
echo 'export UV_LINK_MODE="copy"' >> ~/.bashrc
echo 'export CCACHE_DIR=/home/ubuntu/.cache/ccache' >> ~/.bashrc
echo 'export CMAKE_CXX_COMPILER_LAUNCHER=ccache' >> ~/.bashrc
echo 'export PATH="$PATH:/home/ubuntu/.local/bin/:/home/ubuntu/.venv/bin/"' >> ~/.bashrc

export PATH="/home/ubuntu/.local/bin/:/home/ubuntu/.venv/bin/:$PATH"
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
export UV_PYTHON_INSTALL_DIR="/home/ubuntu/uv/python"
export UV_INDEX_STRATEGY="unsafe-best-match"
export UV_LINK_MODE="copy"

sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends ccache git curl wget ca-certificates python3-pip python3-venv python3-dev python3-pytest pre-commit \
        gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 jq lsof vim numactl xz-utils make clangd-14 \
    && sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

sudo ln -sf /usr/bin/clangd-14 /usr/bin/clangd

VLLM_DIR="$(pwd)"
cd ~/

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python ${PYTHON_VERSION} --seed ~/.venv

cd $VLLM_DIR
source ~/.venv/bin/activate

uv pip install --upgrade pip
uv pip install -r requirements/cpu.txt

echo 'export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/home/ubuntu/.venv/lib/libiomp5.so"' >> ~/.bashrc

cp requirements/test.in requirements/cpu-test.in 
rm requirements/test.txt
sed -i 's/^torch==.*/torch==2.9.1/g' requirements/cpu-test.in && \
sed -i 's/torchaudio.*/torchaudio/g' requirements/cpu-test.in && \
sed -i 's/torchvision.*/torchvision/g' requirements/cpu-test.in && \
uv pip compile requirements/cpu-test.in -o requirements/test.txt --index-strategy unsafe-best-match --torch-backend cpu

uv pip install -r requirements/cpu-build.txt
uv pip install -e tests/vllm_test_utils
uv pip install -r requirements/dev.txt

pre-commit install --hook-type pre-commit --hook-type commit-msg

git config --global user.name "jiang1.li"
git config --global user.email "jiang1.li@intel.com"
git config --global credential.helper store
git config --global core.editor "vim"

echo 'source /home/ubuntu/.venv/bin/activate' >> ~/.bashrc