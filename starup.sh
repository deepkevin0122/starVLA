#!/usr/bin/env bash
set -e

CODE_ROOT=/dataset/vkevinzhao/code
STARVLA_REPO=${CODE_ROOT}/starVLA
LIBERO_REPO=${CODE_ROOT}/LIBERO

echo "==> Using CODE_ROOT=${CODE_ROOT}"

if ! command -v conda &>/dev/null; then
  echo "[ERROR] conda not found. Please install Miniforge first."
  exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
  echo "[WARN] nvidia-smi not found. CUDA runtime may be unavailable."
  echo "       starVLA / torch GPU may not work."
fi

cd ${CODE_ROOT}
if [ ! -d "starVLA" ]; then
  echo "==> Cloning starVLA repository"
  git clone https://github.com/deepkevin0122/starVLA
else
  echo "==> starVLA repo already exists, skipping clone"
fi

cd ${STARVLA_REPO}
git pull

if ! conda env list | grep -q "^starVLA"; then
  echo "==> Creating conda env: starVLA (python=3.10)"
  conda create -n starVLA python=3.10 -y
else
  echo "==> Conda env starVLA already exists"
fi

echo "==> Activating starVLA env"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate starVLA

############################
# 3. 安装 starVLA 依赖
############################
echo "==> Installing starVLA python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

############################
# 4. 安装 FlashAttention2（可选但强烈建议）
############################
echo "==> Installing FlashAttention2 (requires CUDA toolchain)"
echo "    If this step fails, you can comment it out and continue."

set +e
pip install flash-attn --no-build-isolation
FLASH_STATUS=$?
set -e

if [ ${FLASH_STATUS} -ne 0 ]; then
  echo "[WARN] FlashAttention2 install failed."
  echo "       starVLA can still run, but may be slower."
  echo "       Common reasons:"
  echo "         - CUDA headers / nvcc not available in image"
  echo "         - CUDA version mismatch"
fi

############################
# 5. 安装 starVLA 本体
############################
echo "==> Installing starVLA (editable mode)"
pip install -e .

############################
# 6. 准备 LIBERO 代码
############################
cd ${CODE_ROOT}

if [ ! -d "LIBERO" ]; then
  echo "==> Cloning LIBERO repository"
  git clone https://github.com/deepkevin0122/LIBERO.git
else
  echo "==> LIBERO repo already exists, skipping clone"
fi

cd ${LIBERO_REPO}
git pull

############################
# 7. 创建 / 激活 LIBERO conda 环境
############################
if ! conda env list | grep -q "^libero"; then
  echo "==> Creating conda env: libero (python=3.10)"
  conda create -n libero python=3.10 -y
else
  echo "==> Conda env libero already exists"
fi

echo "==> Activating libero env"
conda activate libero

############################
# 8. 安装 LIBERO 依赖
############################
echo "==> Installing LIBERO dependencies"
pip install --upgrade pip
pip install -r requirements.txt

############################
# 9. 安装指定 torch + CUDA 11.3 版本（LIBERO）
############################
echo "==> Installing PyTorch 1.11.0 + cu113 for LIBERO"
pip install \
  torch==1.11.0+cu113 \
  torchvision==0.12.0+cu113 \
  torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113

############################
# 10. 额外工具依赖
############################
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4

############################
# 完成
############################
echo "======================================"
echo " Environment setup completed."
echo " starVLA env : conda activate starVLA"
echo " LIBERO env  : conda activate libero"
echo "======================================"
