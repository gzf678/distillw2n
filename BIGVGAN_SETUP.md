# BigVGAN2 安装和使用说明

## 问题说明

BigVGAN 仓库没有 `setup.py` 或 `pyproject.toml`，所以不能使用 `pip install -e .` 安装。

## 解决方案

### 方法 1: 使用本地 BigVGAN 目录（推荐）

如果你已经克隆了 BigVGAN 仓库到项目目录：

```bash
# 1. 克隆 BigVGAN 源代码（如果还没有）
cd /mnt/workspace/guanzifan/distillw2n
git clone https://github.com/NVIDIA/BigVGAN.git

# 2. 安装依赖（可选，如果需要训练）
cd BigVGAN
pip install -r requirements.txt

# 3. 返回项目目录
cd ..
```

代码会自动检测并使用 `./BigVGAN` 目录中的代码。

### 方法 2: 安装依赖

如果只需要推理，确保安装了必要的依赖：

```bash
pip install torch torchaudio librosa huggingface_hub
```

### 方法 3: 使用 HuggingFace 模型（已下载）

如果你已经从 HuggingFace 下载了模型：

```bash
# 模型已经下载到: ./bigvgan_v2_24khz_100band_256x
# 代码会自动使用这个模型
```

## 验证安装

运行测试脚本：

```bash
python test_bigvgan_official.py
```

如果看到以下输出，说明安装成功：

```
✓ meldataset imported successfully
✓ bigvgan imported successfully
✓ Model loaded from HuggingFace
```

## 目录结构

确保你的项目目录结构如下：

```
distillw2n/
├── BigVGAN/              # BigVGAN 源代码（从 GitHub 克隆）
│   ├── bigvgan.py
│   ├── meldataset.py
│   └── ...
├── bigvgan_v2_24khz_100band_256x/  # HuggingFace 模型（可选）
│   └── ...
├── bigvgan_utils.py      # 工具函数
├── infer.py              # 推理脚本
└── ...
```

## 常见问题

### Q: 为什么不需要 `pip install -e .`？

A: BigVGAN 仓库没有标准的 Python 包结构。代码会自动将 `BigVGAN` 目录添加到 Python 路径中，所以可以直接导入。

### Q: 如果 BigVGAN 在其他位置怎么办？

A: 代码会自动搜索以下路径：
- `./BigVGAN`
- `../BigVGAN`
- `../../BigVGAN`

或者你可以手动添加到 Python 路径：

```python
import sys
sys.path.insert(0, '/path/to/BigVGAN')
```

### Q: 需要编译 CUDA 内核吗？

A: 对于推理，不需要。只有在使用 `use_cuda_kernel=True` 时才需要编译 CUDA 内核。

## 使用

现在可以直接运行推理：

```bash
python infer.py
```

代码会自动：
1. 检测并使用本地的 BigVGAN 代码
2. 从 HuggingFace 加载模型（或使用本地模型）
3. 使用 BigVGAN2 进行高质量音频生成


