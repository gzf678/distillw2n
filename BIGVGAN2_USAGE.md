# 使用 BigVGAN2 Vocoder 的说明

## 概述

根据 README.md 的建议，本项目已添加了对 BigVGAN2 vocoder 的支持。BigVGAN2 是一个经过大规模训练的通用神经 vocoder，相比当前训练数据少于100小时的 vocoder，具有更好的音质和泛化能力。

## 安装 BigVGAN2

### 方法 1: 从 GitHub 安装

```bash
git clone https://github.com/NVIDIA/BigVGAN.git
cd BigVGAN
pip install -e .
```

### 方法 2: 使用 pip（如果可用）

```bash
pip install bigvgan
```

### 安装 HuggingFace Hub（用于下载预训练模型）

```bash
pip install huggingface_hub
```

## 使用方法

### 修改 infer.py

在 `infer.py` 文件的开头，有一个配置选项：

```python
USE_BIGVGAN2 = True  # 设置为 True 使用 BigVGAN2，False 使用原始 decoder
```

### 运行推理

```bash
python infer.py
```

如果 BigVGAN2 成功加载，你会看到：
```
✓ BigVGAN2 loaded successfully!
  Using model with config: 24000 Hz
Found X audio files to process
Using BigVGAN2 vocoder
```

如果 BigVGAN2 不可用，脚本会自动回退到原始的 decoder。

## 工作原理

当前的实现方式：

1. **使用原始 decoder 生成音频**：首先使用训练好的 decoder 生成音频波形
2. **提取 Mel Spectrogram**：从生成的音频中提取 mel spectrogram（使用 BigVGAN2 兼容的参数）
3. **BigVGAN2 Vocoder**：使用 BigVGAN2 将 mel spectrogram 转换回高质量音频

这种方式利用了 BigVGAN2 的高质量 vocoder，同时不需要重新训练模型。

## 注意事项

1. **采样率**：BigVGAN2 模型通常训练用于 24kHz，但我们的模型使用 16kHz。脚本会自动处理采样率转换。

2. **Mel 参数**：BigVGAN2 期望特定的 mel spectrogram 参数（100 mel bands, hop_length=256）。脚本会自动调整这些参数。

3. **性能**：使用 BigVGAN2 会增加推理时间，但会显著提升音质。

## 未来改进

理想情况下，应该修改 decoder 直接输出 mel spectrogram，然后使用 BigVGAN2 转换。这需要：
1. 修改 decoder 架构（输出 mel 而不是音频）
2. 重新训练模型

当前的实现是一个实用的折中方案，可以在不重新训练的情况下利用 BigVGAN2 的优势。

## 故障排除

### BigVGAN2 加载失败

如果看到 "Failed to load BigVGAN2"，请检查：
1. BigVGAN2 是否正确安装
2. 网络连接（需要下载 HuggingFace 模型）
3. CUDA 是否可用

脚本会自动回退到原始 decoder，不会中断推理过程。

### 内存不足

BigVGAN2 模型较大，如果遇到内存不足：
1. 使用较小的 batch size
2. 确保有足够的 GPU 内存
3. 考虑使用 CPU（较慢）

## 参考

- [BigVGAN GitHub](https://github.com/NVIDIA/BigVGAN)
- [BigVGAN HuggingFace](https://huggingface.co/nvidia/bigvgan_base_24khz_100band)
- [BigVGAN Paper](https://arxiv.org/abs/2206.04658)


