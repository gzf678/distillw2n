# 训练指南 - Multi-Whisper 数据集 Finetune

## 概述

本指南说明如何使用三个耳语-正常语音配对数据集进行模型 finetune：
1. **wtimit**: `/mnt/workspace/guanzifan/data/wtimit_16k/aligned_pairs_uid.json`
2. **aishell6**: `/mnt/data/share/AISHELL6-Whisper/aligned_pairs_uid_new.json`
3. **WHSP_LGU**: `/mnt/data/share/WHSP_LGU/aligned_pairs_uid_new.json`

## 数据集格式

每个 JSON 文件包含配对的耳语和正常语音路径：

### wtimit 格式
```json
{
    "norm_path": "/path/to/normal.wav",
    "whsp_path": "/path/to/whisper.wav",
    "uid": "S012U163W",
    "dataset": "wtimit",
    "speaker": "s012",
    "duration": 3.4365625
}
```

### aishell6 格式
```json
{
    "norm_path": "/path/to/normal.wav",
    "whsp_path": "/path/to/whisper.wav",
    "uid": "S0105_M-0105-1_052556-053228",
    "dataset": "aishell6",
    "duration": 6.72
}
```

### WHSP_LGU 格式
```json
{
    "norm_path": "/path/to/normal.wav",
    "whsp_path": "/path/to/whisper.wav",
    "uid": "user074_task010",
    "dataset": "WHSP_LGU",
    "speaker": "user074"
}
```

## 代码修改说明

### 1. 新增数据集类

创建了 `datahelper/multi_whisper_dataset.py`，实现了 `MultiWhisperDataset` 类：
- 自动加载多个 JSON 文件
- 处理不同格式的 JSON（有些有 speaker，有些没有）
- 返回格式：`(normal_waveform, whisper_waveform, vad_waveform, sample_rate)`

### 2. 更新训练配置

在 `u2ss2u.py` 的 `train()` 函数中：
- `dataset`: 设置为 `'multi_whisper'`
- `resume_training`: 设置为 `True`（从 checkpoint finetune）
- `restore_checkpoint_path`: 指向官方 checkpoint
- `version`: 更新为 `"s2uu2s-multi-whisper-finetune"`

### 3. 数据集路径

代码中硬编码了三个 JSON 文件路径：
```python
json_paths = [
    '/mnt/workspace/guanzifan/data/wtimit_16k/aligned_pairs_uid.json',
    '/mnt/data/share/AISHELL6-Whisper/aligned_pairs_uid_new.json',
    '/mnt/data/share/WHSP_LGU/aligned_pairs_uid_new.json',
]
```

## 使用方法

### 1. 确保数据文件存在

检查 JSON 文件和音频文件是否存在：
```bash
# 检查 JSON 文件
ls -lh /mnt/workspace/guanzifan/data/wtimit_16k/aligned_pairs_uid.json
ls -lh /mnt/data/share/AISHELL6-Whisper/aligned_pairs_uid_new.json
ls -lh /mnt/data/share/WHSP_LGU/aligned_pairs_uid_new.json

# 检查 checkpoint
ls -lh /mnt/workspace/guanzifan/distillw2n/experiments/s2uu2s/epoch.440-step.409942.ckpt
```

### 2. 运行训练

```bash
cd /mnt/workspace/guanzifan/distillw2n
python u2ss2u.py
```

### 3. 训练配置

当前配置（在 `u2ss2u.py` 的 `train()` 函数中）：
- **Batch size**: 32
- **Learning rate**: 1e-6
- **Checkpoint**: `./experiments/s2uu2s/epoch.440-step.409942.ckpt`
- **Max epochs**: 10000
- **Precision**: 16-mixed
- **Strategy**: ddp_find_unused_parameters_true

### 4. 监控训练

训练日志会保存到：
```
experiments/s2uu2s/s2uu2s-multi-whisper-finetune/
```

使用 TensorBoard 查看：
```bash
tensorboard --logdir experiments/s2uu2s/s2uu2s-multi-whisper-finetune/
```

## 关键修改点

1. **数据集加载** (`u2ss2u.py` line 289-298):
   - 添加了 `'multi_whisper'` 数据集选项
   - 使用 `MultiWhisperDataset` 加载三个 JSON 文件

2. **验证数据集** (`u2ss2u.py` line 343-350):
   - 验证时也使用 `MultiWhisperDataset`
   - 使用 10% 的数据作为验证集

3. **Checkpoint 加载** (`u2ss2u.py` line 365, 405):
   - `resume_training: True` 启用 checkpoint 加载
   - `ckpt_path` 指向官方 checkpoint

## 注意事项

1. **数据格式兼容性**：
   - 新数据集返回格式与原有数据集兼容
   - VAD 波形使用正常语音（因为新数据集没有单独的 VAD）

2. **Checkpoint 路径**：
   - 确保 checkpoint 路径正确：`./experiments/s2uu2s/epoch.440-step.409942.ckpt`
   - 如果路径不同，修改 `restore_checkpoint_path`

3. **多 GPU 训练**：
   - 代码使用 `ddp_find_unused_parameters_true` 策略
   - 支持多 GPU 分布式训练

4. **数据加载**：
   - 使用 7 个 worker 进程加载数据
   - 如果内存不足，可以减少 `num_workers`

## 故障排除

### 问题 1: JSON 文件找不到
**解决**: 检查 JSON 文件路径是否正确

### 问题 2: 音频文件找不到
**解决**: 检查 JSON 中的路径是否有效，音频文件是否存在

### 问题 3: Checkpoint 加载失败
**解决**: 
- 检查 checkpoint 路径
- 确保 checkpoint 文件完整
- 检查模型架构是否匹配

### 问题 4: 内存不足
**解决**:
- 减少 `batch_size`
- 减少 `num_workers`
- 减少 `segment_length`

## 预期输出

训练开始时会看到：
```
Loading 1234 whisper-normal pairs from 3 JSON file(s)
Loading checkpoint from: ./experiments/s2uu2s/epoch.440-step.409942.ckpt
...
```

训练过程中会记录：
- `g_loss`: Generator loss
- `d_loss`: Discriminator loss
- `val_pesq`: Validation PESQ score
- 其他损失项（f0_loss, energy_loss, content_loss, spk_loss）

