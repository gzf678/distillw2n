from u2ss2u import StreamableModel
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
import os
from pathlib import Path
from tqdm import tqdm

DEVICE="cuda:5"

# 配置路径
whisper_dir = '/mnt/workspace/guanzifan/wtimit_16k/aligned_data_test/whisper/'  # whisper音频目录（输入）
normal_dir = '/mnt/data/share/wtimit_16k/nist/TEST/normal/'  # normal音频目录（用于提取说话人嵌入）
output_dir = '/mnt/workspace/guanzifan/eval_tmp/distill_wtimit_test_align'  # 输出音频目录

# 初始化模型
print("Loading models...")
model = StreamableModel(
    batch_size=42,
    sample_rate=16_000,
    segment_length=32270,
    padding='same',
    dataset='timit')

checkpoint_path = '/mnt/workspace/guanzifan/distillw2n/experiments/s2uu2s/epoch.440-step.409942.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(DEVICE)
model.eval()

hubert_soft = torch.hub.load("bshall/hubert:main", f"hubert_soft").to(DEVICE)

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
speaker_model = speaker_model.to(DEVICE)
speaker_model.eval()

# 初始化编码器和解码器
reencoder = model.reencoder.to(DEVICE)
decoder = model.decoder.to(DEVICE)

# 递归查找所有whisper音频文件
whisper_path = Path(whisper_dir)
normal_path = Path(normal_dir)
output_path = Path(output_dir)

whisper_files = list(whisper_path.rglob("*.WAV"))
print(f"Found {len(whisper_files)} whisper audio files to process")

# 预处理：为每个说话人找到对应的normal文件（缓存）
speaker_normal_cache = {}

def get_speaker_normal_file(whisper_file_path):
    """
    根据whisper文件路径，找到对应说话人的normal文件
    例如：whisper/SG/000/s000u074w.WAV -> normal/SG/000/s000u036n.WAV
    """
    # 获取相对于whisper_dir的路径，例如：SG/000/s000u074w.WAV
    rel_path = whisper_file_path.relative_to(whisper_path)
    
    # 提取说话人路径部分（SG/000），去掉文件名
    speaker_path = rel_path.parent  # SG/000
    
    # 如果已经缓存，直接返回
    if speaker_path in speaker_normal_cache:
        return speaker_normal_cache[speaker_path]
    
    # 构建对应的normal目录路径
    normal_speaker_dir = normal_path / speaker_path
    
    # 查找该说话人目录下的normal文件（以n结尾的.WAV文件）
    normal_files = list(normal_speaker_dir.glob("*n.WAV"))
    
    if len(normal_files) == 0:
        # 如果没有找到以n结尾的，尝试找所有.WAV文件
        normal_files = list(normal_speaker_dir.glob("*.WAV"))
    
    if len(normal_files) == 0:
        return None
    
    # 使用第一个找到的normal文件
    normal_file = normal_files[0]
    speaker_normal_cache[speaker_path] = normal_file
    return normal_file

# 处理每个音频文件
for whisper_file in tqdm(whisper_files, desc="Processing audio files"):
    try:
        # 找到对应说话人的normal文件
        normal_file = get_speaker_normal_file(whisper_file)
        
        if normal_file is None:
            print(f"Warning: No normal file found for speaker in {whisper_file}, skipping")
            continue
        
        if not normal_file.exists():
            print(f"Warning: Normal file does not exist: {normal_file}, skipping")
            continue
        
        # 加载normal音频并提取说话人嵌入
        x_trg, sr_trg = torchaudio.load(str(normal_file))
        # 处理多声道
        if x_trg.shape[0] > 1:
            x_trg = x_trg[0:1]
        x_trg = torchaudio.functional.resample(x_trg, sr_trg, 16000)
        spkemb = speaker_model.infer_segment(x_trg.squeeze(0))[0]
        
        # 加载whisper音频（source）
        x, sr = torchaudio.load(str(whisper_file))
        # 处理多声道
        if x.shape[0] > 1:
            x = x[0:1]
        x = torchaudio.functional.resample(x, sr, 16000)
        
        # 提取hubert特征
        hubert = hubert_soft.units(x.unsqueeze(0).to(DEVICE))
        hubert = hubert.clone().to(DEVICE)
        hubert = torch.transpose(hubert, -1, -2)
        
        # 转换语音（使用同一说话人的normal音频作为reference）
        z = reencoder(hubert.to(DEVICE), spkemb.to(DEVICE))
        z = decoder(z.to(DEVICE))
        
        # 计算输出路径（保持目录结构）
        relative_path = whisper_file.relative_to(whisper_path)
        output_file = output_path / relative_path
        
        # 创建输出目录
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存转换后的音频
        torchaudio.save(str(output_file), z.squeeze(1).detach().cpu(), 16000)
        
    except Exception as e:
        print(f"Error processing {whisper_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print(f"Processing complete! Output saved to {output_dir}")