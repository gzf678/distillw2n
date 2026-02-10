from u2ss2u import StreamableModel
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
import os
import re
from pathlib import Path
from tqdm import tqdm

DEVICE = "cuda:5"

# 配置路径
whisper_dir = '/mnt/workspace/guanzifan/data/AISHELL6-Whisper/aligned_test_data/whsp/'  # whisper音频目录（输入）
normal_dir = '/mnt/workspace/guanzifan/data/AISHELL6-Whisper/aligned_test_data/norm/'  # normal音频目录（用于提取说话人嵌入）
output_dir = '/mnt/workspace/guanzifan/data/AISHELL6-Whisper/distillw2n/'  # 输出音频目录

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

# 同时查找 .wav 和 .WAV 文件（不区分大小写）
whisper_files_wav = list(whisper_path.rglob("*.wav"))
whisper_files_WAV = list(whisper_path.rglob("*.WAV"))
whisper_files = list(set(whisper_files_wav + whisper_files_WAV))

print(f"Found {len(whisper_files)} whisper audio files to process")

# 预处理：为每个说话人建立normal文件索引（按说话人ID和句子ID）
speaker_normal_index = {}

def parse_filename(filename):
    """
    解析文件名，提取说话人ID和时间戳
    例如：S0189_M-0189-2_057112-057588.wav -> (S0189, '057112-057588')
          S0189_M-0189-1_012804-013364.wav -> (S0189, '012804-013364')
    """
    # 匹配格式：{speaker_id}_M-{speaker_id}-{type}_{timestamp}.wav
    # 例如：S0189_M-0189-2_057112-057588.wav
    pattern = r'([^_]+)_M-\d+-\d+_(\d+-\d+)\.wav'
    match = re.search(pattern, filename)
    if match:
        speaker_id = match.group(1)  # S0189
        timestamp = match.group(2)  # 057112-057588
        return speaker_id, timestamp
    return None, None

def build_normal_index():
    """建立normal文件的索引：{speaker_id: [file_path1, file_path2, ...]}"""
    normal_files_wav = list(normal_path.rglob("*.wav"))
    normal_files_WAV = list(normal_path.rglob("*.WAV"))
    normal_files = list(set(normal_files_wav + normal_files_WAV))
    
    for normal_file in normal_files:
        speaker_id, timestamp = parse_filename(normal_file.name)
        if speaker_id and timestamp:
            if speaker_id not in speaker_normal_index:
                speaker_normal_index[speaker_id] = []
            speaker_normal_index[speaker_id].append(normal_file)
    
    print(f"Indexed normal files for {len(speaker_normal_index)} speakers")
    for speaker_id, files in speaker_normal_index.items():
        print(f"  {speaker_id}: {len(files)} normal files")

def get_different_sentence_normal_file(whisper_file_path):
    """
    根据whisper文件路径，找到同一说话人但不同时间戳（不同话）的normal文件
    例如：whisper/S0189/S0189_M-0189-2_057112-057588.wav 
         -> normal/S0189/S0189_M-0189-1_012804-013364.wav (同一说话人，不同话)
    """
    whisper_filename = whisper_file_path.name
    speaker_id, whisper_timestamp = parse_filename(whisper_filename)
    
    if speaker_id is None or whisper_timestamp is None:
        return None
    
    # 查找同一说话人的所有normal文件
    if speaker_id not in speaker_normal_index:
        return None
    
    speaker_normals = speaker_normal_index[speaker_id]
    
    # 找到时间戳不同的normal文件（排除相同时间戳的，虽然whisper是2，normal是1，但时间戳应该不同）
    # 实际上whisper和normal的时间戳应该不会相同，但为了安全起见还是检查一下
    available_normals = []
    for normal_file in speaker_normals:
        _, normal_timestamp = parse_filename(normal_file.name)
        if normal_timestamp != whisper_timestamp:
            available_normals.append(normal_file)
    
    if len(available_normals) == 0:
        return None
    
    # 使用第一个找到的不同时间戳的normal文件
    return available_normals[0]

# 建立normal文件索引
build_normal_index()

# 处理每个音频文件
for whisper_file in tqdm(whisper_files, desc="Processing audio files"):
    try:
        # 找到同一说话人但不同句子的normal文件
        normal_file = get_different_sentence_normal_file(whisper_file)
        
        if normal_file is None:
            print(f"Warning: No matching normal file found for {whisper_file}, skipping")
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
        
        # 转换语音（使用同一说话人但不同句子的normal音频作为reference）
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