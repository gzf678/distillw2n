from u2ss2u import StreamableModel
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
import os
from pathlib import Path
from tqdm import tqdm

DEVICE = "cuda:5"

# 配置路径
whisper_dir = '/mnt/data/share/chains/whsp/'  # whisper音频目录（输入，耳语）
normal_dir = '/mnt/data/share/chains/solo/'  # normal音频目录（用于提取说话人嵌入，正常语音）
output_dir = '/mnt/workspace/guanzifan/eval_tmp/distill_w2n_chains'  # 输出音频目录

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

# 预处理：为每个说话人建立normal文件索引
speaker_normal_index = {}

def build_normal_index():
    """建立normal文件的索引：{speaker_id: [file_path1, file_path2, ...]}"""
    normal_files_wav = list(normal_path.rglob("*.wav"))
    normal_files_WAV = list(normal_path.rglob("*.WAV"))
    normal_files = list(set(normal_files_wav + normal_files_WAV))
    
    for normal_file in normal_files:
        # 获取相对于normal_dir的路径，例如：frf01/file1.wav
        rel_path = normal_file.relative_to(normal_path)
        # 提取说话人目录（frf01），这是父目录
        speaker_id = rel_path.parent.name  # frf01
        
        if speaker_id not in speaker_normal_index:
            speaker_normal_index[speaker_id] = []
        speaker_normal_index[speaker_id].append(normal_file)
    
    print(f"Indexed normal files for {len(speaker_normal_index)} speakers")
    for speaker_id, files in speaker_normal_index.items():
        print(f"  {speaker_id}: {len(files)} normal files")

def get_different_normal_file(whisper_file_path):
    """
    根据whisper文件路径，找到同一说话人但不同文件名的normal文件
    例如：whsp/frf01/file1.wav -> solo/frf01/file2.wav (同一说话人frf01，但不同文件名file1 vs file2)
    """
    # 获取相对于whisper_dir的路径，例如：frf01/file1.wav
    rel_path = whisper_file_path.relative_to(whisper_path)
    
    # 提取说话人ID（frf01），这是父目录名
    speaker_id = rel_path.parent.name  # frf01
    
    # 获取whisper文件名（不含路径）
    whisper_filename = rel_path.name  # file1.wav
    
    # 查找同一说话人的所有normal文件
    if speaker_id not in speaker_normal_index:
        return None
    
    speaker_normals = speaker_normal_index[speaker_id]
    
    # 找到文件名不同的normal文件（排除相同文件名的）
    available_normals = []
    for normal_file in speaker_normals:
        normal_filename = normal_file.relative_to(normal_path).name
        # 如果文件名不同，则可以使用（确保是不同的音频）
        if normal_filename != whisper_filename:
            available_normals.append(normal_file)
    
    if len(available_normals) == 0:
        return None
    
    # 使用第一个找到的不同文件名的normal文件
    return available_normals[0]

# 建立normal文件索引
build_normal_index()

# 处理每个音频文件
for whisper_file in tqdm(whisper_files, desc="Processing audio files"):
    try:
        # 找到同一说话人但不同文件名的normal文件
        normal_file = get_different_normal_file(whisper_file)
        
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
        
        # 转换语音（使用同一说话人但不同文件名的normal音频作为reference）
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