from u2ss2u import StreamableModel
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
import os
from pathlib import Path
from tqdm import tqdm
import warnings

# Try to import BigVGAN2 utilities
try:
    from bigvgan_utils import load_bigvgan2_from_huggingface, extract_mel_from_audio, mel_to_audio_with_bigvgan
    BIGVGAN_UTILS_AVAILABLE = True
except ImportError:
    BIGVGAN_UTILS_AVAILABLE = False
    warnings.warn("bigvgan_utils not available. BigVGAN2 support will be disabled.")

# Configuration: Set to True to use BigVGAN2 vocoder, False to use original decoder
USE_BIGVGAN2 = True  # Set this to True to use BigVGAN2, False to use original decoder

DEVICE="cuda:5"

# 配置路径
input_dir = '/mnt/data/share/wtimit_16k/nist/TEST/whisper/'  # 输入音频目录
output_dir = '/mnt/workspace/guanzifan/eval_tmp/distill_w2n_wtimit_new'  # 输出音频目录
# reference_audio = '/mnt/workspace/guanzifan/data/LibriSpeech/test-clean/8463/287645/8463-287645-0005.flac'  # 参考说话人音频（用于提取说话人嵌入）
reference_audio="/mnt/workspace/guanzifan/data/LibriSpeech/test-clean/1188/133604/1188-133604-0023.flac"

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

# 加载参考音频并提取说话人嵌入
print(f"Loading reference audio: {reference_audio}")
x_trg, sr = torchaudio.load(reference_audio)
spkemb = speaker_model.infer_segment(x_trg.squeeze(0))[0]

# 初始化编码器和解码器
reencoder = model.reencoder.to(DEVICE)
decoder = model.decoder.to(DEVICE)

# Try to load BigVGAN2 if requested
bigvgan_model = None
bigvgan_config = None
if USE_BIGVGAN2 and BIGVGAN_UTILS_AVAILABLE:
    try:
        print("Attempting to load BigVGAN2 vocoder...")
        bigvgan_model, bigvgan_config = load_bigvgan2_from_huggingface(device=DEVICE)
        if bigvgan_model is not None:
            print("✓ BigVGAN2 loaded successfully!")
            print(f"  Using model with config: {bigvgan_config.get('sample_rate', 'unknown')} Hz")
        else:
            print("⚠ Failed to load BigVGAN2, falling back to original decoder")
            USE_BIGVGAN2 = False
    except ImportError as e:
        print(f"⚠ BigVGAN2 utilities not available: {e}")
        print("  Install BigVGAN2: pip install bigvgan or clone from https://github.com/NVIDIA/BigVGAN")
        print("  Falling back to original decoder")
        USE_BIGVGAN2 = False
    except Exception as e:
        import traceback
        print(f"⚠ Error loading BigVGAN2: {e}")
        print("  Full error details:")
        print(traceback.format_exc())
        print("  Falling back to original decoder")
        USE_BIGVGAN2 = False

if USE_BIGVGAN2 and bigvgan_model is None:
    USE_BIGVGAN2 = False
    print("⚠ BigVGAN2 not available, using original decoder")

# 递归查找所有音频文件
input_path = Path(input_dir)
output_path = Path(output_dir)
# Try multiple audio formats
audio_files = list(input_path.rglob("*.wav")) + list(input_path.rglob("*.WAV")) + list(input_path.rglob("*.flac")) + list(input_path.rglob("*.FLAC"))

print(f"Found {len(audio_files)} audio files to process")
print(f"Using {'BigVGAN2' if USE_BIGVGAN2 and bigvgan_model is not None else 'original decoder'} vocoder")

# 处理每个音频文件
for audio_file in tqdm(audio_files, desc="Processing audio files"):
    try:
        # 加载音频
        x, sr = torchaudio.load(str(audio_file))
        x = torchaudio.functional.resample(x, sr, 16000)
        
        # 提取hubert特征
        hubert = hubert_soft.units(x.unsqueeze(0).to(DEVICE))
        hubert = hubert.clone().to(DEVICE)
        hubert = torch.transpose(hubert, -1, -2)
        
        # 转换语音
        z = reencoder(hubert.to(DEVICE), spkemb.to(DEVICE))
        
        if USE_BIGVGAN2 and bigvgan_model is not None:
            # Use original decoder to get audio, then extract mel and use BigVGAN2
            # This approach uses BigVGAN2's superior vocoder quality
            z_audio = decoder(z.to(DEVICE))
            
            # Extract mel spectrogram from decoder output using BigVGAN's meldataset
            # This ensures compatibility with BigVGAN2's expected mel format
            from bigvgan_utils import extract_mel_from_audio, mel_to_audio_with_bigvgan
            
            # Extract mel using BigVGAN's mel computation (ensures compatibility)
            mel = extract_mel_from_audio(z_audio, bigvgan_model, sample_rate=16000)
            
            # Convert mel to audio using BigVGAN2
            # The function handles resampling automatically if needed
            z = mel_to_audio_with_bigvgan(mel, bigvgan_model, sample_rate=16000)
            z = z.unsqueeze(1)  # Add channel dimension for consistency
        else:
            # Use original decoder
            z = decoder(z.to(DEVICE))
        
        # 计算输出路径（保持目录结构）
        relative_path = audio_file.relative_to(input_path)
        output_file = output_path / relative_path
        
        # 创建输出目录
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存转换后的音频
        torchaudio.save(str(output_file), z.squeeze(1).detach().cpu(), 16000)
        
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        continue

print(f"Processing complete! Output saved to {output_dir}")

# from u2ss2u import StreamableModel
# import torch
# import torchaudio
# import nemo.collections.asr as nemo_asr
# import os
# from pathlib import Path
# from tqdm import tqdm

# DEVICE="cuda:0"

# # 配置路径
# input_dir = '/mnt/data/share/chains/solo/'  # 输入音频目录
# output_dir = '/mnt/workspace/guanzifan/eval_tmp/distill_w2n'  # 输出音频目录
# reference_audio = "/mnt/workspace/guanzifan/data/LibriSpeech/test-clean/8463/287645/8463-287645-0005.flac"  # 参考说话人音频（用于提取说话人嵌入）
# # 如果设置为 None，将使用输入音频本身来提取说话人嵌入（保持原音色）
# # 如果设置为路径，例如 './raw/gt/s000u003n.wav'，将使用该音频的音色进行转换

# # 初始化模型
# print("Loading models...")
# model = StreamableModel(
#     batch_size=42,
#     sample_rate=16_000,
#     segment_length=32270,
#     padding='same',
#     dataset='timit')

# checkpoint_path = '/mnt/workspace/guanzifan/distillw2n/experiments/s2uu2s/epoch.440-step.409942.ckpt'
# checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True)
# model.load_state_dict(checkpoint['state_dict'], strict=False)
# model = model.to(DEVICE)
# model.eval()

# hubert_soft = torch.hub.load("bshall/hubert:main", f"hubert_soft").to(DEVICE)

# speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
# speaker_model = speaker_model.to(DEVICE)
# speaker_model.eval()

# # 如果提供了参考音频，则加载并提取说话人嵌入
# # 否则将在处理每个文件时使用输入音频本身
# if reference_audio is not None:
#     print(f"Loading reference audio: {reference_audio}")
#     x_trg, sr = torchaudio.load(reference_audio)
#     # 处理多声道
#     if x_trg.shape[0] > 1:
#         x_trg = x_trg[0:1]
#     x_trg = torchaudio.functional.resample(x_trg, sr, 16000)
#     global_spkemb = speaker_model.infer_segment(x_trg.squeeze(0))[0]
#     use_global_spkemb = True
#     print("Using global reference audio for voice conversion")
# else:
#     global_spkemb = None
#     use_global_spkemb = False
#     print("No reference audio provided, will use input audio itself (preserving original voice)")

# # 初始化编码器和解码器
# reencoder = model.reencoder.to(DEVICE)
# decoder = model.decoder.to(DEVICE)

# # 递归查找所有音频文件
# input_path = Path(input_dir)
# output_path = Path(output_dir)
# audio_files = list(input_path.rglob("*.WAV"))

# print(f"Found {len(audio_files)} audio files to process")

# # 处理每个音频文件
# for audio_file in tqdm(audio_files, desc="Processing audio files"):
#     try:
#         # 加载音频
#         x, sr = torchaudio.load(str(audio_file))
        
#         # 检查输入音频是否有效
#         if x.numel() == 0:
#             print(f"Warning: Empty audio file {audio_file}, skipping")
#             continue
        
#         # 处理多声道音频（取第一个声道）
#         if x.shape[0] > 1:
#             x = x[0:1]
        
#         x = torchaudio.functional.resample(x, sr, 16000)
        
#         # 检查重采样后的音频
#         if torch.isnan(x).any() or torch.isinf(x).any():
#             print(f"Warning: NaN or Inf in input audio {audio_file}, skipping")
#             continue
        
#         # 归一化输入音频（如果需要）
#         x_max = torch.abs(x).max()
#         if x_max > 1.0:
#             x = x / x_max
        
#         # 提取说话人嵌入
#         if use_global_spkemb:
#             # 使用全局参考音频的说话人嵌入
#             spkemb = global_spkemb
#         else:
#             # 使用输入音频本身提取说话人嵌入（保持原音色）
#             spkemb = speaker_model.infer_segment(x.squeeze(0))[0]
        
#         # 提取hubert特征
#         hubert = hubert_soft.units(x.unsqueeze(0).to(DEVICE))
#         hubert = hubert.clone().to(DEVICE)
#         hubert = torch.transpose(hubert, -1, -2)
        
#         # 转换语音
#         z = reencoder(hubert.to(DEVICE), spkemb.to(DEVICE))
#         z = decoder(z.to(DEVICE))
        
#         # 处理输出音频：移除批次维度并移到CPU
#         z = z.squeeze(1).detach().cpu()
        
#         # 检查并处理 NaN 和 Inf 值
#         if torch.isnan(z).any() or torch.isinf(z).any():
#             print(f"Warning: NaN or Inf detected in output for {audio_file}, replacing with zeros")
#             z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
        
#         # 归一化音频到 [-1, 1] 范围
#         z_max = torch.abs(z).max()
#         if z_max > 0:
#             z = z / (z_max + 1e-8) * 0.95  # 0.95 避免削波
#         else:
#             print(f"Warning: Zero audio detected for {audio_file}")
        
#         # 裁剪到有效范围
#         z = torch.clamp(z, -1.0, 1.0)
        
#         # 计算输出路径（保持目录结构）
#         relative_path = audio_file.relative_to(input_path)
#         output_file = output_path / relative_path
        
#         # 创建输出目录
#         output_file.parent.mkdir(parents=True, exist_ok=True)
        
#         # 保存转换后的音频
#         torchaudio.save(str(output_file), z, 16000)
        
#     except Exception as e:
#         print(f"Error processing {audio_file}: {str(e)}")
#         continue

# print(f"Processing complete! Output saved to {output_dir}")