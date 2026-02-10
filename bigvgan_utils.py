"""
Utilities for loading and using BigVGAN2 vocoder.
"""

import torch
import torch.nn as nn
import torchaudio
import warnings


def load_bigvgan2_from_huggingface(model_name="nvidia/bigvgan_v2_24khz_100band_256x", device='cuda', use_cuda_kernel=False):
    """
    Load BigVGAN2 model from HuggingFace using the official API.
    
    Based on the official HuggingFace documentation:
    https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x
    
    Official usage:
    ```python
    import bigvgan
    model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to(device)
    ```
    
    Args:
        model_name: HuggingFace model name (default: nvidia/bigvgan_v2_24khz_100band_256x)
        device: Device to load model on
        use_cuda_kernel: Whether to use CUDA kernel for faster inference
    
    Returns:
        model: BigVGAN2 model
        config: Model configuration dict with sample_rate, n_mels, etc.
    """
    try:
        # Try to import bigvgan - support multiple import methods
        try:
            import bigvgan
        except ImportError:
            # Try adding BigVGAN directory to path
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            bigvgan_paths = [
                os.path.join(current_dir, 'BigVGAN'),
                os.path.join(current_dir, '../BigVGAN'),
                os.path.join(current_dir, '../../BigVGAN'),
            ]
            for bigvgan_path in bigvgan_paths:
                if os.path.exists(bigvgan_path) and os.path.exists(os.path.join(bigvgan_path, 'bigvgan.py')):
                    if bigvgan_path not in sys.path:
                        sys.path.insert(0, bigvgan_path)
                    import bigvgan
                    print(f"  Loaded bigvgan from local path: {bigvgan_path}")
                    break
            else:
                raise ImportError(
                    "BigVGAN package not found. Please either:\n"
                    "  1. Install it: git clone https://github.com/NVIDIA/BigVGAN.git\n"
                    "  2. Or place BigVGAN directory in the project root"
                )
        
        print(f"Loading BigVGAN2 from HuggingFace: {model_name}")
        print(f"  Using device: {device}")
        print(f"  CUDA kernel: {use_cuda_kernel}")
        
        # Use the official from_pretrained method
        model = bigvgan.BigVGAN.from_pretrained(model_name, use_cuda_kernel=use_cuda_kernel)
        
        # Remove weight norm and set to eval mode (as per official docs)
        model.remove_weight_norm()
        model = model.eval().to(device)
        
        # Extract config from model.h
        config = {
            'sample_rate': model.h.sampling_rate,
            'n_mels': getattr(model.h, 'n_mel_channels', None) or getattr(model.h, 'n_mels', None) or getattr(model.h, 'num_mels', None),
            'hop_length': getattr(model.h, 'hop_size', None),
            'win_length': getattr(model.h, 'win_size', None),
            'n_fft': getattr(model.h, 'n_fft', None),
        }
        
        print(f"✓ BigVGAN2 model loaded successfully!")
        print(f"  Sample rate: {config['sample_rate']} Hz")
        if config['n_mels']:
            print(f"  Mel bands: {config['n_mels']}")
        if config['hop_length']:
            print(f"  Hop length: {config['hop_length']}")
        
        return model, config
        
    except ImportError as e:
        error_msg = (
            f"BigVGAN package not found. Please install it:\n"
            f"  1. git clone https://github.com/NVIDIA/BigVGAN.git\n"
            f"  2. cd BigVGAN\n"
            f"  3. pip install -e .\n"
            f"  4. pip install huggingface_hub librosa\n"
            f"\nOriginal error: {e}"
        )
        print(f"⚠ {error_msg}")
        return None, None
    except Exception as e:
        import traceback
        error_msg = f"Failed to load BigVGAN2 from HuggingFace: {e}"
        print(f"⚠ {error_msg}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        warnings.warn(error_msg)
        return None, None


def create_mel_decoder(original_decoder, n_mels=100, n_fft=1024, hop_length=256):
    """
    Create a modified decoder that outputs mel spectrogram.
    
    Note: This is a wrapper that modifies the decoder's output layer.
    The original decoder outputs audio waveform, we modify it to output mel spectrogram.
    
    Args:
        original_decoder: Original decoder that outputs audio
        n_mels: Number of mel bands
        n_fft: FFT size for mel computation
        hop_length: Hop length for mel computation
    
    Returns:
        Modified decoder that outputs mel spectrogram
    """
    class MelDecoderWrapper(nn.Module):
        def __init__(self, original_decoder, n_mels, n_fft, hop_length):
            super().__init__()
            # Keep the original decoder layers except the last one
            self.decoder_layers = nn.Sequential(*list(original_decoder.layers.children())[:-1])
            
            # Get the number of input channels for the mel output layer
            # The last layer of original decoder is: Conv1d(n_channels, 1, ...)
            # We need to find n_channels
            last_conv = original_decoder.layers[-1]
            n_channels = last_conv.in_channels
            
            # Create mel output layer
            # Output n_mels channels instead of 1
            padding = last_conv.padding[0] if isinstance(last_conv.padding, tuple) else last_conv.padding
            self.mel_output = nn.Conv1d(
                n_channels, 
                n_mels, 
                kernel_size=last_conv.kernel_size[0] if isinstance(last_conv.kernel_size, tuple) else last_conv.kernel_size,
                padding=padding
            )
            
            # Mel spectrogram parameters
            self.n_mels = n_mels
            self.n_fft = n_fft
            self.hop_length = hop_length
            
        def forward(self, x):
            # Get intermediate representation
            x = self.decoder_layers(x)
            # Output mel spectrogram
            mel = self.mel_output(x)
            # Apply log scaling (common for mel spectrograms used with vocoders)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            return mel
    
    return MelDecoderWrapper(original_decoder, n_mels, n_fft, hop_length)


def mel_to_audio_with_bigvgan(mel, vocoder, sample_rate=None):
    """
    Convert mel spectrogram to audio using BigVGAN2.
    
    Based on official usage:
    ```python
    with torch.inference_mode():
        wav_gen = model(mel)  # [B, 1, T_time]
    ```
    
    Args:
        mel: Mel spectrogram tensor [B, n_mels, T_frame]
        vocoder: BigVGAN2 model
        sample_rate: Target sample rate (if None, uses model.h.sampling_rate)
    
    Returns:
        Audio waveform tensor [B, T_audio] with values in [-1, 1]
    """
    if sample_rate is None:
        sample_rate = vocoder.h.sampling_rate
    
    with torch.inference_mode():
        # Generate audio: output shape [B, 1, T_time]
        audio = vocoder(mel)
    
    # Remove channel dimension: [B, 1, T] -> [B, T]
    audio = audio.squeeze(1)
    
    # Resample if needed (if target sample rate differs from model's)
    if sample_rate != vocoder.h.sampling_rate:
        audio = torchaudio.functional.resample(audio, vocoder.h.sampling_rate, sample_rate)
    
    return audio


def extract_mel_from_audio(audio, bigvgan_model, sample_rate=None):
    """
    Extract mel spectrogram from audio waveform using BigVGAN's meldataset.
    
    Based on official usage:
    ```python
    wav = torch.FloatTensor(wav).unsqueeze(0)  # [B, T]
    mel = get_mel_spectrogram(wav, model.h).to(device)  # [B, C_mel, T_frame]
    ```
    
    Args:
        audio: Audio waveform tensor [B, T] or [B, C, T] with values in [-1, 1]
        bigvgan_model: BigVGAN model (to get mel parameters from model.h)
        sample_rate: Sample rate of input audio (if None, assumes model.h.sampling_rate)
    
    Returns:
        Mel spectrogram tensor [B, C_mel, T_frame]
    """
    try:
        # Try importing meldataset - support multiple import methods
        try:
            from meldataset import get_mel_spectrogram
        except ImportError:
            # Try adding BigVGAN directory to path
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            bigvgan_paths = [
                os.path.join(current_dir, 'BigVGAN'),
                os.path.join(current_dir, '../BigVGAN'),
                os.path.join(current_dir, '../../BigVGAN'),
            ]
            for bigvgan_path in bigvgan_paths:
                if os.path.exists(bigvgan_path) and os.path.exists(os.path.join(bigvgan_path, 'meldataset.py')):
                    if bigvgan_path not in sys.path:
                        sys.path.insert(0, bigvgan_path)
                    from meldataset import get_mel_spectrogram
                    print(f"  Loaded meldataset from local path: {bigvgan_path}")
                    break
            else:
                raise ImportError("meldataset not found")
    except ImportError:
        # Fallback to torchaudio if meldataset is not available
        print("Warning: meldataset not found, using torchaudio fallback")
        print("  Make sure BigVGAN is properly installed and meldataset.py is accessible")
        return extract_mel_from_audio_torchaudio(audio, bigvgan_model, sample_rate)
    
    # Ensure audio is [B, T] format (as per official docs)
    if audio.dim() == 3:
        # [B, C, T] -> [B, T] (take first channel or average if multiple channels)
        if audio.shape[1] == 1:
            audio = audio.squeeze(1)
        else:
            # If multiple channels, take the first one
            audio = audio[:, 0, :]
    elif audio.dim() == 2:
        # Already [B, T], good
        pass
    else:
        raise ValueError(f"Audio must be [B, T] or [B, C, T], got shape {audio.shape}")
    
    # Use model's sampling rate if not specified
    target_sr = bigvgan_model.h.sampling_rate
    if sample_rate is None:
        sample_rate = target_sr
    
    # Resample if needed (as per official docs: librosa.load with sr=model.h.sampling_rate)
    if sample_rate != target_sr:
        audio = torchaudio.functional.resample(audio, sample_rate, target_sr)
    
    # Compute mel using BigVGAN's meldataset (ensures compatibility)
    # Official: mel = get_mel_spectrogram(wav, model.h).to(device)
    mel = get_mel_spectrogram(audio, bigvgan_model.h).to(audio.device)
    
    return mel


def extract_mel_from_audio_torchaudio(audio, bigvgan_model, sample_rate=None):
    """
    Fallback: Extract mel spectrogram using torchaudio (if meldataset not available).
    
    Args:
        audio: Audio waveform tensor [B, T] or [B, C, T]
        bigvgan_model: BigVGAN model (to get mel parameters)
        sample_rate: Sample rate
    
    Returns:
        Mel spectrogram tensor [B, n_mels, T_mel]
    """
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)  # Add channel dimension
    
    if sample_rate is None:
        sample_rate = bigvgan_model.h.sampling_rate
    
    # Get mel parameters from model
    n_mels = getattr(bigvgan_model.h, 'n_mel_channels', None) or getattr(bigvgan_model.h, 'n_mels', None) or 128
    hop_length = getattr(bigvgan_model.h, 'hop_size', None) or 256
    win_length = getattr(bigvgan_model.h, 'win_size', None) or 1024
    n_fft = getattr(bigvgan_model.h, 'n_fft', None) or 1024
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=sample_rate // 2
    ).to(audio.device)
    
    mel = mel_transform(audio)
    # Convert to log scale (BigVGAN expects log mel)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    
    return mel

