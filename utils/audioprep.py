import torch
import torchaudio
import torch.nn.functional as F

# Resampling if necessary
def resample_if_needed(signal, orig_sr, target_sr):
    # Ensure sample rates are Python ints (not tensors)
    # Handle Tensor case - only convert if it's a scalar tensor
    if isinstance(orig_sr, torch.Tensor):
        if orig_sr.numel() == 1:
            orig_sr = int(orig_sr.item())
        else:
            # If it's not a scalar, something is wrong - raise error with helpful message
            raise ValueError(
                f"orig_sr should be a scalar (sample rate), but got a Tensor with {orig_sr.numel()} elements. "
                f"This usually means an audio waveform was passed instead of sample rate."
            )
    if isinstance(target_sr, torch.Tensor):
        if target_sr.numel() == 1:
            target_sr = int(target_sr.item())
        else:
            raise ValueError(
                f"target_sr should be a scalar (sample rate), but got a Tensor with {target_sr.numel()} elements."
            )
    
    # Ensure they are ints
    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    
    if orig_sr != target_sr:
        return torchaudio.functional.resample(signal, orig_sr, target_sr)
    return signal
# Squeeze and normalize
def squeeze_and_normalize(signal):
    signal = torch.squeeze(signal)
    return signal * (0.95 / torch.max(signal))
# Pad if necessary
def pad_if_needed(signal, length):
    if signal.shape[0] < length:
        return F.pad(signal, [0, length - signal.shape[0]], "constant")
    return signal

def process_signal(signal, orig_sr, target_sr, target_len, segment_len):
    signal = resample_if_needed(signal, orig_sr, target_sr)
    signal = squeeze_and_normalize(signal)
    signal = signal[:target_len]
    signal = pad_if_needed(signal, segment_len)
    return signal