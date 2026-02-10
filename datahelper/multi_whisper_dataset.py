"""Multi-dataset whisper-normal pair loader from JSON files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from pathlib import Path
from typing import Tuple, Union, List, Dict
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import torch


class MultiWhisperDataset(Dataset):
    """
    Dataset that loads whisper-normal pairs from multiple JSON files.
    
    Supports three datasets:
    1. wtimit: /mnt/workspace/guanzifan/data/wtimit_16k/aligned_pairs_uid.json
    2. aishell6: /mnt/data/share/AISHELL6-Whisper/aligned_pairs_uid_new.json
    3. WHSP_LGU: /mnt/data/share/WHSP_LGU/aligned_pairs_uid_new.json
    """
    
    def __init__(
        self,
        json_paths: Union[str, List[str]],
        training: bool = True,
    ) -> None:
        """
        Args:
            json_paths: Path(s) to JSON file(s) containing dataset pairs
            training: Whether this is for training (True) or validation (False)
        """
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        
        self.training = training
        self.pairs = []
        
        # Load all JSON files
        for json_path in json_paths:
            self._load_json(json_path)
        
        print(f"Loaded {len(self.pairs)} whisper-normal pairs from {len(json_paths)} JSON file(s)")
    
    def _load_json(self, json_path: Union[str, Path]) -> None:
        """Load pairs from a JSON file."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            norm_path = Path(item['norm_path'])
            whsp_path = Path(item['whsp_path'])
            
            # Validate paths exist
            if not norm_path.exists():
                print(f"Warning: Normal audio not found: {norm_path}")
                continue
            if not whsp_path.exists():
                print(f"Warning: Whisper audio not found: {whsp_path}")
                continue
            
            # Store pair info
            pair_info = {
                'norm_path': str(norm_path),
                'whsp_path': str(whsp_path),
                'uid': item.get('uid', ''),
                'dataset': item.get('dataset', 'unknown'),
                'speaker': item.get('speaker', None),
                'duration': item.get('duration', None),
            }
            self.pairs.append(pair_info)
    
    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        """
        Load the n-th sample from the dataset.
        
        Returns:
            Tuple of:
            - normal_waveform: Normal speech waveform [C, T]
            - whisper_waveform: Whisper speech waveform [C, T]  
            - vad_waveform: VAD waveform [C, T]
            - sample_rate: Sample rate
        """
        pair = self.pairs[n]
        norm_path = Path(pair['norm_path'])
        dataset_name = pair.get('dataset', 'unknown')
        
        # Load normal audio
        norm_waveform, sample_rate = torchaudio.load(norm_path)
        
        # Use whsp_path from JSON directly (this is the whisper audio)
        vad_ppw_path = Path(pair['whsp_path'])
        if not vad_ppw_path.exists():
            # Fallback: try to infer from norm_path
            vad_ppw_path = Path(str(norm_path).replace('normal', 'whisper').replace('n.wav', 'w.wav').replace('n.WAV', 'w.WAV'))
        
        # For vad path, try to infer from norm_path, fallback to normal
        vad_path = Path(str(norm_path).replace('normal', 'vad'))
        if not vad_path.exists():
            vad_path = norm_path  # Use normal as fallback
        
        # Load whisper audio (from whsp_path in JSON)
        if not vad_ppw_path.exists():
            raise FileNotFoundError(f"Whisper audio not found: {vad_ppw_path} (from JSON: {pair['whsp_path']})")
        whsp_waveform, whsp_sr = torchaudio.load(vad_ppw_path)
        
        # Load VAD audio
        vad_waveform, vad_sr = torchaudio.load(vad_path)
        
        # Convert sample rates to Python ints (torchaudio may return tensors)
        if isinstance(sample_rate, torch.Tensor):
            sample_rate = int(sample_rate.item())
        if isinstance(whsp_sr, torch.Tensor):
            whsp_sr = int(whsp_sr.item())
        if isinstance(vad_sr, torch.Tensor):
            vad_sr = int(vad_sr.item())
        sample_rate = int(sample_rate)
        whsp_sr = int(whsp_sr)
        vad_sr = int(vad_sr)
        
        # Ensure same sample rate
        if whsp_sr != sample_rate:
            whsp_waveform = torchaudio.functional.resample(whsp_waveform, whsp_sr, sample_rate)
        if vad_sr != sample_rate:
            vad_waveform = torchaudio.functional.resample(vad_waveform, vad_sr, sample_rate)
        
        # Ensure same number of channels
        if norm_waveform.shape[0] != whsp_waveform.shape[0]:
            # Take first channel if mismatch
            if norm_waveform.shape[0] > whsp_waveform.shape[0]:
                norm_waveform = norm_waveform[:1]
            else:
                whsp_waveform = whsp_waveform[:1]
        
        if norm_waveform.shape[0] != vad_waveform.shape[0]:
            # Take first channel if mismatch
            if norm_waveform.shape[0] > vad_waveform.shape[0]:
                norm_waveform = norm_waveform[:1]
            else:
                vad_waveform = vad_waveform[:1]
        
        # Ensure sample_rate is Python int (not tensor)
        sample_rate = int(sample_rate)
        
        return (
            norm_waveform,      # normal
            whsp_waveform,      # pseudo-whisper (whisper/vad-ppw)
            vad_waveform,       # vad
            sample_rate,
        )
    
    def __len__(self) -> int:
        return len(self.pairs)


if __name__ == "__main__":
    # Test the dataset
    json_paths = [
        '/mnt/workspace/guanzifan/data/wtimit_16k/aligned_pairs_uid.json',
        '/mnt/data/share/AISHELL6-Whisper/aligned_pairs_uid_new.json',
        '/mnt/data/share/WHSP_LGU/aligned_pairs_uid_new.json',
    ]
    
    ds = MultiWhisperDataset(json_paths, training=True)
    print(f"Dataset size: {len(ds)}")
    
    # Test loading one sample
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample 0: normal shape={sample[0].shape}, whisper shape={sample[1].shape}, sr={sample[3]}")

