"""TIMIT data generator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from pathlib import Path
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

def fill_nans(data_matrix):
    """Fills NaN's with nearest neighbours.

    This method is adapted from the method `fill`, which you can find here:
    https://stackoverflow.com/posts/9262129/revisions

    :param data_matrix: numpy array of real-valued data.
    :return: data_matrix: Same but without NaN's.
    """
    
    indices = distance_transform_edt(
        np.isnan(data_matrix), return_distances=False, return_indices=True
    )
    return data_matrix[tuple(indices)]

class WTIMIT(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
    ) -> None:
      self._parse_filesystem(root)
    
    def _parse_filesystem(self, root: str) -> None:
      # 直接查找 normal 目录，而不是 wtimit/normal
      data_dir = Path(os.path.join(root, "normal"))
      if not data_dir.exists():
        # 向后兼容：如果 normal 不存在，尝试 wtimit/normal
        data_dir = Path(os.path.join(root, "wtimit/normal"))
      self._flist = []
      for in_path in data_dir.rglob("*.wav"):
        # if in_path.name.startswith("s10"):
        # s[0-1]
        # if re.match(r"s[0-1]\d{2}u0(0[3-9]|1[0-2])n\.wav$", in_path.name):
        self._flist.append(in_path)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
      input_file = self._flist[n]
      waveform, sample_rate = torchaudio.load(input_file)
      
      # 查找 whisper 文件（whisper 目录，文件名从 n.wav 改为 w.wav）
      fileid_audio = Path(str(input_file).replace('normal', "whisper").replace('n.wav', 'w.wav'))
      if not fileid_audio.exists():
        # 向后兼容：如果 whisper 不存在，尝试 vad-ppw
        fileid_audio = Path(str(input_file).replace('normal', "vad-ppw"))
      waveform_p, sample_rate = torchaudio.load(fileid_audio)
      
      # 查找 VAD 文件（如果不存在，使用 normal 文件）
      fileid_audio = Path(str(input_file).replace('normal', "vad"))
      if not fileid_audio.exists():
        fileid_audio = input_file  # 如果 VAD 不存在，使用 normal 文件
      waveform_v, sample_rate = torchaudio.load(fileid_audio)
      return (
            waveform,
            waveform_p,
            waveform_v,
            sample_rate,
            input_file)
    
    def __len__(self) -> int:
      return len(self._flist)
    
if __name__ == "__main__":
  ds = WTIMIT("YOURPATH")
  print(len(ds))