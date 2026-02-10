import torch
import torch.nn as nn
from typing import List, Tuple

class ReconstructionLoss(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    but uses STFT instead of mel-spectrogram
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = 0
        input = input.to(torch.float32)
        target = target.to(torch.float32)
        for i in range(6, 12):
            s = 2 ** i
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # for 64 bins.
            x = torch.stft(input, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            x = torch.abs(x)
            y = torch.stft(target, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, :, :y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, :, :x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            # 增加 eps 从 1e-5 到 1e-8，避免 log(0) = -inf 在 FP16 中
            loss += alpha * torch.mean(torch.square(torch.log(x + 1e-8) - torch.log(y + 1e-8)))
        return loss / (12 - 6)


class ResNet2d(nn.Module):
    def __init__(
        self,
        n_channels: int,
        factor: int,
        stride: Tuple[int, int]
    ) -> None:
        # https://arxiv.org/pdf/2005.00341.pdf
        # The original paper uses layer normalization, but here
        # we use batch normalization.
        super().__init__()
        self.conv0 = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=(3, 3),
            padding='same')
        self.bn0 = nn.BatchNorm2d(
            n_channels
        )
        self.conv1 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=(stride[0] + 2, stride[1] + 2),
            stride=stride)
        self.bn1 = nn.BatchNorm2d(
            factor * n_channels
        )
        self.conv2 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=1,
            stride=stride)
        self.bn2 = nn.BatchNorm2d(
            factor * n_channels
        )
        self.pad = nn.ReflectionPad2d([
            (stride[1] + 1) // 2,
            (stride[1] + 2) // 2,
            (stride[0] + 1) // 2,
            (stride[0] + 2) // 2,
        ])
        self.activation = nn.LeakyReLU(0.3)

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = self.activation(x)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)

        # shortcut
        y = self.conv2(input)
        y = self.bn2(y)

        x += y
        x = self.activation(x)
        return x


class WaveDiscriminator(nn.Module):
    r"""MelGAN discriminator from https://arxiv.org/pdf/1910.06711.pdf
    """
    def __init__(self, resolution: int = 1, n_channels: int = 4) -> None:
        super().__init__()
        assert resolution >= 1
        if resolution == 1:
            self.avg_pool = nn.Identity()
        else:
            self.avg_pool = nn.AvgPool1d(resolution * 2, stride=resolution)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.layers = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, n_channels, kernel_size=15, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(n_channels, 4 * n_channels, kernel_size=41, stride=4, padding=20, groups=4)),
            nn.utils.weight_norm(nn.Conv1d(4 * n_channels, 16 * n_channels, kernel_size=41, stride=4, padding=20, groups=16)),
            nn.utils.weight_norm(nn.Conv1d(16 * n_channels, 64 * n_channels, kernel_size=41, stride=4, padding=20, groups=64)),
            nn.utils.weight_norm(nn.Conv1d(64 * n_channels, 256 * n_channels, kernel_size=41, stride=4, padding=20, groups=256)),
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 256 * n_channels, kernel_size=5, padding=2)),
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 1, kernel_size=3, padding=1)),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Convert to float32 for numerical stability in mixed precision training
        # Similar to STFTDiscriminator, weight_norm and deep conv layers can be unstable in float16
        x = x.to(torch.float32)
        
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WaveDiscriminator] Input contains NaN/Inf: NaN={torch.isnan(x).sum()}, Inf={torch.isinf(x).sum()}, shape={x.shape}, min={x.min().item():.6f}, max={x.max().item():.6f}")
        
        x = self.avg_pool(x)
        feats = []
        has_nan = False
        nan_layer_idx = None
        
        for i, layer in enumerate(self.layers[:-1]):
            x_before = x.clone()
            x = layer(x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"[WaveDiscriminator] Layer {i} ({type(layer).__name__}) output contains NaN/Inf: NaN={torch.isnan(x).sum()}, Inf={torch.isinf(x).sum()}, shape={x.shape}")
                print(f"[WaveDiscriminator] Layer {i} input stats: min={x_before.min().item():.6f}, max={x_before.max().item():.6f}, mean={x_before.mean().item():.6f}")
                print(f"[WaveDiscriminator] Layer {i} output stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")
                if hasattr(layer, 'weight_g') and layer.weight_g is not None:
                    print(f"[WaveDiscriminator] Layer {i} weight_g stats: min={layer.weight_g.min().item():.6f}, max={layer.weight_g.max().item():.6f}")
                    if torch.isnan(layer.weight_g).any():
                        print(f"[WaveDiscriminator] Layer {i} weight_g contains NaN!")
                # 保存当前层的输出形状（用零填充），然后继续计算后续层以获取正确的形状
                feats.append(torch.zeros_like(x))
                x = torch.zeros_like(x)  # 用零替换 NaN，继续计算后续层
                has_nan = True
                nan_layer_idx = i
                # 不 break，继续计算后续层以获取正确的形状
            else:
                feats.append(x)
            
            x = self.activation(x)
            if torch.isnan(x).any():
                print(f"[WaveDiscriminator] After activation {i} contains NaN: {torch.isnan(x).sum()}")
                # 如果激活后出现 NaN，也用零替换
                x = torch.zeros_like(x)
                if not has_nan:
                    # 如果之前没有 NaN，现在添加一个特征
                    feats.append(torch.zeros_like(x))
                    has_nan = True
                    nan_layer_idx = i
        
        # 执行最后一层
        # 此时 feats 应该已经有 len(self.layers) - 1 个特征（中间层）
        x = self.layers[-1](x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.zeros_like(x)
            if not has_nan:
                print(f"[WaveDiscriminator] Final layer output contains NaN, replaced with zeros")
        feats.append(x)
        
        # 最终检查：确保长度正确
        assert len(feats) == len(self.layers), f"Expected {len(self.layers)} features, got {len(feats)}"
        
        # 最终检查：确保长度正确
        assert len(feats) == len(self.layers), f"Expected {len(self.layers)} features, got {len(feats)}"
        
        return feats
    

class STFTDiscriminator(nn.Module):
    r"""STFT-based discriminator from https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(
        self, n_fft: int = 1024, hop_length: int = 256,
        n_channels: int = 32
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n = n_fft // 2 + 1
        for _ in range(6):
            n = (n - 1) // 2 + 1
        self.layers = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=7, padding='same'),
            nn.LeakyReLU(0.3, inplace=True),
            ResNet2d(n_channels, 2, stride=(2, 1)),
            ResNet2d(2 * n_channels, 2, stride=(2, 2)),
            ResNet2d(4 * n_channels, 1, stride=(2, 1)),
            ResNet2d(4 * n_channels, 2, stride=(2, 2)),
            ResNet2d(8 * n_channels, 1, stride=(2, 1)),
            ResNet2d(8 * n_channels, 2, stride=(2, 2)),
            nn.Conv2d(16 * n_channels, 1, kernel_size=(n, 1))
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[1] == 1
        # input: [batch, channel, sequence]
        x = torch.squeeze(input, 1).to(torch.float32)  # torch.stft() doesn't accept float16
        
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[STFTDiscriminator] Input contains NaN/Inf: NaN={torch.isnan(x).sum()}, Inf={torch.isinf(x).sum()}, shape={x.shape}, min={x.min().item():.6f}, max={x.max().item():.6f}")
        
        x = torch.stft(x, self.n_fft, self.hop_length, normalized=True, onesided=True, return_complex=True)
        
        # 检查STFT输出
        if torch.isnan(x.real).any() or torch.isnan(x.imag).any():
            print(f"[STFTDiscriminator] STFT output contains NaN: real_NaN={torch.isnan(x.real).sum()}, imag_NaN={torch.isnan(x.imag).sum()}, shape={x.shape}")
            print(f"[STFTDiscriminator] STFT stats: real_min={x.real.min().item():.6f}, real_max={x.real.max().item():.6f}, imag_min={x.imag.min().item():.6f}, imag_max={x.imag.max().item():.6f}")
        
        x = torch.abs(x)
        
        # 检查abs后的输出
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[STFTDiscriminator] After abs contains NaN/Inf: NaN={torch.isnan(x).sum()}, Inf={torch.isinf(x).sum()}, shape={x.shape}, min={x.min().item():.6f}, max={x.max().item():.6f}")
        
        x = torch.unsqueeze(x, dim=1)
        
        # 逐层检查
        for i, layer in enumerate(self.layers):
            x_before = x.clone()
            x = layer(x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"[STFTDiscriminator] Layer {i} ({type(layer).__name__}) output contains NaN/Inf: NaN={torch.isnan(x).sum()}, Inf={torch.isinf(x).sum()}, shape={x.shape}")
                print(f"[STFTDiscriminator] Layer {i} input stats: min={x_before.min().item():.6f}, max={x_before.max().item():.6f}, mean={x_before.mean().item():.6f}")
                print(f"[STFTDiscriminator] Layer {i} output stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")
                if hasattr(layer, 'weight'):
                    print(f"[STFTDiscriminator] Layer {i} weight stats: min={layer.weight.min().item():.6f}, max={layer.weight.max().item():.6f}, mean={layer.weight.mean().item():.6f}")
                    if torch.isnan(layer.weight).any():
                        print(f"[STFTDiscriminator] Layer {i} weight contains NaN!")
                break  # 找到第一个产生NaN的层就停止
        
        return x