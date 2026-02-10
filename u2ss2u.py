# (c) 2024-2025 Tan Tianyi
# This code is adopted from an unofficial SoundStream implementation in Pytorch.
# The original implementation can be found at https://github.com/kaiidams/soundstream-pytorch.
# We are using it under the MIT license. Thanks to the original author for providing this great work.

from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
try:
    import pytorch_lightning as pl
except ImportError:
    class pl:
        class LightningModule:
            pass
        class Callback:
            pass

from datahelper import TIMIT, WTIMIT, LJSPEECH, LIBRITTS, WHISPER
from datahelper.multi_whisper_dataset import MultiWhisperDataset
from models.s2u import call_feature_by_name, DVAEDecoder
from models.u2s import Reencoder, Decoder
from models.discriminators import WaveDiscriminator, ReconstructionLoss, STFTDiscriminator
from models.loss import t_axis_distill_loss, MultiScaleMelSpectrogramLoss
from utils.config import Config
from utils.audioprep import process_signal
from utils.s2f0 import load_F0_models, wav2F0
from pesq import pesq
import nemo.collections.asr as nemo_asr
     
    
class StreamableModel(pl.LightningModule):
    def __init__(
        self,
        n_channels: int = 16,
        padding: str = "same",
        n_reencoder_layer: int = 1,
        n_encoder_layer: int = 12,
        batch_size: int = 32,
        n_embed_dim: int = 256,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        segment_length: int = 32270,
        lr: float = 1e-6,
        b1: float = 0.5,
        b2: float = 0.9,
        dataset: str = 'ljspeech',
        reen_nn_type: str = 'adapt',
        feature_type: str = 'mfcc',
        trainable: bool = True,
        pseudo_rate: float = 0.4,
        datasets_root: str = 'YOURPATH',
        F0_model_path: str = './libs/JDC/bst.t7',

    ) -> None:
        # https://arxiv.org/pdf/2009.02095.pdf
        # 2. Method
        # SEANet uses Adam with lr=1e-4, beta1=0.5, beta2=0.9
        # batch_size=16
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.spec = call_feature_by_name(feature_type, trainable)
        self.reencoder = Reencoder(n_layers=n_reencoder_layer, wavenet_embed_dim=n_embed_dim, nn_type=reen_nn_type)
        self.encoder = DVAEDecoder(idim=n_embed_dim, odim=n_embed_dim, n_layer=n_encoder_layer)
        self.decoder = Decoder(n_channels=n_channels, padding=padding)
        # self.linear = nn.Linear(256, 512)

        device_id = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        map_location = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large", map_location=map_location)
        self.speaker_model.eval()

        self.wave_discriminators = nn.ModuleList([
            WaveDiscriminator(resolution=1),
            WaveDiscriminator(resolution=2),
            WaveDiscriminator(resolution=4)
        ])
        self.rec_loss = ReconstructionLoss()
        self.stft_discriminator = STFTDiscriminator()

        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        self.hubert_loss = t_axis_distill_loss()
        self.energy_loss = MultiScaleMelSpectrogramLoss(sampling_rate=sample_rate)

        self.hubert_soft = torch.hub.load("bshall/hubert:main", f"hubert_soft").to(torch.cuda.current_device())
        self.pitch_extractor = load_F0_models(F0_model_path, device="cuda:{}".format(torch.cuda.current_device()))
        self.segment_length = segment_length
        self.datasets_root = datasets_root
        self.pseudo_rate = pseudo_rate

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_g = torch.optim.Adam(
            chain(
                self.spec.parameters(),
                self.encoder.parameters(),
                self.reencoder.parameters(),
                self.decoder.parameters(),
            ),
            lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.Adam(
            chain(
                self.wave_discriminators.parameters(),
                self.stft_discriminator.parameters()
            ),
            lr=lr, betas=(b1, b2))
        # scheduler_d = torch.optim.lr_scheduler.StepLR(
        #     optimizer_d, step_size=2, gamma=0.95
        # )
        return [optimizer_g, optimizer_d], []

    def forward(self, input, spkemb):
        spectrogram = self.spec(input)
        spectrogram = spectrogram.transpose(-1, -2)
        x = self.encoder(spectrogram)
        hubert_like = torch.nn.functional.pad(x, (0, 0, 0, 1, 0, 0))
        x = torch.transpose(hubert_like, -1, -2)
        # hubert = self.hubert_soft.units(input)
        # hubert = torch.nn.functional.pad(hubert, (0, 0, 0, 2, 0, 0))
        # x = torch.transpose(hubert, -1, -2)
        x = self.reencoder(x, spkemb)
        x = self.decoder(x)
        return x, hubert_like

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        # sch = self.lr_schedulers() 
        inputs = batch[:, None, :] # 1:normal 2:ppw 3:vad
        input = inputs[:, :,self.segment_length*2:self.segment_length*3]
        input_0 = inputs[:, :,:32270]
        # if random.random() < self.pseudo_rate:
        #     input_0 = inputs[:, :, self.segment_length*1:self.segment_length*2] # normal
        # else:
        #     input_0 = inputs[:, :, :self.segment_length]  # ppw
        spkemb = torch.cat([self.speaker_model.infer_segment(w16)[0] for w16 in input.squeeze().squeeze().cpu()], dim=0)

        # train generator
        self.toggle_optimizer(optimizer_g)
        output, hubert_like = self.forward(input_0, spkemb)

        # F0 Loss
        to_mel = self.to_mel.to(input.device)
        mels = to_mel(input).squeeze()
        pred_mels = to_mel(output).squeeze()
        mel_mean, mel_std = -4, 4
        mels = (torch.log(1e-5 + mels) - mel_mean) / mel_std
        pred_mels = (torch.log(1e-5 + pred_mels) - mel_mean) / mel_std
        F0_real = wav2F0(mels, self.pitch_extractor, input.device, norm=False)
        F0_pred = wav2F0(pred_mels, self.pitch_extractor, input.device, norm=False)
        f0_loss = F.smooth_l1_loss(F0_real, F0_pred)
        self.log("f0_loss", f0_loss, prog_bar=False)
        # Energy Loss
        energy_loss = self.energy_loss(input, output)
        self.log("energy_loss", energy_loss, prog_bar=False)
        # Content Loss 
        pred_hubert = self.hubert_soft.units(output.to(torch.cuda.current_device()))
        pred_hubert = pred_hubert.to(input.device)
        content_loss = self.hubert_loss(pred_hubert, hubert_like)
        self.log("content_loss", content_loss, prog_bar=False)
        # Speaker Embedding Loss
        pred_spkemb = torch.cat([self.speaker_model.infer_segment(w16)[0] for w16 in output.squeeze().squeeze().cpu()], dim=0)
        spk_loss = self.hubert_loss(pred_spkemb, spkemb)
        self.log("spk_loss", spk_loss, prog_bar=False)

        stft_out = self.stft_discriminator(output)
        g_stft_loss = torch.mean(torch.relu(1 - stft_out))
        self.log("g_stft_loss", g_stft_loss)

        g_wave_loss = 0
        g_feat_loss = 0
        for i in range(3):
            feats1 = self.wave_discriminators[i](input)
            feats2 = self.wave_discriminators[i](output)
            assert len(feats1) == len(feats2)
            g_wave_loss += torch.mean(torch.relu(1 - feats2[-1]))
            g_feat_loss += sum(torch.mean(
                torch.abs(f1 - f2))
                for f1, f2 in zip(feats1[:-1], feats2[:-1])) / (len(feats1) - 1)
        self.log("g_wave_loss", g_wave_loss / 3)
        self.log("g_feat_loss", g_feat_loss / 3)

        g_rec_loss = self.rec_loss(output[:, 0, :], input[:, 0, :])
        self.log("g_rec_loss", g_rec_loss, prog_bar=True)

        g_feat_loss = g_feat_loss / 3
        g_adv_loss = (g_stft_loss + g_wave_loss) / 4
        g_loss = g_adv_loss  + g_rec_loss +  100 * g_feat_loss  + 0.5 * f0_loss  + 0.5 * energy_loss  + spk_loss + content_loss

        self.log("g_loss", g_loss, prog_bar=True)

        self.manual_backward(g_loss)
        torch.nn.utils.clip_grad_norm_(self.spec.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.reencoder.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=0.5)
        
        # 检测并置零 NaN/Inf 梯度，但必须执行 step() 以让 Lightning 跟踪 global_step
        has_nan_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 只检查生成器相关的参数
                if 'encoder' in name or 'reencoder' in name or 'decoder' in name:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        param.grad.zero_()  # 置零 NaN/Inf 梯度
                        print(f"[Generator] ⚠ Gradient contains NaN/Inf: {name}, zeroing it")
        
        # 打印生成器梯度
        log_every_n_steps = getattr(self.hparams, 'log_every_n_steps', 10)
        if batch_idx % log_every_n_steps == 0:
            print(f"\n[Generator Gradients] Step {self.global_step}, Batch {batch_idx}")
            total_norm = 0.0
            param_count = 0
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # 只打印生成器相关的参数
                    if 'encoder' in name or 'reencoder' in name or 'decoder' in name:
                        grad_norm = param.grad.norm().item()
                        grad_mean = param.grad.mean().item()
                        grad_std = param.grad.std().item()
                        has_nan = torch.isnan(param.grad).any().item()
                        has_inf = torch.isinf(param.grad).any().item()
                        total_norm += grad_norm ** 2
                        param_count += 1
                        if has_nan or has_inf or grad_norm > 100:
                            print(f"  {name}: norm={grad_norm:.6f}, mean={grad_mean:.6f}, std={grad_std:.6f}, has_nan={has_nan}, has_inf={has_inf}")
            total_norm = total_norm ** 0.5
            print(f"  [Generator] Total gradient norm: {total_norm:.6f} (from {param_count} parameters)")
        
        # 即使有 NaN 梯度（已置零），也必须执行 step()，否则 Lightning 不会增加 global_step
        if has_nan_grad:
            print(f"[Generator] ⚠ NaN gradients detected and zeroed, but still executing optimizer.step() for Lightning tracking")
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        # sch.step()

        # train discriminator
        output, hubert_like = self.forward(input, spkemb)

        stft_out = self.stft_discriminator(input)
        d_stft_loss = torch.mean(torch.relu(1 - stft_out))
        stft_out = self.stft_discriminator(output)
        d_stft_loss += torch.mean(torch.relu(1 + stft_out))

        d_wave_loss = 0
        for i in range(3):
            feats = self.wave_discriminators[i](input)
            d_wave_loss += torch.mean(torch.relu(1 - feats[-1]))
            feats = self.wave_discriminators[i](output)
            d_wave_loss += torch.mean(torch.relu(1 + feats[-1]))

        d_loss = (d_stft_loss + d_wave_loss) / 4

        self.log("d_stft_loss", d_stft_loss)
        self.log("d_wave_loss", d_wave_loss / 3)

        d_loss = (d_stft_loss + d_wave_loss) / 4
        self.log("d_loss", d_loss, prog_bar=True)

        self.manual_backward(d_loss)
        
        # 检测并置零 NaN/Inf 梯度，但必须执行 step() 以让 Lightning 跟踪 global_step
        has_nan_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 只检查判别器相关的参数
                if 'wave_discriminators' in name or 'stft_discriminator' in name:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        param.grad.zero_()  # 置零 NaN/Inf 梯度
                        print(f"[Discriminator] ⚠ Gradient contains NaN/Inf: {name}, zeroing it")
        
        # 打印判别器梯度
        log_every_n_steps = getattr(self.hparams, 'log_every_n_steps', 10)
        if batch_idx % log_every_n_steps == 0:
            print(f"\n[Discriminator Gradients] Step {self.global_step}, Batch {batch_idx}")
            total_norm = 0.0
            param_count = 0
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # 只打印判别器相关的参数
                    if 'wave_discriminators' in name or 'stft_discriminator' in name:
                        grad_norm = param.grad.norm().item()
                        grad_mean = param.grad.mean().item()
                        grad_std = param.grad.std().item()
                        has_nan = torch.isnan(param.grad).any().item()
                        has_inf = torch.isinf(param.grad).any().item()
                        total_norm += grad_norm ** 2
                        param_count += 1
                        if has_nan or has_inf or grad_norm > 100:
                            print(f"  {name}: norm={grad_norm:.6f}, mean={grad_mean:.6f}, std={grad_std:.6f}, has_nan={has_nan}, has_inf={has_inf}")
            total_norm = total_norm ** 0.5
            print(f"  [Discriminator] Total gradient norm: {total_norm:.6f} (from {param_count} parameters)")
        
        # 即使有 NaN 梯度（已置零），也必须执行 step()，否则 Lightning 不会增加 global_step
        if has_nan_grad:
            print(f"[Discriminator] ⚠ NaN gradients detected and zeroed, but still executing optimizer.step() for Lightning tracking")
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        import numpy as np
        inputs = batch[:, None, :]
        input = inputs[:, :, :32270] # normal / ppw
        whisper = inputs[:, :,32270:32270*2] # ppw
        spkemb = torch.cat([self.speaker_model.infer_segment(w16)[0] for w16 in input.squeeze().squeeze().cpu()], dim=0)
        with torch.no_grad():
            output, _ = self.forward(whisper, spkemb)
            val_pesq_tot = 0
            val_pesq_count = 0
            MAX_WAV_VALUE = 32767.0
            
            # 检查整个 output 张量是否有 NaN/Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"[Validation] ⚠ Warning: output contains NaN/Inf, skipping PESQ calculation")
                self.log("val_pesq", 0.0, on_epoch=True, prog_bar=True, sync_dist=True)
                return
            
            for y_16k, y_g_hat_16k in zip(input, output):
                # 检查单个样本是否有 NaN/Inf
                if torch.isnan(y_16k).any() or torch.isinf(y_16k).any():
                    print(f"[Validation] ⚠ Warning: y_16k contains NaN/Inf, skipping this sample")
                    continue
                if torch.isnan(y_g_hat_16k).any() or torch.isinf(y_g_hat_16k).any():
                    print(f"[Validation] ⚠ Warning: y_g_hat_16k contains NaN/Inf, skipping this sample")
                    continue
                
                # 转换为 numpy 并清理
                y_np = y_16k[0].cpu().numpy()
                y_g_hat_np = y_g_hat_16k[0].cpu().numpy()
                
                # 检查并清理 NaN/Inf
                if np.isnan(y_np).any() or np.isinf(y_np).any():
                    y_np = np.nan_to_num(y_np, nan=0.0, posinf=1.0, neginf=-1.0)
                if np.isnan(y_g_hat_np).any() or np.isinf(y_g_hat_np).any():
                    y_g_hat_np = np.nan_to_num(y_g_hat_np, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 限制值范围到 [-1, 1]
                y_np = np.clip(y_np, -1.0, 1.0)
                y_g_hat_np = np.clip(y_g_hat_np, -1.0, 1.0)
                
                # 确保长度一致
                min_len = min(len(y_np), len(y_g_hat_np))
                if min_len < 160:  # PESQ 需要至少一定长度的音频
                    print(f"[Validation] ⚠ Warning: Audio too short ({min_len} samples), skipping PESQ")
                    continue
                y_np = y_np[:min_len]
                y_g_hat_np = y_g_hat_np[:min_len]
                
                # 转换为整数
                y_int_16k = np.clip(y_np * MAX_WAV_VALUE, -32768, 32767).astype(np.int16)
                y_g_hat_int_16k = np.clip(y_g_hat_np * MAX_WAV_VALUE, -32768, 32767).astype(np.int16)
                
                # 计算 PESQ（带异常处理）
                try:
                    pesq_score = pesq(16000, y_int_16k, y_g_hat_int_16k, "wb")
                    if not (np.isnan(pesq_score) or np.isinf(pesq_score)):
                        val_pesq_tot += pesq_score
                        val_pesq_count += 1
                    else:
                        print(f"[Validation] ⚠ Warning: PESQ returned NaN/Inf, skipping")
                except Exception as e:
                    print(f"[Validation] ⚠ Warning: PESQ calculation failed: {e}, skipping")
                    continue
            
            # 计算平均 PESQ
            if val_pesq_count > 0:
                avg_pesq = val_pesq_tot / val_pesq_count
            else:
                avg_pesq = 0.0
                print(f"[Validation] ⚠ Warning: No valid PESQ scores calculated")
            
        self.log("val_pesq", avg_pesq, on_epoch=True, prog_bar=True, sync_dist=True)

    def train_dataloader(self):
        return self._make_dataloader(True)
    
    def val_dataloader(self):
        return self._make_dataloader_val()

    def _make_dataloader(self, train: bool):
        
        class VoiceDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, sample_rate, segment_length):
                self._dataset = dataset
                self._sample_rate = sample_rate
                self._segment_length = segment_length

            def __getitem__(self, index):
                import random
                data = self._dataset[index]
                # MultiWhisperDataset returns: (norm_waveform, whsp_waveform, vad_waveform, sample_rate) - 4 elements
                # Old WTIMIT returns: (waveform, waveform_p, waveform_v, sample_rate, input_file) - 5 elements
                if len(data) == 4:
                    # MultiWhisperDataset format
                    x, x_p, x_v, sample_rate = data
                elif len(data) == 5:
                    # Old WTIMIT format
                    x, x_p, x_v, sample_rate, _ = data
                else:
                    raise ValueError(f"Unexpected data format: expected 4 or 5 elements, got {len(data)}")
                
                # Ensure sample_rate is int
                if isinstance(sample_rate, torch.Tensor):
                    sample_rate = int(sample_rate.item())
                sample_rate = int(sample_rate)
                
                target_len = min(x.shape[-1], x_p.shape[-1], x_v.shape[-1])
                x = process_signal(x, sample_rate, self._sample_rate, target_len, self._segment_length)
                x_p = process_signal(x_p, sample_rate, self._sample_rate, target_len, self._segment_length)
                x_v = process_signal(x_v, sample_rate, self._sample_rate, target_len, self._segment_length)
                pos = random.randint(0, x.shape[0] - self._segment_length)
                x = x[pos:pos + self._segment_length]
                x_p = x_p[pos:pos + self._segment_length]
                x_v = x_v[pos:pos + self._segment_length]
                output = torch.cat((x, x_p, x_v), 0)
                return output

            def __len__(self):
                return len(self._dataset)
            
        def collate(examples):
            return torch.stack(examples)

        if self.hparams.dataset == 'wtimit':
            # 使用三个数据集：wtimit, aishell6, WHSP_LGU
            from pathlib import Path
            json_paths = [
                '/mnt/workspace/guanzifan/data/wtimit_16k/aligned_pairs_uid.json',
                '/mnt/data/share/AISHELL6-Whisper/aligned_pairs_uid_new.json',
                '/mnt/data/share/WHSP_LGU/aligned_pairs_uid_new.json',
            ]
            print(f"[DataLoader] Loading multi-whisper datasets from JSON files:")
            for json_path in json_paths:
                print(f"   - {json_path}")
            ds = MultiWhisperDataset(json_paths, training=True)
            print(f"[DataLoader] Multi-whisper dataset loaded: {len(ds)} samples")
            if len(ds) == 0:
                print(f"[DataLoader] Checking JSON files:")
                for json_path in json_paths:
                    json_file = Path(json_path)
                    print(f"   - {json_file}: exists={json_file.exists()}")
                    if json_file.exists():
                        import json
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        print(f"     Contains {len(data)} entries")
                raise ValueError(f"Multi-whisper dataset is empty! Check JSON paths: {json_paths}")
        elif self.hparams.dataset == 'ljspeech':
            ds = LJSPEECH(self.datasets_root)
        elif self.hparams.dataset == 'libritts':
            ds = LIBRITTS(self.datasets_root)
        elif self.hparams.dataset == 'timit':
            ds = TIMIT(self.datasets_root, training=True)
        else:
            raise ValueError(f"Unknown dataset: {self.hparams.dataset}")
        
        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)
        print(f"[DataLoader] VoiceDataset created: {len(ds)} samples")
        if len(ds) == 0:
            raise ValueError(f"VoiceDataset is empty after processing!")

        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams['batch_size'], shuffle=True,
            collate_fn=collate, num_workers=7)
        return loader

    def _make_dataloader_val(self, sub_rate: float=0.1):

        class VoiceDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, sample_rate, segment_length):
                self._dataset = dataset
                self._sample_rate = sample_rate
                self._segment_length = segment_length

            def __getitem__(self, index):
                import random
                data = self._dataset[index]
                # MultiWhisperDataset returns: (norm_waveform, whsp_waveform, vad_waveform, sample_rate) - 4 elements
                # Old WHISPER returns: (waveform, waveform_p, sample_rate) - 3 elements
                if len(data) == 4:
                    # MultiWhisperDataset format
                    x, x_p, x_v, sample_rate = data
                elif len(data) == 3:
                    # Old WHISPER format
                    x, x_p, sample_rate = data
                else:
                    raise ValueError(f"Unexpected data format: expected 3 or 4 elements, got {len(data)}")
                
                # Ensure sample_rate is int
                if isinstance(sample_rate, torch.Tensor):
                    sample_rate = int(sample_rate.item())
                sample_rate = int(sample_rate)
                
                target_len = min(x.shape[-1], x_p.shape[-1])
                x = process_signal(x, sample_rate, self._sample_rate, target_len, self._segment_length)
                x_p = process_signal(x_p, sample_rate, self._sample_rate, target_len, self._segment_length)
                pos = random.randint(0, x.shape[0] - self._segment_length)
                x = x[pos:pos + self._segment_length]
                x_p = x_p[pos:pos + self._segment_length]
                output = torch.cat((x, x_p), 0)
                return output

            def __len__(self):
                return len(self._dataset)

        def collate(examples):
            return torch.stack(examples)

        # 验证集也使用三个数据集
        if self.hparams.dataset == 'wtimit':
            from pathlib import Path
            json_paths = [
                '/mnt/workspace/guanzifan/data/wtimit_16k/aligned_pairs_uid.json',
                '/mnt/data/share/AISHELL6-Whisper/aligned_pairs_uid_new.json',
                '/mnt/data/share/WHSP_LGU/aligned_pairs_uid_new.json',
            ]
            print(f"[ValDataLoader] Loading multi-whisper datasets from JSON files:")
            for json_path in json_paths:
                print(f"   - {json_path}")
            ds = MultiWhisperDataset(json_paths, training=False)
            print(f"[ValDataLoader] Multi-whisper dataset loaded: {len(ds)} samples")
        else:
            ds = WHISPER(self.datasets_root)
        
        subset_len = int(sub_rate * len(ds))
        ds = torch.utils.data.Subset(ds, range(subset_len))

        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams['batch_size'], shuffle=False,
            collate_fn=collate, num_workers=7)
        return loader


def train():
    config = {
        # Configuration ID
        'id': "experiments",
        'name': "s2uu2s",
        'version': "s2uu2s-libri-ti",
        # Training configuration
        'batch_size': 32, #s2u 128, s2uu2s 32
        'save_checkpoint_dir': "",
        'restore_checkpoint_path': "/mnt/workspace/guanzifan/distillw2n/experiments/s2uu2s/epoch.440-step.409942.ckpt",
        'resume_training': True,
        'training_epochs': 10000,
        'log_every_n_steps': 2,
        'dataset': 'wtimit',
        'datasets_root': '/mnt/workspace/guanzifan/data/wtimit_16k',
        'feature_type': 'mfcc',
        'reen_nn_type': 'adapt',
    }
    config = Config(config)
    model = StreamableModel(
        n_channels=config.n_channels,
        n_embed_dim=config.n_embed_dim,
        n_encoder_layer=config.n_encoder_layer,
        padding=config.padding,
        batch_size=config.batch_size,
        sample_rate=config.sample_rate,
        segment_length=config.segment_length,
        lr=config.lr,
        b1=config.b1,
        b2=config.b2,
        dataset=config.dataset,
        feature_type=config.feature_type,
        reen_nn_type=config.reen_nn_type,
        pseudo_rate=config.pseudo_rate,
        datasets_root=config.datasets_root,
        F0_model_path=config.F0_model_path)
    
    # 手动加载 generator 权重（不加载 discriminator 和 optimizer 状态）
    if config.resume_training and config.restore_checkpoint_path:
        print(f"Loading generator weights from checkpoint: {config.restore_checkpoint_path}")
        print("=" * 80)
        print("🔒 LOADING POLICY:")
        print("   - ✅ Loading: encoder, reencoder, decoder (generator weights only)")
        print("   - ❌ Skipping: ALL discriminator weights (will use random init)")
        print("   - ❌ Skipping: spec weights (spec is frozen)")
        print("   - ❌ Skipping: ALL optimizer states (will use fresh optimizers)")
        print("=" * 80)
        try:
            checkpoint = torch.load(config.restore_checkpoint_path, map_location='cpu', weights_only=True)
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # 定义生成器模块的前缀（只加载这些模块，包括spec）
            generator_prefixes = ['encoder.', 'reencoder.', 'decoder.', 'spec.']
            
            # 定义需要跳过的模块
            skip_prefixes = [
                'wave_discriminators.',  # 判别器：随机初始化
                'stft_discriminator.',   # 判别器：随机初始化
                'discriminator.',
                'speaker_model.',        # 预训练模型
                'to_mel.',              # torchaudio transform
                'hubert_soft.',         # 预训练模型
                'pitch_extractor.',     # 预训练模型
            ]
            
            # 跳过所有 Buffer
            skip_suffixes = ['.running_mean', '.running_var', '.num_batches_tracked', '.buffer']
            
            # 获取当前模型的所有参数名称（包括冻结的spec参数）
            model_all_params = {name: param for name, param in model.named_parameters()}
            generator_param_names = {name for name in model_all_params.keys() 
                                    if any(name.startswith(prefix) for prefix in generator_prefixes)}
            
            # 精准过滤：只加载生成器的可训练参数
            generator_state_dict = {}
            skipped_keys = []
            nan_keys = []
            
            for key, value in state_dict.items():
                # 跳过判别器的所有参数
                if any(key.startswith(skip_prefix) for skip_prefix in skip_prefixes):
                    skipped_keys.append(key)
                    continue
                
                # 跳过所有 Buffer
                if any(key.endswith(suffix) for suffix in skip_suffixes):
                    skipped_keys.append(key)
                    continue
                
                # 只加载生成器的可训练参数
                is_generator_param = any(key.startswith(prefix) for prefix in generator_prefixes)
                if not is_generator_param:
                    skipped_keys.append(key)
                    continue
                
                # 检查是否是模型中的参数（包括冻结的spec参数）
                if key not in generator_param_names:
                    skipped_keys.append(key)
                    continue
                
                # 检查是否包含 NaN/Inf
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        nan_keys.append(key)
                        print(f"  ⚠ Skipping {key}: NaN/Inf detected")
                        continue
                
                # 验证形状匹配
                if key in model_all_params:
                    model_param = model_all_params[key]
                    if value.shape != model_param.shape:
                        print(f"  ⚠ Shape mismatch for {key}: checkpoint {value.shape} vs model {model_param.shape}, skipping")
                        skipped_keys.append(key)
                        continue
                
                # 通过所有检查，添加到加载列表
                generator_state_dict[key] = value
            
            print(f"\n📦 Loading Summary:")
            print(f"   ✅ Generator weights to load: {len(generator_state_dict)}")
            print(f"   ❌ Skipped (discriminators): {sum(1 for k in skipped_keys if any(k.startswith(p) for p in ['wave_discriminators.', 'stft_discriminator.']))}")
            print(f"   ✅ Loading (spec): {sum(1 for k in generator_state_dict.keys() if k.startswith('spec.'))}")
            print(f"   ❌ Skipped (buffers): {sum(1 for k in skipped_keys if any(k.endswith(s) for s in skip_suffixes))}")
            print(f"   ⚠ Skipped (NaN/Inf): {len(nan_keys)}")
            
            # 加载生成器权重
            print(f"\n🔄 Loading generator weights...")
            missing_keys, unexpected_keys = model.load_state_dict(generator_state_dict, strict=False)
            
            if missing_keys:
                print(f"   ⚠ Missing keys (will use random init): {len(missing_keys)}")
                if len(missing_keys) <= 10:
                    for key in missing_keys:
                        print(f"      - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"      - {key}")
                    print(f"      ... and {len(missing_keys) - 5} more")
            
            print(f"   ✅ Successfully loaded {len(generator_state_dict)} generator weights")
            print("=" * 80)
            
        except Exception as e:
            print(f"⚠ Warning: Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with randomly initialized weights...")
    
    pl.seed_everything(config.seed, workers=True)
    trainer = pl.Trainer(
        max_epochs=config.training_epochs,
        log_every_n_steps=config.log_every_n_steps,
        precision='16-mixed',
        logger=pl.loggers.TensorBoardLogger(config.save_checkpoint_dir+config.id, name=config.name, version=config.version),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_last=True,
                every_n_train_steps=1000,  # 每1000步保存一次checkpoint
                save_on_train_epoch_end=False,  # 不在epoch结束时保存（只按步数保存）
                # monitor='val_pesq',
                # save_top_k=2,
                # mode='max'
            )
        ],
        strategy='ddp_find_unused_parameters_true'
    )
    # 不传入 ckpt_path，避免加载 optimizer 状态（已在上面手动加载模型权重）
    trainer.fit(
        model,
        ckpt_path=None  # 不加载 optimizer 状态，只使用上面手动加载的模型权重
    )

    return model


if __name__ == "__main__":
    train()

# # (c) 2024-2025 Tan Tianyi
# # This code is adopted from an unofficial SoundStream implementation in Pytorch.
# # The original implementation can be found at https://github.com/kaiidams/soundstream-pytorch.
# # We are using it under the MIT license. Thanks to the original author for providing this great work.

# from itertools import chain

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchaudio
# try:
#     import pytorch_lightning as pl
# except ImportError:
#     class pl:
#         class LightningModule:
#             pass
#         class Callback:
#             pass

# from datahelper import TIMIT, WTIMIT, LJSPEECH, LIBRITTS, WHISPER
# from datahelper.multi_whisper_dataset import MultiWhisperDataset
# from models.s2u import call_feature_by_name, DVAEDecoder
# from models.u2s import Reencoder, Decoder
# from models.discriminators import WaveDiscriminator, ReconstructionLoss, STFTDiscriminator
# from models.loss import t_axis_distill_loss, MultiScaleMelSpectrogramLoss
# from utils.config import Config
# from utils.audioprep import process_signal
# from utils.s2f0 import load_F0_models, wav2F0
# from pesq import pesq
# import nemo.collections.asr as nemo_asr
     
    
# class StreamableModel(pl.LightningModule):
#     def __init__(
#         self,
#         n_channels: int = 16,
#         padding: str = "same",
#         n_reencoder_layer: int = 1,
#         n_encoder_layer: int = 12,
#         batch_size: int = 32,
#         n_embed_dim: int = 256,
#         sample_rate: int = 16_000,
#         n_mels: int = 80,
#         n_fft: int = 1024,
#         win_length: int = 1024,
#         hop_length: int = 320,
#         segment_length: int = 32270,
#         lr: float = 1e-6,
#         b1: float = 0.5,
#         b2: float = 0.9,
#         dataset: str = 'ljspeech',
#         reen_nn_type: str = 'adapt',
#         feature_type: str = 'mfcc',
#         trainable: bool = True,
#         pseudo_rate: float = 0.4,
#         datasets_root: str = 'YOURPATH',
#         F0_model_path: str = './libs/JDC/bst.t7',

#     ) -> None:
#         # https://arxiv.org/pdf/2009.02095.pdf
#         # 2. Method
#         # SEANet uses Adam with lr=1e-4, beta1=0.5, beta2=0.9
#         # batch_size=16
#         super().__init__()
#         self.save_hyperparameters()
#         self.automatic_optimization = False

#         self.spec = call_feature_by_name(feature_type, trainable)
#         # 1. 冻结 spec 模块的参数，避免训练时产生 NaN 梯度
#         for p in self.spec.parameters():
#             p.requires_grad = False
#         self.reencoder = Reencoder(n_layers=n_reencoder_layer, wavenet_embed_dim=n_embed_dim, nn_type=reen_nn_type)
#         self.encoder = DVAEDecoder(idim=n_embed_dim, odim=n_embed_dim, n_layer=n_encoder_layer)
#         self.decoder = Decoder(n_channels=n_channels, padding=padding)
#         # self.linear = nn.Linear(256, 512)

#         device_id = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
#         map_location = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
#         self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large", map_location=map_location)
#         self.speaker_model.eval()

#         self.wave_discriminators = nn.ModuleList([
#             WaveDiscriminator(resolution=1),
#             WaveDiscriminator(resolution=2),
#             WaveDiscriminator(resolution=4)
#         ])
#         self.rec_loss = ReconstructionLoss()
#         self.stft_discriminator = STFTDiscriminator()
        
#         # 检查判别器权重初始化是否正常
#         def check_weights(m, name=""):
#             has_nan = False
#             for param_name, param in m.named_parameters():
#                 if torch.isnan(param).any() or torch.isinf(param).any():
#                     has_nan = True
#                     nan_count = torch.isnan(param).sum().item()
#                     inf_count = torch.isinf(param).sum().item()
#                     total = param.numel()
#                     print(f"⚠ Warning: {name}.{param_name} contains NaN/Inf after initialization!")
#                     print(f"   NaN: {nan_count}/{total}, Inf: {inf_count}/{total}, shape={param.shape}")
#                     if not torch.isnan(param).all():
#                         print(f"   Stats: min={param.min().item():.6f}, max={param.max().item():.6f}, mean={param.mean().item():.6f}")
#             return has_nan
        
#         print("Checking discriminator weights after initialization...")
#         wave_disc_nan = False
#         for i, disc in enumerate(self.wave_discriminators):
#             if check_weights(disc, f"wave_discriminators[{i}]"):
#                 wave_disc_nan = True
#         stft_disc_nan = check_weights(self.stft_discriminator, "stft_discriminator")
        
#         if wave_disc_nan or stft_disc_nan:
#             print("⚠ CRITICAL: Discriminators contain NaN/Inf right after initialization!")
#             print("  This suggests the checkpoint file itself contains corrupted weights.")
#             print("  The discriminators will be re-initialized during checkpoint loading.")

#         self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
#         self.hubert_loss = t_axis_distill_loss()
#         self.energy_loss = MultiScaleMelSpectrogramLoss(sampling_rate=sample_rate)

#         self.hubert_soft = torch.hub.load("bshall/hubert:main", f"hubert_soft").to(torch.cuda.current_device())
#         self.pitch_extractor = load_F0_models(F0_model_path, device="cuda:{}".format(torch.cuda.current_device()))
#         self.segment_length = segment_length
#         self.datasets_root = datasets_root
#         self.pseudo_rate = pseudo_rate

#     def configure_optimizers(self):
#         lr = self.hparams.lr
#         b1 = self.hparams.b1
#         b2 = self.hparams.b2

#         # 3. 确保 optimizer 不包含 spec（spec 已冻结）
#         optimizer_g = torch.optim.Adam(
#             chain(
#                 self.encoder.parameters(),
#                 self.reencoder.parameters(),
#                 self.decoder.parameters(),
#             ),
#             lr=lr, betas=(b1, b2))
#         optimizer_d = torch.optim.Adam(
#             chain(
#                 self.wave_discriminators.parameters(),
#                 self.stft_discriminator.parameters()
#             ),
#             lr=lr, betas=(b1, b2))
#         # scheduler_d = torch.optim.lr_scheduler.StepLR(
#         #     optimizer_d, step_size=2, gamma=0.95
#         # )
#         return [optimizer_g, optimizer_d], []

#     def forward(self, input, spkemb):
#         # 2. spec 使用 FP32，避免 AMP 导致的 NaN
#         with torch.cuda.amp.autocast(enabled=False):
#             spectrogram = self.spec(input.float())
#         spectrogram = spectrogram.transpose(-1, -2)
#         x = self.encoder(spectrogram)
#         hubert_like = torch.nn.functional.pad(x, (0, 0, 0, 1, 0, 0))
#         x = torch.transpose(hubert_like, -1, -2)
#         # hubert = self.hubert_soft.units(input)
#         # hubert = torch.nn.functional.pad(hubert, (0, 0, 0, 2, 0, 0))
#         # x = torch.transpose(hubert, -1, -2)
#         x = self.reencoder(x, spkemb)
#         x = self.decoder(x)
#         return x, hubert_like

#     def training_step(self, batch, batch_idx):
#         optimizer_g, optimizer_d = self.optimizers()
#         # sch = self.lr_schedulers() 
#         inputs = batch[:, None, :] # 1:normal 2:ppw 3:vad
#         input = inputs[:, :,self.segment_length*2:self.segment_length*3]
#         input_0 = inputs[:, :,:32270]
#         # if random.random() < self.pseudo_rate:
#         #     input_0 = inputs[:, :, self.segment_length*1:self.segment_length*2] # normal
#         # else:
#         #     input_0 = inputs[:, :, :self.segment_length]  # ppw
#         spkemb = torch.cat([self.speaker_model.infer_segment(w16)[0] for w16 in input.squeeze().squeeze().cpu()], dim=0)

#         # train generator
#         self.toggle_optimizer(optimizer_g)
#         output, hubert_like = self.forward(input_0, spkemb)
        
#         # 检查输出和输入是否有 NaN/Inf（仅用于诊断，不清理）
#         if torch.isnan(output).any() or torch.isinf(output).any():
#             print(f"[Generator] Output contains NaN/Inf: NaN={torch.isnan(output).sum()}, Inf={torch.isinf(output).sum()}")
#             print(f"[Generator] Output shape: {output.shape}, min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}")
        
#         if torch.isnan(input).any() or torch.isinf(input).any():
#             print(f"[Generator] Input contains NaN/Inf: NaN={torch.isnan(input).sum()}, Inf={torch.isinf(input).sum()}")
#             print(f"[Generator] Input shape: {input.shape}, min={input.min().item():.6f}, max={input.max().item():.6f}, mean={input.mean().item():.6f}")

#         # F0 Loss
#         to_mel = self.to_mel.to(input.device)
#         mels = to_mel(input).squeeze()
#         pred_mels = to_mel(output).squeeze()
#         mel_mean, mel_std = -4, 4
#         # 3. 确保 log 计算避免为 0，使用更大的 eps
#         mels = (torch.log(mels + 1e-8) - mel_mean) / mel_std
#         pred_mels = (torch.log(pred_mels + 1e-8) - mel_mean) / mel_std
#         F0_real = wav2F0(mels, self.pitch_extractor, input.device, norm=False)
#         F0_pred = wav2F0(pred_mels, self.pitch_extractor, input.device, norm=False)
#         f0_loss = F.smooth_l1_loss(F0_real, F0_pred)
#         print(f"[Generator] f0_loss: shape={f0_loss.shape}, value={f0_loss.item():.6f}, is_nan={torch.isnan(f0_loss).item()}")
#         self.log("f0_loss", f0_loss, prog_bar=False)
        
#         # Energy Loss
#         energy_loss = self.energy_loss(input, output)
#         print(f"[Generator] energy_loss: shape={energy_loss.shape}, value={energy_loss.item():.6f}, is_nan={torch.isnan(energy_loss).item()}")
#         self.log("energy_loss", energy_loss, prog_bar=False)
        
#         # Content Loss 
#         pred_hubert = self.hubert_soft.units(output.to(torch.cuda.current_device()))
#         pred_hubert = pred_hubert.to(input.device)
#         if torch.isnan(pred_hubert).any():
#             print(f"[Generator] pred_hubert contains NaN: {torch.isnan(pred_hubert).sum()}")
#         content_loss = self.hubert_loss(pred_hubert, hubert_like)
#         print(f"[Generator] content_loss: shape={content_loss.shape}, value={content_loss.item():.6f}, is_nan={torch.isnan(content_loss).item()}")
#         self.log("content_loss", content_loss, prog_bar=False)
        
#         # Speaker Embedding Loss
#         pred_spkemb = torch.cat([self.speaker_model.infer_segment(w16)[0] for w16 in output.squeeze().squeeze().cpu()], dim=0)
#         if torch.isnan(pred_spkemb).any():
#             print(f"[Generator] pred_spkemb contains NaN: {torch.isnan(pred_spkemb).sum()}")
#         spk_loss = self.hubert_loss(pred_spkemb, spkemb)
#         print(f"[Generator] spk_loss: shape={spk_loss.shape}, value={spk_loss.item():.6f}, is_nan={torch.isnan(spk_loss).item()}")
#         self.log("spk_loss", spk_loss, prog_bar=False)

#         # 检查输入到判别器的数据
#         print(f"[Generator] Input to STFT discriminator: shape={output.shape}, min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}, has_nan={torch.isnan(output).any().item()}, has_inf={torch.isinf(output).any().item()}")
#         stft_out = self.stft_discriminator(output)
#         if torch.isnan(stft_out).any():
#             print(f"[Generator] stft_out contains NaN: {torch.isnan(stft_out).sum()}, shape={stft_out.shape}")
#             print(f"[Generator] stft_out stats: min={stft_out.min().item():.6f}, max={stft_out.max().item():.6f}, mean={stft_out.mean().item():.6f}")
#         g_stft_loss = torch.mean(torch.relu(1 - stft_out))
#         print(f"[Generator] g_stft_loss: shape={g_stft_loss.shape}, value={g_stft_loss.item():.6f}, is_nan={torch.isnan(g_stft_loss).item()}")
#         self.log("g_stft_loss", g_stft_loss)

#         g_wave_loss = 0
#         g_feat_loss = 0
#         print(f"[Generator] Input to Wave discriminators: input shape={input.shape}, output shape={output.shape}")
#         for i in range(3):
#             print(f"[Generator] Processing Wave discriminator {i}")
#             feats1 = self.wave_discriminators[i](input)
#             feats2 = self.wave_discriminators[i](output)
#             assert len(feats1) == len(feats2)
#             if torch.isnan(feats2[-1]).any():
#                 print(f"[Generator] wave_loss[{i}] feats2[-1] contains NaN: {torch.isnan(feats2[-1]).sum()}, shape={feats2[-1].shape}, min={feats2[-1].min().item():.6f}, max={feats2[-1].max().item():.6f}")
#             wave_loss_i = torch.mean(torch.relu(1 - feats2[-1]))
#             if torch.isnan(wave_loss_i).any():
#                 print(f"[Generator] wave_loss[{i}] contains NaN")
#             g_wave_loss += wave_loss_i
            
#             # 计算特征损失，处理形状不匹配的情况
#             feat_losses = []
#             for idx, (f1, f2) in enumerate(zip(feats1[:-1], feats2[:-1])):
#                 # 检查形状是否匹配
#                 if f1.shape != f2.shape:
#                     print(f"[Generator] Warning: feat_loss[{i}][{idx}] shape mismatch: f1={f1.shape}, f2={f2.shape}")
#                     # 如果形状不匹配，截断到较小的尺寸
#                     min_batch = min(f1.shape[0], f2.shape[0])
#                     min_channels = min(f1.shape[1], f2.shape[1])
#                     min_length = min(f1.shape[2], f2.shape[2])
#                     f1 = f1[:min_batch, :min_channels, :min_length]
#                     f2 = f2[:min_batch, :min_channels, :min_length]
                
#                 feat_losses.append(torch.mean(torch.abs(f1 - f2)))
            
#             feat_loss_i = sum(feat_losses) / len(feat_losses) if len(feat_losses) > 0 else torch.tensor(0.0, device=input.device)
#             if torch.isnan(feat_loss_i).any():
#                 print(f"[Generator] feat_loss[{i}] contains NaN")
#             g_feat_loss += feat_loss_i
#         g_wave_loss = g_wave_loss / 3
#         print(f"[Generator] g_wave_loss: shape={g_wave_loss.shape}, value={g_wave_loss.item():.6f}, is_nan={torch.isnan(g_wave_loss).item()}")
#         print(f"[Generator] g_feat_loss (before /3): shape={g_feat_loss.shape}, value={g_feat_loss.item():.6f}, is_nan={torch.isnan(g_feat_loss).item()}")
#         self.log("g_wave_loss", g_wave_loss / 3)
#         self.log("g_feat_loss", g_feat_loss / 3)

#         g_rec_loss = self.rec_loss(output[:, 0, :], input[:, 0, :])
#         print(f"[Generator] g_rec_loss: shape={g_rec_loss.shape}, value={g_rec_loss.item():.6f}, is_nan={torch.isnan(g_rec_loss).item()}")
#         self.log("g_rec_loss", g_rec_loss, prog_bar=True)

#         g_feat_loss = g_feat_loss / 3
#         g_adv_loss = (g_stft_loss + g_wave_loss) / 4
#         print(f"[Generator] g_adv_loss: shape={g_adv_loss.shape}, value={g_adv_loss.item():.6f}, is_nan={torch.isnan(g_adv_loss).item()}")
#         g_loss = g_adv_loss  + g_rec_loss +  100 * g_feat_loss  + 0.5 * f0_loss  + 0.5 * energy_loss  + spk_loss + content_loss
#         print(f"[Generator] g_loss components: g_adv={g_adv_loss.item():.6f}, g_rec={g_rec_loss.item():.6f}, g_feat={(100*g_feat_loss).item():.6f}, f0={(0.5*f0_loss).item():.6f}, energy={(0.5*energy_loss).item():.6f}, spk={spk_loss.item():.6f}, content={content_loss.item():.6f}")
#         print(f"[Generator] g_loss: shape={g_loss.shape}, value={g_loss.item():.6f}, is_nan={torch.isnan(g_loss).item()}, is_inf={torch.isinf(g_loss).item()}")

#         self.log("g_loss", g_loss, prog_bar=True)

#         # 如果损失是 NaN/Inf，跳过这一步的更新
#         if torch.isnan(g_loss).any() or torch.isinf(g_loss).any():
#             print(f"[Generator] ⚠ Skipping backward pass: g_loss contains NaN/Inf")
#             optimizer_g.zero_grad()
#             self.untoggle_optimizer(optimizer_g)
#         else:
#             self.manual_backward(g_loss)
#             # 梯度裁剪：防止梯度爆炸（spec 已冻结，不需要裁剪）
#             torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
#             torch.nn.utils.clip_grad_norm_(self.reencoder.parameters(), max_norm=1.0)
#             torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
            
#             # 检查梯度是否包含 NaN，并将 NaN 梯度置零（spec 已冻结，不需要检查）
#             has_nan_grad = False
#             for name, param in chain(self.encoder.named_parameters(), 
#                                     self.reencoder.named_parameters(), self.decoder.named_parameters()):
#                 if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
#                     print(f"[Generator] ⚠ Gradient contains NaN/Inf: {name}, zeroing it")
#                     has_nan_grad = True
#                     param.grad.zero_()  # 将 NaN 梯度置零
            
#             # 关键：即使有 NaN 梯度（已置零），也必须执行 step()，否则 Lightning 不会增加 global_step
#             if has_nan_grad:
#                 print(f"[Generator] ⚠ NaN gradients detected and zeroed, but still executing optimizer.step() for Lightning tracking")
#             optimizer_g.step()  # 必须执行，让 Lightning 跟踪训练进度
#             optimizer_g.zero_grad()
#             self.untoggle_optimizer(optimizer_g)
#         # sch.step()

#         # train discriminator
#         output, hubert_like = self.forward(input, spkemb)
        
#         print(f"[Discriminator] Input to STFT discriminator (real): shape={input.shape}, min={input.min().item():.6f}, max={input.max().item():.6f}, has_nan={torch.isnan(input).any().item()}")
#         stft_out = self.stft_discriminator(input)
#         if torch.isnan(stft_out).any():
#             print(f"[Discriminator] stft_out (input) contains NaN: {torch.isnan(stft_out).sum()}, shape={stft_out.shape}")
#         d_stft_loss = torch.mean(torch.relu(1 - stft_out))
#         print(f"[Discriminator] d_stft_loss (real): shape={d_stft_loss.shape}, value={d_stft_loss.item():.6f}, is_nan={torch.isnan(d_stft_loss).item()}")
        
#         print(f"[Discriminator] Input to STFT discriminator (fake): shape={output.shape}, min={output.min().item():.6f}, max={output.max().item():.6f}, has_nan={torch.isnan(output).any().item()}")
#         stft_out = self.stft_discriminator(output)
#         if torch.isnan(stft_out).any():
#             print(f"[Discriminator] stft_out (output) contains NaN: {torch.isnan(stft_out).sum()}, shape={stft_out.shape}")
#         d_stft_loss_fake = torch.mean(torch.relu(1 + stft_out))
#         print(f"[Discriminator] d_stft_loss (fake): shape={d_stft_loss_fake.shape}, value={d_stft_loss_fake.item():.6f}, is_nan={torch.isnan(d_stft_loss_fake).item()}")
#         d_stft_loss += d_stft_loss_fake
#         print(f"[Discriminator] d_stft_loss (total): shape={d_stft_loss.shape}, value={d_stft_loss.item():.6f}, is_nan={torch.isnan(d_stft_loss).item()}")

#         d_wave_loss = 0
#         for i in range(3):
#             print(f"[Discriminator] Processing Wave discriminator {i} (real)")
#             feats = self.wave_discriminators[i](input)
#             if torch.isnan(feats[-1]).any():
#                 print(f"[Discriminator] wave_feats[{i}] (input) contains NaN: {torch.isnan(feats[-1]).sum()}, shape={feats[-1].shape}")
#             wave_loss_real = torch.mean(torch.relu(1 - feats[-1]))
#             if torch.isnan(wave_loss_real).any():
#                 print(f"[Discriminator] wave_loss[{i}] (real) contains NaN")
#             d_wave_loss += wave_loss_real
            
#             print(f"[Discriminator] Processing Wave discriminator {i} (fake)")
#             feats = self.wave_discriminators[i](output)
#             if torch.isnan(feats[-1]).any():
#                 print(f"[Discriminator] wave_feats[{i}] (output) contains NaN: {torch.isnan(feats[-1]).sum()}, shape={feats[-1].shape}")
#             wave_loss_fake = torch.mean(torch.relu(1 + feats[-1]))
#             if torch.isnan(wave_loss_fake).any():
#                 print(f"[Discriminator] wave_loss[{i}] (fake) contains NaN")
#             d_wave_loss += wave_loss_fake
#         print(f"[Discriminator] d_wave_loss (total): shape={d_wave_loss.shape}, value={d_wave_loss.item():.6f}, is_nan={torch.isnan(d_wave_loss).item()}")

#         d_loss = (d_stft_loss + d_wave_loss) / 4
#         print(f"[Discriminator] d_loss: shape={d_loss.shape}, value={d_loss.item():.6f}, is_nan={torch.isnan(d_loss).item()}, is_inf={torch.isinf(d_loss).item()}")
#         print(f"[Discriminator] d_loss components: d_stft={d_stft_loss.item():.6f}, d_wave={d_wave_loss.item():.6f}")

#         self.log("d_stft_loss", d_stft_loss)
#         self.log("d_wave_loss", d_wave_loss / 3)
#         self.log("d_loss", d_loss, prog_bar=True)

#         # 如果损失是 NaN/Inf，跳过这一步的更新
#         if torch.isnan(d_loss).any() or torch.isinf(d_loss).any():
#             print(f"[Discriminator] ⚠ Skipping backward pass: d_loss contains NaN/Inf")
#             optimizer_d.zero_grad()
#             self.untoggle_optimizer(optimizer_d)
#         else:
#             self.manual_backward(d_loss)
#             # 梯度裁剪：防止梯度爆炸
#             torch.nn.utils.clip_grad_norm_(self.wave_discriminators.parameters(), max_norm=1.0)
#             torch.nn.utils.clip_grad_norm_(self.stft_discriminator.parameters(), max_norm=1.0)
            
#             # 检查梯度是否包含 NaN，并将 NaN 梯度置零
#             has_nan_grad = False
#             for name, param in chain(self.wave_discriminators.named_parameters(), self.stft_discriminator.named_parameters()):
#                 if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
#                     print(f"[Discriminator] ⚠ Gradient contains NaN/Inf: {name}, zeroing it")
#                     has_nan_grad = True
#                     param.grad.zero_()  # 将 NaN 梯度置零
            
#             # 关键：即使有 NaN 梯度（已置零），也必须执行 step()，否则 Lightning 不会增加 global_step
#             if has_nan_grad:
#                 print(f"[Discriminator] ⚠ NaN gradients detected and zeroed, but still executing optimizer.step() for Lightning tracking")
#             optimizer_d.step()  # 必须执行，让 Lightning 跟踪训练进度
#             optimizer_d.zero_grad()
#             self.untoggle_optimizer(optimizer_d)
        
#         # 返回 loss 以便 Lightning 跟踪训练进度
#         # 在 manual_optimization 模式下，必须返回 Tensor 或不返回（None）
#         # 返回 Tensor 以确保 Lightning 能正确跟踪训练进度并触发 callbacks
#         if torch.isnan(d_loss).any() or torch.isinf(d_loss).any():
#             loss_tensor = torch.tensor(0.0, device=d_loss.device, requires_grad=False)
#         else:
#             # 确保返回的是标量 Tensor（不是 Python float）
#             if d_loss.numel() == 1:
#                 loss_tensor = d_loss.detach()  # 已经是标量 tensor
#             else:
#                 loss_tensor = d_loss.mean().detach() 
                
#         if self.global_step % 100 == 0:
#             print(f"[Debug] global_step = {self.global_step}")
#              # 取平均值并转为标量 tensor
#         return None

#     def validation_step(self, batch, batch_idx):
#         inputs = batch[:, None, :]
#         input = inputs[:, :, :32270] # normal / ppw
#         whisper = inputs[:, :,32270:32270*2] # ppw
#         spkemb = torch.cat([self.speaker_model.infer_segment(w16)[0] for w16 in input.squeeze().squeeze().cpu()], dim=0)
#         with torch.no_grad():
#             output, _ = self.forward(whisper, spkemb)
            
#             # 检查 output 是否包含 NaN/Inf
#             if torch.isnan(output).any() or torch.isinf(output).any():
#                 print(f"[Validation] Output contains NaN/Inf: NaN={torch.isnan(output).sum()}, Inf={torch.isinf(output).sum()}")
#                 # 如果包含 NaN/Inf，跳过 PESQ 计算
#                 self.log("val_pesq", 0.0, on_epoch=True, prog_bar=True, sync_dist=True)
#                 return
            
#             val_pesq_tot = 0
#             val_pesq_count = 0
#             MAX_WAV_VALUE = 32767.0
#             for y_16k, y_g_hat_16k in zip(input, output):
#                 # 检查单个样本是否包含 NaN/Inf
#                 if torch.isnan(y_16k).any() or torch.isinf(y_16k).any() or torch.isnan(y_g_hat_16k).any() or torch.isinf(y_g_hat_16k).any():
#                     print(f"[Validation] Skipping sample with NaN/Inf")
#                     continue
                
#                 # 确保长度一致
#                 min_len = min(y_16k.shape[1], y_g_hat_16k.shape[1])
#                 y_16k = y_16k[:, :min_len]
#                 y_g_hat_16k = y_g_hat_16k[:, :min_len]
                
#                 # 转换为 numpy 数组
#                 y_16k_np = y_16k[0].cpu().numpy()
#                 y_g_hat_16k_np = y_g_hat_16k[0].cpu().numpy()
                
#                 # 检查 numpy 数组是否包含 NaN/Inf
#                 import numpy as np
#                 if np.isnan(y_16k_np).any() or np.isinf(y_16k_np).any() or np.isnan(y_g_hat_16k_np).any() or np.isinf(y_g_hat_16k_np).any():
#                     print(f"[Validation] Skipping sample: numpy array contains NaN/Inf")
#                     continue
                
#                 # 限制范围并转换为整数
#                 y_16k_np = np.clip(y_16k_np, -1.0, 1.0)
#                 y_g_hat_16k_np = np.clip(y_g_hat_16k_np, -1.0, 1.0)
                
#                 y_int_16k = (y_16k_np * MAX_WAV_VALUE).astype(np.int16)
#                 y_g_hat_int_16k = (y_g_hat_16k_np * MAX_WAV_VALUE).astype(np.int16)
                
#                 # 再次检查转换后的数组
#                 if np.isnan(y_int_16k).any() or np.isinf(y_int_16k).any() or np.isnan(y_g_hat_int_16k).any() or np.isinf(y_g_hat_int_16k).any():
#                     print(f"[Validation] Skipping sample: converted array contains NaN/Inf")
#                     continue
                
#                 try:
#                     pesq_score = pesq(16000, y_int_16k, y_g_hat_int_16k, "wb")
#                     if not np.isnan(pesq_score) and not np.isinf(pesq_score):
#                         val_pesq_tot += pesq_score
#                         val_pesq_count += 1
#                 except (ValueError, RuntimeError) as e:
#                     print(f"[Validation] PESQ calculation failed: {e}")
#                     continue
            
#             # 计算平均 PESQ
#             avg_pesq = val_pesq_tot / val_pesq_count if val_pesq_count > 0 else 0.0
#             self.log("val_pesq", avg_pesq, on_epoch=True, prog_bar=True, sync_dist=True)
#             self.log("val_pesq_count", val_pesq_count, on_epoch=True, prog_bar=False, sync_dist=True)

#     def train_dataloader(self):
#         return self._make_dataloader(True)
    
#     def val_dataloader(self):
#         return self._make_dataloader_val()

#     def _make_dataloader(self, train: bool):
        
#         class VoiceDataset(torch.utils.data.Dataset):
#             def __init__(self, dataset, sample_rate, segment_length):
#                 self._dataset = dataset
#                 self._sample_rate = sample_rate
#                 self._segment_length = segment_length

#             def __getitem__(self, index):
#                 import random
#                 data = self._dataset[index]
#                 # Handle different dataset return formats:
#                 # WTIMIT returns: (waveform, waveform_p, waveform_v, sample_rate, input_file) - 5 elements
#                 # MultiWhisperDataset returns: (norm_waveform, whsp_waveform, vad_waveform, sample_rate) - 4 elements
#                 if len(data) == 5:
#                     # WTIMIT format: (waveform, waveform_p, waveform_v, sample_rate, input_file)
#                     x, x_p, x_v, sample_rate, _ = data
#                 elif len(data) == 4:
#                     # MultiWhisperDataset format: (norm_waveform, whsp_waveform, vad_waveform, sample_rate)
#                     x, x_p, x_v, sample_rate = data
#                 else:
#                     raise ValueError(f"Unexpected data format: expected 4 or 5 elements, got {len(data)}")
                
#                 # Ensure sample_rate is int
#                 if isinstance(sample_rate, torch.Tensor):
#                     sample_rate = int(sample_rate.item())
#                 sample_rate = int(sample_rate)
                
#                 # 检查原始数据是否包含NaN/Inf（仅用于诊断）
#                 if torch.isnan(x).any() or torch.isinf(x).any():
#                     print(f"[DataLoader] x contains NaN/Inf at index {index}: NaN={torch.isnan(x).sum()}, Inf={torch.isinf(x).sum()}")
#                 if torch.isnan(x_p).any() or torch.isinf(x_p).any():
#                     print(f"[DataLoader] x_p contains NaN/Inf at index {index}: NaN={torch.isnan(x_p).sum()}, Inf={torch.isinf(x_p).sum()}")
#                 if torch.isnan(x_v).any() or torch.isinf(x_v).any():
#                     print(f"[DataLoader] x_v contains NaN/Inf at index {index}: NaN={torch.isnan(x_v).sum()}, Inf={torch.isinf(x_v).sum()}")
                
#                 target_len = min(x.shape[-1], x_p.shape[-1], x_v.shape[-1])
#                 x = process_signal(x,sample_rate, self._sample_rate, target_len, self._segment_length)
#                 x_p = process_signal(x_p,sample_rate, self._sample_rate, target_len, self._segment_length)
#                 x_v = process_signal(x_v,sample_rate, self._sample_rate, target_len, self._segment_length)
                
#                 # 检查处理后的数据（仅用于诊断）
#                 if torch.isnan(x).any() or torch.isinf(x).any():
#                     print(f"[DataLoader] x after process_signal contains NaN/Inf at index {index}")
#                 if torch.isnan(x_p).any() or torch.isinf(x_p).any():
#                     print(f"[DataLoader] x_p after process_signal contains NaN/Inf at index {index}")
#                 if torch.isnan(x_v).any() or torch.isinf(x_v).any():
#                     print(f"[DataLoader] x_v after process_signal contains NaN/Inf at index {index}")
                
#                 pos = random.randint(0, x.shape[0] - self._segment_length)
#                 x = x[pos:pos + self._segment_length]
#                 x_p = x_p[pos:pos + self._segment_length]
#                 x_v = x_v[pos:pos + self._segment_length]
                
#                 output = torch.cat((x, x_p, x_v), 0)
#                 return output

#             def __len__(self):
#                 return len(self._dataset)
            
#         def collate(examples):
#             return torch.stack(examples)

#         if self.hparams.dataset == 'ljspeech':
#             ds = LJSPEECH(self.datasets_root)
#         elif self.hparams.dataset == 'libritts':
#             ds = LIBRITTS(self.datasets_root)
#         elif self.hparams.dataset == 'timit':
#             ds = TIMIT(self.datasets_root, training=True)
#         elif self.hparams.dataset == 'wtimit':
#             ds = WTIMIT(self.datasets_root)
#         elif self.hparams.dataset == 'multi_whisper':
#             # Load multiple datasets from JSON files
#             json_paths = [
#                 '/mnt/workspace/guanzifan/data/wtimit_16k/aligned_pairs_uid.json',
#                 '/mnt/data/share/AISHELL6-Whisper/aligned_pairs_uid_new.json',
#                 '/mnt/data/share/WHSP_LGU/aligned_pairs_uid_new.json',
#             ]
#             ds = MultiWhisperDataset(json_paths, training=True)
#         else:
#             raise ValueError(f"Unknown dataset: {self.hparams.dataset}")
#         ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

#         # subset_len = int(0.2 * len(ds))
#         # ds = torch.utils.data.Subset(ds, range(subset_len))

#         loader = torch.utils.data.DataLoader(
#             ds, batch_size=self.hparams['batch_size'], shuffle=True,
#             collate_fn=collate, num_workers=7)
#         return loader

#     def _make_dataloader_val(self, sub_rate: float=0.1):

#         class VoiceDataset(torch.utils.data.Dataset):
#             def __init__(self, dataset, sample_rate, segment_length):
#                 self._dataset = dataset
#                 self._sample_rate = sample_rate
#                 self._segment_length = segment_length

#             def __getitem__(self, index):
#                 import random
#                 data = self._dataset[index]
#                 # Handle different dataset return formats
#                 # WHISPER returns: (waveform, waveform_p, sample_rate) - 3 elements
#                 # MultiWhisperDataset returns: (norm_waveform, whsp_waveform, vad_waveform, sample_rate) - 4 elements
#                 if len(data) == 4:
#                     # MultiWhisperDataset format: (norm_waveform, whsp_waveform, vad_waveform, sample_rate)
#                     x, x_p, _, sample_rate = data
#                 elif len(data) == 3:
#                     # WHISPER format: (waveform, waveform_p, sample_rate)
#                     x, x_p, sample_rate = data
#                 else:
#                     raise ValueError(f"Unexpected data format: expected 3 or 4 elements, got {len(data)}")
                
#                 # Ensure sample_rate is int
#                 if isinstance(sample_rate, torch.Tensor):
#                     sample_rate = int(sample_rate.item())
#                 sample_rate = int(sample_rate)
                
#                 target_len = min(x.shape[-1], x_p.shape[-1])
#                 x = process_signal(x,sample_rate, self._sample_rate, target_len, self._segment_length)
#                 x_p = process_signal(x_p,sample_rate, self._sample_rate, target_len, self._segment_length)
#                 pos = random.randint(0, x.shape[0] - self._segment_length)
#                 x = x[pos:pos + self._segment_length]
#                 x_p = x_p[pos:pos + self._segment_length]
#                 output = torch.cat((x, x_p), 0)
#                 return output

#             def __len__(self):
#                 return len(self._dataset)

#         def collate(examples):
#             return torch.stack(examples)

#         # For validation, use a subset of the training multi_whisper dataset
#         if self.hparams.dataset == 'multi_whisper':
#             json_paths = [
#                 '/mnt/workspace/guanzifan/data/wtimit_16k/aligned_pairs_uid.json',
#                 '/mnt/data/share/AISHELL6-Whisper/aligned_pairs_uid_new.json',
#                 '/mnt/data/share/WHSP_LGU/aligned_pairs_uid_new.json',
#             ]
#             ds = MultiWhisperDataset(json_paths, training=False)
#             subset_len = int(sub_rate * len(ds))
#             ds = torch.utils.data.Subset(ds, range(subset_len))
#         else:
#             ds = WHISPER(self.datasets_root)
#             subset_len = int(sub_rate * len(ds))
#             ds = torch.utils.data.Subset(ds, range(subset_len))

#         ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

#         loader = torch.utils.data.DataLoader(
#             ds, batch_size=self.hparams['batch_size'], shuffle=False,
#             collate_fn=collate, num_workers=7)
#         return loader


# def train():
#     config = {
#         # Configuration ID
#         'id': "experiments",
#         'name': "s2uu2s",
#         'version': "s2uu2s-wtimit",
#         # Training configuration
#         'batch_size': 32, #s2u 128, s2uu2s 32
#         'save_checkpoint_dir': "/mnt/workspace/guanzifan/distillw2n/",
#         'restore_checkpoint_path': "/mnt/workspace/guanzifan/distillw2n/experiments/s2uu2s/epoch.440-step.409942.ckpt",
#         'resume_training': True,
#         'training_epochs': 10000,
#         'log_every_n_steps': 2,
#         'dataset': 'multi_whisper',
#         'datasets_root': '',  # Not used for multi_whisper dataset
#         'feature_type': 'mfcc',
#         'reen_nn_type': 'adapt',
#     }
#     config = Config(config)
#     model = StreamableModel(
#         n_channels=config.n_channels,
#         n_embed_dim=config.n_embed_dim,
#         n_encoder_layer=config.n_encoder_layer,
#         padding=config.padding,
#         batch_size=config.batch_size,
#         sample_rate=config.sample_rate,
#         segment_length=config.segment_length,
#         lr=config.lr,
#         b1=config.b1,
#         b2=config.b2,
#         dataset=config.dataset,
#         feature_type=config.feature_type,
#         reen_nn_type=config.reen_nn_type,
#         pseudo_rate=config.pseudo_rate,
#         datasets_root=config.datasets_root,
#         F0_model_path=config.F0_model_path)
    
#     # 手动加载模型权重（不加载优化器状态，避免参数组不匹配）
#     if config.resume_training and config.restore_checkpoint_path:
#         print(f"Loading model weights from checkpoint: {config.restore_checkpoint_path}")
#         print("=" * 80)
#         print("🔒 STRICT LOADING POLICY: Only loading generator trainable weights")
#         print("   - ✅ Loading: spec, encoder, reencoder, decoder (trainable params only)")
#         print("   - ❌ Skipping: ALL discriminator weights (wave_discriminators, stft_discriminator)")
#         print("   - ❌ Skipping: ALL buffers (BatchNorm running_mean/var, etc.)")
#         print("   - ❌ Skipping: ALL optimizer states, scalers, etc.")
#         print("=" * 80)
#         try:
#             checkpoint = torch.load(config.restore_checkpoint_path, map_location='cpu', weights_only=True)
#             state_dict = checkpoint.get('state_dict', checkpoint)
            
#             # 定义生成器模块的前缀（只加载这些模块的可训练参数）
#             # 根据 checkpoint 分析：spec(8), reencoder(9), encoder(113), decoder(60) 都是干净的
#             generator_prefixes = ['spec.', 'encoder.', 'reencoder.', 'decoder.']
            
#             # 定义需要跳过的模块（完全跳过判别器和预训练模型）
#             skip_prefixes = [
#                 'wave_discriminators.',  # 63个参数，全部有 NaN
#                 'stft_discriminator.',   # 130个参数，大部分有 NaN
#                 'discriminator.',
#                 'speaker_model.',        # 预训练模型，不应该从 checkpoint 加载
#                 'to_mel.',              # torchaudio transform，不应该加载
#                 'hubert_soft.',         # 预训练模型，不应该加载
#                 'pitch_extractor.',     # 预训练模型，不应该加载
#             ]
            
#             # 定义需要跳过的 Buffer（BatchNorm 的 running_mean, running_var 等）
#             skip_suffixes = ['.running_mean', '.running_var', '.num_batches_tracked', '.buffer']
            
#             # 定义需要跳过的特殊参数（即使 checkpoint 中没有 NaN，这些参数在训练时也容易产生 NaN 梯度）
#             # 根据训练日志：mel_basis, wsin, wcos 在训练时产生了 NaN 梯度
#             skip_special_params = [
#                 'mel_basis',  # spec.melspec_layer.mel_basis - 训练时产生 NaN 梯度
#                 'wsin',       # spec.melspec_layer.stft.wsin - 训练时产生 NaN 梯度
#                 'wcos',       # spec.melspec_layer.stft.wcos - 训练时产生 NaN 梯度
#                 'window_mask', # spec.melspec_layer.stft.window_mask - 可能不稳定
#                 'amin',       # spec.spec.amin - 可能不稳定
#                 'ref',        # spec.spec.ref - 可能不稳定
#             ]
            
#             # 获取当前模型的可训练参数名称（用于验证）
#             model_trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
#             generator_trainable_names = {name for name in model_trainable_params.keys() 
#                                         if any(name.startswith(prefix) for prefix in generator_prefixes)}
            
#             print(f"\n📊 Checkpoint Analysis:")
#             print(f"   Total keys in checkpoint: {len(state_dict)}")
#             print(f"   Generator trainable params in model: {len(generator_trainable_names)}")
            
#             # 精准过滤：只加载生成器的可训练参数
#             generator_state_dict = {}
#             skipped_keys = []
#             nan_keys = []
            
#             for key, value in state_dict.items():
#                 # 跳过判别器的所有参数
#                 if any(key.startswith(skip_prefix) for skip_prefix in skip_prefixes):
#                     skipped_keys.append(key)
#                     continue
                
#                 # 跳过所有 Buffer（BatchNorm 的 running_mean, running_var 等）
#                 if any(key.endswith(suffix) for suffix in skip_suffixes):
#                     skipped_keys.append(key)
#                     continue
                
#                 # 跳过特殊的不稳定参数（mel_basis, wsin, wcos 等）
#                 if any(special_param in key for special_param in skip_special_params):
#                     skipped_keys.append(key)
#                     print(f"  ⚠ Skipping unstable parameter: {key} (will use random init)")
#                     continue
                
#                 # 只加载生成器的可训练参数
#                 is_generator_param = any(key.startswith(prefix) for prefix in generator_prefixes)
#                 if not is_generator_param:
#                     skipped_keys.append(key)
#                     continue
                
#                 # 检查是否是模型中的可训练参数（不是 buffer）
#                 if key not in generator_trainable_names:
#                     skipped_keys.append(key)
#                     continue
                
#                 # 检查是否包含 NaN/Inf
#                 if isinstance(value, torch.Tensor):
#                     if torch.isnan(value).any() or torch.isinf(value).any():
#                         nan_keys.append(key)
#                         nan_count = torch.isnan(value).sum().item()
#                         inf_count = torch.isinf(value).sum().item()
#                         print(f"  ⚠ Skipping {key}: NaN={nan_count}/{value.numel()}, Inf={inf_count}/{value.numel()}")
#                         continue
                
#                 # 验证形状匹配
#                 if key in model_trainable_params:
#                     model_param = model_trainable_params[key]
#                     if value.shape != model_param.shape:
#                         print(f"  ⚠ Shape mismatch for {key}: checkpoint {value.shape} vs model {model_param.shape}, skipping")
#                         skipped_keys.append(key)
#                         continue
                
#                 # 通过所有检查，添加到加载列表
#                 generator_state_dict[key] = value
            
#             print(f"\n📦 Loading Summary:")
#             print(f"   ✅ Generator weights to load: {len(generator_state_dict)}")
#             print(f"   ❌ Skipped (discriminators): {sum(1 for k in skipped_keys if any(k.startswith(p) for p in skip_prefixes))}")
#             print(f"   ❌ Skipped (buffers): {sum(1 for k in skipped_keys if any(k.endswith(s) for s in skip_suffixes))}")
#             print(f"   ❌ Skipped (other): {len(skipped_keys) - sum(1 for k in skipped_keys if any(k.startswith(p) for p in skip_prefixes) or any(k.endswith(s) for s in skip_suffixes))}")
#             print(f"   ⚠ Skipped (NaN/Inf): {len(nan_keys)}")
            
#             if nan_keys:
#                 print(f"\n⚠ Warning: Found {len(nan_keys)} generator keys with NaN/Inf in checkpoint!")
#                 print(f"   These keys will NOT be loaded (using random initialization instead)")
#                 print(f"   First few: {nan_keys[:5]}")
            
#             # 加载生成器权重
#             print(f"\n🔄 Loading generator weights...")
#             missing_keys, unexpected_keys = model.load_state_dict(generator_state_dict, strict=False)
            
#             if missing_keys:
#                 print(f"   ⚠ Missing keys (will use random init): {len(missing_keys)}")
#                 if len(missing_keys) <= 10:
#                     for key in missing_keys:
#                         print(f"      - {key}")
#                 else:
#                     for key in missing_keys[:5]:
#                         print(f"      - {key}")
#                     print(f"      ... and {len(missing_keys) - 5} more")
            
#             if unexpected_keys:
#                 print(f"   ⚠ Unexpected keys (ignored): {len(unexpected_keys)}")
            
#             print(f"   ✅ Successfully loaded {len(generator_state_dict)} generator weights")
            
#             # 验证加载后的模型权重
#             print(f"\n🔍 Verifying loaded weights...")
#             post_load_nan = []
#             for name, param in model.named_parameters():
#                 if any(name.startswith(prefix) for prefix in generator_prefixes):
#                     # 特别检查不稳定的参数
#                     is_unstable_param = any(special_param in name for special_param in skip_special_params)
#                     if torch.isnan(param).any() or torch.isinf(param).any():
#                         post_load_nan.append(name)
#                         nan_count = torch.isnan(param).sum().item()
#                         inf_count = torch.isinf(param).sum().item()
#                         param_type = " (unstable param)" if is_unstable_param else ""
#                         print(f"  ⚠ {name}: NaN={nan_count}/{param.numel()}, Inf={inf_count}/{param.numel()}{param_type}")
                    
#                     # 即使没有 NaN，如果是不稳定参数，也检查是否需要重新初始化
#                     if is_unstable_param and (torch.abs(param).max() > 1e6 or torch.abs(param).max() < 1e-10):
#                         print(f"  ⚠ {name}: Extreme values detected (max={torch.abs(param).max().item():.2e}), re-initializing...")
#                         post_load_nan.append(name)
            
#             if post_load_nan:
#                 print(f"\n⚠ WARNING: {len(post_load_nan)} generator parameters contain NaN/Inf after loading!")
#                 print(f"   This should not happen if checkpoint was clean. Affected: {post_load_nan[:5]}")
#                 print(f"   🔧 Re-initializing affected parameters...")
                
#                 # 重新初始化包含 NaN 的参数
#                 for name in post_load_nan:
#                     param = dict(model.named_parameters())[name]
#                     with torch.no_grad():
#                         if 'mel_basis' in name:
#                             # mel_basis 应该从 librosa 重新初始化
#                             print(f"      Re-initializing {name} (mel_basis)...")
#                             # 这里需要根据实际的 mel_basis 形状重新初始化
#                             # 暂时使用随机初始化
#                             nn.init.xavier_uniform_(param)
#                         elif 'wsin' in name or 'wcos' in name:
#                             # STFT 窗口函数，使用正弦/余弦初始化
#                             print(f"      Re-initializing {name} (STFT window)...")
#                             # 暂时使用随机初始化
#                             nn.init.xavier_uniform_(param)
#                         elif 'linear' in name or 'conv' in name:
#                             # 线性层或卷积层
#                             print(f"      Re-initializing {name}...")
#                             if len(param.shape) == 1:
#                                 nn.init.zeros_(param)
#                             else:
#                                 nn.init.xavier_uniform_(param)
#                         else:
#                             # 其他参数
#                             print(f"      Re-initializing {name}...")
#                             if len(param.shape) == 1:
#                                 nn.init.zeros_(param)
#                             else:
#                                 nn.init.xavier_uniform_(param)
                
#                 # 再次验证
#                 post_reinit_nan = []
#                 for name, param in model.named_parameters():
#                     if any(name.startswith(prefix) for prefix in generator_prefixes):
#                         if torch.isnan(param).any() or torch.isinf(param).any():
#                             post_reinit_nan.append(name)
                
#                 if post_reinit_nan:
#                     print(f"   ⚠ Still {len(post_reinit_nan)} parameters contain NaN/Inf after re-init!")
#                 else:
#                     print(f"   ✅ All generator weights are clean after re-initialization")
#             else:
#                 print(f"   ✅ All generator weights are clean")
            
#             # 验证判别器是随机初始化的（不应该从 checkpoint 加载）
#             print(f"\n🔍 Verifying discriminators are fresh (not loaded from checkpoint)...")
#             disc_has_nan = False
#             for name, param in model.named_parameters():
#                 if any(name.startswith(prefix) for prefix in skip_prefixes):
#                     if torch.isnan(param).any() or torch.isinf(param).any():
#                         disc_has_nan = True
#                         print(f"  ⚠ {name}: Contains NaN/Inf (should be fresh random init)")
            
#             if disc_has_nan:
#                 print(f"   ⚠ Warning: Some discriminator weights contain NaN/Inf!")
#                 print(f"   This should not happen - discriminators should be fresh random init")
#             else:
#                 print(f"   ✅ Discriminators are clean (fresh random initialization)")
            
#             print("=" * 80)
#             print("✅ Checkpoint loading completed successfully")
#             print("=" * 80)
            
#         except Exception as e:
#             print(f"⚠ Warning: Failed to load checkpoint: {e}")
#             import traceback
#             traceback.print_exc()
#             print("Continuing with randomly initialized weights...")
    
#     pl.seed_everything(config.seed, workers=True)
    
#     # 计算 checkpoint 保存路径
#     checkpoint_dir = f"{config.save_checkpoint_dir}{config.id}/{config.name}/{config.version}/checkpoints"
#     print(f"Checkpoint will be saved to: {checkpoint_dir}")
    
#     trainer = pl.Trainer(
#         max_epochs=config.training_epochs,
#         log_every_n_steps=config.log_every_n_steps,
#         precision='16-mixed',
#         logger=pl.loggers.TensorBoardLogger(config.save_checkpoint_dir+config.id, name=config.name, version=config.version),
#         callbacks=[
#             pl.callbacks.ModelCheckpoint(
#                 dirpath=checkpoint_dir,  # 明确指定保存路径
#                 filename='epoch={epoch}-step={step}',
#                 save_last=True,  # 保存最后一个 checkpoint
#                 every_n_train_steps=1000,  # 每1000步保存一次（不依赖 epoch 或验证集）
#                 save_on_train_epoch_end=True,  # 在训练 epoch 结束时也保存
#                 auto_insert_metric_name=False,
#                 enable_version_counter=False  # 禁用版本计数器，避免路径问题
#             )
#         ],
#         strategy='ddp_find_unused_parameters_true'
#     )
#     # 不传入 ckpt_path，避免加载优化器状态（已在上面手动加载模型权重）
#     trainer.fit(
#         model,
#         ckpt_path=None
#     )

#     return model


# if __name__ == "__main__":
#     train() 



