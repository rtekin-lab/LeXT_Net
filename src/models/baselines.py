# -*- coding: utf-8 -*-
"""
baselines.py  —  Baseline model architectures
==============================================
Four baseline models for comparative evaluation, plus AttentionUNet.
All models share the same interface: (B, T, 3) → (B, T, 12).
No external dependencies beyond PyTorch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. BiLSTM ─────────────────────────────────────────────────────────────────

class BiLSTMModel(nn.Module):
    """Bidirectional LSTM — 2 layers, hidden=128 per direction."""
    def __init__(self, input_size=3, hidden_size=128,
                 num_layers=2, output_size=12):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.1 if num_layers > 1 else 0.0)
        self.fc   = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.relu(out))


# ── 2. TCN baseline (Gated Dilated TCN) ──────────────────────────────────────

class _GatedTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv_f = nn.Conv1d(channels, channels, kernel_size,
                                padding=pad, dilation=dilation)
        self.conv_g = nn.Conv1d(channels, channels, kernel_size,
                                padding=pad, dilation=dilation)
        self.bn     = nn.BatchNorm1d(channels)

    def forward(self, x):
        h = torch.tanh(self.conv_f(x)) * torch.sigmoid(self.conv_g(x))
        return F.relu(self.bn(h) + x)


class TCNBaseline(nn.Module):
    """Gated dilated TCN baseline — 6 blocks, num_filters=64."""
    def __init__(self, input_channels=3, output_channels=12,
                 num_filters=64, kernel_size=3, num_layers=6):
        super().__init__()
        self.in_proj  = nn.Conv1d(input_channels, num_filters, 1)
        self.tcn      = nn.ModuleList([
            _GatedTCNBlock(num_filters, kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        self.out_proj = nn.Conv1d(num_filters, output_channels, 1)

    def forward(self, x):
        z = self.in_proj(x.permute(0, 2, 1))
        for blk in self.tcn:
            z = blk(z)
        return self.out_proj(z).permute(0, 2, 1)


# ── 3. ECG-Transformer ────────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class _SeqLN(nn.Module):
    """LayerNorm applied along the channel dimension of (B, C, T) tensors."""
    def __init__(self, ch): super().__init__(); self.ln = nn.LayerNorm(ch)
    def forward(self, x): return self.ln(x.transpose(1,2)).transpose(1,2)


class ECGTransformer(nn.Module):
    """
    ECG-Transformer baseline.
    Conv stem (stride 2×2) → Transformer encoder → bilinear upsample.
    """
    def __init__(self, input_size=3, output_size=12,
                 d_model=128, nhead=4, num_layers=3,
                 dim_ff=256, dropout=0.1):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(input_size, d_model // 2, 7, padding=3, stride=2),
            _SeqLN(d_model // 2), nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, 7, padding=3, stride=2),
            _SeqLN(d_model), nn.GELU(),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pos      = _PositionalEncoding(d_model)
        self.proj     = nn.Linear(d_model, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        T  = x.size(1)
        z  = self.pre(x.transpose(1, 2))
        z  = self.pos(z.transpose(1, 2))
        z  = self.encoder(z)
        z  = self.proj(z).transpose(1, 2)
        z  = F.interpolate(z, size=T, mode="linear", align_corners=False)
        return z.transpose(1, 2)


# ── 4. UNet++ ─────────────────────────────────────────────────────────────────

def _crop(x, ref):
    if x.size(-1) == ref.size(-1): return x
    if x.size(-1) >  ref.size(-1): return x[..., :ref.size(-1)]
    return F.pad(x, (0, ref.size(-1) - x.size(-1)))


class _CB(nn.Module):
    """Conv1d-BN-GELU double block."""
    def __init__(self, i, o, k=3):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(i, o, k, padding=p), nn.BatchNorm1d(o), nn.GELU(),
            nn.Conv1d(o, o, k, padding=p), nn.BatchNorm1d(o), nn.GELU())
    def forward(self, x): return self.net(x)


class UNetPlusPlus(nn.Module):
    """1-D UNet++ with nested skip connections — 4 encoder depths."""
    def __init__(self, in_channels=3, out_channels=12, base=64,
                 input_size=None, output_size=None):
        super().__init__()
        if in_channels  is None: in_channels  = input_size
        if out_channels is None: out_channels = output_size
        ch = [base, base*2, base*4, base*8]
        self.x00 = _CB(in_channels, ch[0])
        self.x10 = _CB(ch[0], ch[1])
        self.x20 = _CB(ch[1], ch[2])
        self.x30 = _CB(ch[2], ch[3])
        self.pool = nn.MaxPool1d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.x01  = _CB(ch[0]+ch[1], ch[0])
        self.x11  = _CB(ch[1]+ch[2], ch[1])
        self.x21  = _CB(ch[2]+ch[3], ch[2])
        self.x02  = _CB(ch[0]+ch[0]+ch[1], ch[0])
        self.x12  = _CB(ch[1]+ch[1]+ch[2], ch[1])
        self.x03  = _CB(ch[0]*3+ch[1], ch[0])
        self.head = nn.Conv1d(ch[0], out_channels, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _up(self, x, ref): return _crop(self.up(x), ref)

    def forward(self, x):
        x = x.transpose(1, 2)
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x01 = self.x01(torch.cat([x00, self._up(x10, x00)], 1))
        x11 = self.x11(torch.cat([x10, self._up(x20, x10)], 1))
        x21 = self.x21(torch.cat([x20, self._up(x30, x20)], 1))
        x02 = self.x02(torch.cat([x00, x01, self._up(x11, x00)], 1))
        x12 = self.x12(torch.cat([x10, x11, self._up(x21, x10)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self._up(x12, x00)], 1))
        return self.head(x03).transpose(1, 2)


# ── 5. AttentionUNet ──────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class _AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, 1, bias=True), nn.BatchNorm1d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, 1, bias=True), nn.BatchNorm1d(F_int))
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, 1, bias=True), nn.BatchNorm1d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if x.shape[-1] != g.shape[-1]:
            x = F.interpolate(x, size=g.shape[-1], mode="linear", align_corners=False)
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class AttentionUNet(nn.Module):
    """Attention U-Net baseline."""
    def __init__(self, input_size=3, output_size=12):
        super().__init__()
        self.encoder1   = _ConvBlock(input_size, 64)
        self.encoder2   = _ConvBlock(64, 128)
        self.encoder3   = _ConvBlock(128, 256)
        self.encoder4   = _ConvBlock(256, 512)
        self.attention1 = _AttentionBlock(512, 256, 256)
        self.attention2 = _AttentionBlock(256, 128, 128)
        self.attention3 = _AttentionBlock(128, 64, 64)
        self.decoder1   = _ConvBlock(512 + 256, 256)
        self.decoder2   = _ConvBlock(256 + 128, 128)
        self.decoder3   = _ConvBlock(128 + 64, 64)
        self.final_conv = nn.Conv1d(64, output_size, 1)

    def _up(self, x, scale=2):
        return F.interpolate(x, scale_factor=scale,
                             mode="linear", align_corners=False)

    def forward(self, x):
        x  = x.permute(0, 2, 1)
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool1d(e1, 2))
        e3 = self.encoder3(F.max_pool1d(e2, 2))
        e4 = self.encoder4(F.max_pool1d(e3, 2))
        d1 = self.decoder1(torch.cat([self._up(e4), self.attention1(self._up(e4), e3)], 1))
        d2 = self.decoder2(torch.cat([self._up(d1), self.attention2(self._up(d1), e2)], 1))
        d3 = self.decoder3(torch.cat([self._up(d2), self.attention3(self._up(d2), e1)], 1))
        return self.final_conv(d3).permute(0, 2, 1)
