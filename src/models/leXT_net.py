# -*- coding: utf-8 -*-
"""
leXT_net.py  —  LeXT-Net (Lead eXpert Trunk Network)
=====================================================
Self-contained model definition with no external dependencies.

Architecture:
  [P1] TCNTrunk1DEx        — Shared dilated TCN backbone (feat=256, L=8, k=9)
  [H1] BatchedTemporalGate — Single-kernel temporal attention for all 12 leads
  [H2] BatchedProjection   — Grouped conv 256→64 for all leads simultaneously
  [P3] UNetHead1DEx ×12   — Limb leads: depth=1  /  Chest leads: depth=3

Input : (B, T, 3)
Output: (B, T, 12)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────
LIMB_LEADS  = [0, 1, 2, 3, 4, 5]    # I, II, III, aVR, aVL, aVF
CHEST_LEADS = [6, 7, 8, 9, 10, 11]  # V1–V6
N_LEADS     = 12

# ── Helpers ───────────────────────────────────────────────────────────────────

def _kaiming(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def match_TEx(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    tx, tr = x.size(-1), ref.size(-1)
    if tx == tr:
        return x
    return x[..., :tr] if tx > tr else F.pad(x, (0, tr - tx))


# ── [P1] TCN Backbone ─────────────────────────────────────────────────────────

class TCNResBlock1DEx(nn.Module):
    """Dilated residual block: Conv→GN→GELU→Drop ×2 + skip connection."""
    def __init__(self, ch: int, k: int = 9, dilation: int = 1,
                 pdrop: float = 0.1, groups: int = 8):
        super().__init__()
        pad        = (k - 1) // 2 * dilation
        g          = min(groups, ch)
        self.conv1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)
        self.gn1   = nn.GroupNorm(g, ch)
        self.conv2 = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)
        self.gn2   = nn.GroupNorm(g, ch)
        self.drop  = nn.Dropout(pdrop)

    def forward(self, x):
        h = self.drop(F.gelu(self.gn1(self.conv1(x))))
        h = self.drop(F.gelu(self.gn2(self.conv2(h))))
        return x + h


class TCNTrunk1DEx(nn.Module):
    """
    Shared TCN backbone.
    Input : (B, T, C_in)
    Output: (B, feat_ch, T)
    """
    def __init__(self, in_ch: int = 3, feat_ch: int = 256,
                 tcn_layers: int = 8, k: int = 9,
                 dilations=None, pdrop: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_ch, feat_ch, kernel_size=1),
            nn.GroupNorm(min(8, feat_ch), feat_ch),
            nn.GELU(),
        )
        dilations = (list(dilations)[:tcn_layers] if dilations is not None
                     else [2**i for i in range(tcn_layers)])
        self.tcn      = nn.Sequential(*[
            TCNResBlock1DEx(feat_ch, k=k, dilation=d, pdrop=pdrop)
            for d in dilations
        ])
        self.out_norm = nn.GroupNorm(min(8, feat_ch), feat_ch)
        self.apply(_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x.permute(0, 2, 1))
        h = self.tcn(h)
        return F.gelu(self.out_norm(h))


# ── [H1] Batched Temporal Attention Gate ─────────────────────────────────────

class BatchedTemporalGate(nn.Module):
    """Temporal attention gate for all 12 leads — single grouped-conv call."""
    def __init__(self, feat_ch: int, n_leads: int = 12, win: int = 9):
        super().__init__()
        self.feat_ch = feat_ch
        self.n_leads = n_leads
        total_ch     = feat_ch * n_leads
        pad          = win // 2

        self.q_dw = nn.Conv1d(total_ch, total_ch, win,
                              padding=pad, groups=total_ch, bias=False)
        self.k_dw = nn.Conv1d(total_ch, total_ch, win,
                              padding=pad, groups=total_ch, bias=False)
        self.v_pw = nn.Conv1d(total_ch, total_ch, 1,
                              groups=n_leads, bias=False)
        self.norm  = nn.GroupNorm(min(8 * n_leads, total_ch), total_ch)
        self.scale = math.sqrt(feat_ch)

        nn.init.zeros_(self.q_dw.weight)
        nn.init.zeros_(self.k_dw.weight)
        with torch.no_grad():
            w = self.v_pw.weight.squeeze(-1)
            w.zero_()
            for l in range(n_leads):
                s = l * feat_ch
                w[s:s + feat_ch].fill_diagonal_(1.0)

    def forward(self, Fmap: torch.Tensor) -> torch.Tensor:
        B, C, T = Fmap.shape
        L       = self.n_leads
        Ft = Fmap.unsqueeze(1).expand(-1, L, -1, -1).reshape(B, L * C, T)
        Q     = self.q_dw(Ft)
        K     = self.k_dw(Ft)
        V     = self.v_pw(self.norm(Ft))
        Q_r   = Q.view(B, L, C, T)
        K_r   = K.view(B, L, C, T)
        score = (Q_r * K_r).sum(dim=2, keepdim=True) / self.scale
        w     = torch.sigmoid(score)
        V_r   = V.view(B, L, C, T)
        Ft_r  = Ft.view(B, L, C, T)
        return Ft_r + V_r * w


# ── [H2] U-Net Decoder Heads ─────────────────────────────────────────────────

class ConvGNGeLUEx(nn.Module):
    """Double Conv1d → GroupNorm → GELU block."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3,
                 groups: int = 8, pdrop: float = 0.0):
        super().__init__()
        pad = (k - 1) // 2
        g   = min(groups, out_ch)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad),
            nn.GroupNorm(g, out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, k, padding=pad),
            nn.GroupNorm(g, out_ch), nn.GELU(),
            nn.Dropout(pdrop),
        )
    def forward(self, x): return self.net(x)


class UNetHead1DEx(nn.Module):
    """
    Per-lead U-Net decoder head.
    Input : (B, base, T)
    Output: (B, 1, T)
    """
    def __init__(self, base: int = 64, depth: int = 2, pdrop: float = 0.0):
        super().__init__()
        assert depth in (1, 2, 3)
        self.depth = depth
        self.pool  = nn.MaxPool1d(2)
        self.up    = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        self.bott0 = ConvGNGeLUEx(base, base, pdrop=pdrop)
        self.dec0  = ConvGNGeLUEx(base, base, pdrop=pdrop)

        if depth >= 2:
            self.enc1  = ConvGNGeLUEx(base, base * 2, pdrop=pdrop)
            self.bott1 = ConvGNGeLUEx(base * 2, base * 2, pdrop=pdrop)
            self.dec1  = ConvGNGeLUEx(base * 2 + base, base, pdrop=pdrop)

        if depth >= 3:
            self.enc2  = ConvGNGeLUEx(base * 2, base * 4, pdrop=pdrop)
            self.bott2 = ConvGNGeLUEx(base * 4, base * 4, pdrop=pdrop)
            self.dec2a = ConvGNGeLUEx(base * 4 + base * 2, base * 2, pdrop=pdrop)
            self.dec2b = ConvGNGeLUEx(base * 2 + base,     base,     pdrop=pdrop)

        self.out = nn.Conv1d(base, 1, kernel_size=1)
        self.apply(_kaiming)

    def forward(self, e0: torch.Tensor) -> torch.Tensor:
        if self.depth == 1:
            return self.out(self.dec0(self.bott0(e0)))
        if self.depth == 2:
            e1 = self.enc1(self.pool(e0))
            b  = self.bott1(e1)
            d  = match_TEx(self.up(b), e0)
            d  = self.dec1(torch.cat([d, e0], 1))
            return self.out(d)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        b  = self.bott2(e2)
        d1 = match_TEx(self.up(b),  e1)
        d1 = self.dec2a(torch.cat([d1, e1], 1))
        d0 = match_TEx(self.up(d1), e0)
        d0 = self.dec2b(torch.cat([d0, e0], 1))
        return self.out(d0)


class BatchedProjection(nn.Module):
    """feat_ch → 12×base channel reduction — single grouped Conv1d call."""
    def __init__(self, feat_ch: int, base: int, n_leads: int = 12,
                 k: int = 3, pdrop: float = 0.0):
        super().__init__()
        in_ch  = feat_ch * n_leads
        out_ch = base    * n_leads
        pad    = (k - 1) // 2
        g      = n_leads
        self.n_leads = n_leads
        self.base    = base
        self.feat_ch = feat_ch
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, groups=g)
        self.gn1   = nn.GroupNorm(min(8 * g, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, groups=g)
        self.gn2   = nn.GroupNorm(min(8 * g, out_ch), out_ch)
        self.drop  = nn.Dropout(pdrop)
        self.apply(_kaiming)

    def forward(self, gated: torch.Tensor) -> torch.Tensor:
        B, L, C, T = gated.shape
        x = gated.reshape(B, L * C, T)
        h = self.drop(F.gelu(self.gn1(self.conv1(x))))
        h = self.drop(F.gelu(self.gn2(self.conv2(h))))
        return h.reshape(B, L, self.base, T)


# ── Main Model ────────────────────────────────────────────────────────────────

class TCNTrunk12UNetHeadsExtended(nn.Module):
    """
    LeXT-Net — Lead eXpert Trunk Network
    Input : (B, T, 3)
    Output: (B, T, 12)
    """
    def __init__(
        self,
        input_size:         int   = 3,
        output_size:        int   = 12,
        trunk_feat_ch:      int   = 256,
        trunk_layers:       int   = 8,
        trunk_k:            int   = 9,
        trunk_pdrop:        float = 0.1,
        trunk_dilations           = None,
        use_temporal_gate:  bool  = True,
        gate_win:           int   = 9,
        head_base:          int   = 64,
        head_pdrop:         float = 0.0,
        limb_depth:         int   = 1,
        chest_depth:        int   = 3,
        uniform_depth:      bool  = False,
        uniform_depth_val:  int   = 2,
    ):
        super().__init__()
        assert output_size == N_LEADS
        self.use_temporal_gate = use_temporal_gate
        self.n_leads           = output_size
        self.trunk_feat_ch     = trunk_feat_ch
        self.head_base         = head_base

        self.trunk = TCNTrunk1DEx(
            in_ch=input_size, feat_ch=trunk_feat_ch,
            tcn_layers=trunk_layers, k=trunk_k,
            dilations=trunk_dilations, pdrop=trunk_pdrop)

        if use_temporal_gate:
            self.gate = BatchedTemporalGate(
                trunk_feat_ch, n_leads=output_size, win=gate_win)

        self.proj = BatchedProjection(
            trunk_feat_ch, head_base, n_leads=output_size, pdrop=head_pdrop)

        def _d(i): return (uniform_depth_val if uniform_depth
                           else (limb_depth if i in LIMB_LEADS else chest_depth))
        self.heads = nn.ModuleList([
            UNetHead1DEx(base=head_base, depth=_d(i), pdrop=head_pdrop)
            for i in range(output_size)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Fmap = self.trunk(x)
        if self.use_temporal_gate:
            gated = self.gate(Fmap)
        else:
            gated = Fmap.unsqueeze(1).expand(-1, self.n_leads, -1, -1)
        e0 = self.proj(gated)
        ys = [self.heads[i](e0[:, i]) for i in range(self.n_leads)]
        return torch.stack(ys, dim=2).squeeze(1).permute(0, 2, 1)

    def param_summary(self):
        trunk_p = sum(p.numel() for p in self.trunk.parameters())
        gate_p  = (sum(p.numel() for p in self.gate.parameters())
                   if self.use_temporal_gate else 0)
        proj_p  = sum(p.numel() for p in self.proj.parameters())
        head_p  = sum(p.numel() for h in self.heads for p in h.parameters())
        total   = trunk_p + gate_p + proj_p + head_p
        print(f"  Trunk      : {trunk_p:>10,}")
        print(f"  Gate       : {gate_p:>10,}")
        print(f"  Projection : {proj_p:>10,}")
        print(f"  Heads      : {head_p:>10,}")
        print(f"  TOTAL      : {total:>10,}")


# ── Loss Functions ────────────────────────────────────────────────────────────

class InterLeadConsistencyLoss(nn.Module):
    """Einthoven + Goldberger physiological constraints."""
    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight

    def forward(self, y: torch.Tensor, ramp: float = 1.0) -> torch.Tensor:
        I, II, III     = y[:, :, 0], y[:, :, 1], y[:, :, 2]
        aVR, aVL, aVF  = y[:, :, 3], y[:, :, 4], y[:, :, 5]
        r1 = III - (II - I)
        r2 = aVR + (I + II) * 0.5
        r3 = aVL - (I - III) * 0.5
        r4 = aVF - (II + III) * 0.5
        loss = (r1.pow(2).mean() + r2.pow(2).mean() +
                r3.pow(2).mean() + r4.pow(2).mean()) * 0.25
        return self.weight * ramp * loss


class ChestSmoothLoss(nn.Module):
    """Precordial lead transition smoothness (V1→V6)."""
    def __init__(self, weight: float = 0.02):
        super().__init__()
        self.weight = weight

    def forward(self, y: torch.Tensor, ramp: float = 1.0) -> torch.Tensor:
        diff = y[:, :, 7:12] - y[:, :, 6:11]
        return self.weight * ramp * diff.pow(2).mean()


class CombinedLoss(nn.Module):
    """MAE + PCC + Physics constraint + Chest smoothness."""
    def __init__(self, w_mae=0.70, w_pcc=0.25, w_cons=0.05,
                 w_smooth=0.02, ramp_epochs=20, eps=1e-8):
        super().__init__()
        self.w_mae       = w_mae
        self.w_pcc       = w_pcc
        self.cons        = InterLeadConsistencyLoss(weight=w_cons)
        self.smooth      = ChestSmoothLoss(weight=w_smooth)
        self.ramp_epochs = max(ramp_epochs, 1)
        self.eps         = eps

    @staticmethod
    def _mae(yp, yt):
        return (yp - yt).abs().mean()

    def _pcc(self, yp, yt):
        yp_ = yp - yp.mean(1, keepdim=True)
        yt_ = yt - yt.mean(1, keepdim=True)
        num = (yp_ * yt_).sum(1)
        den = (yp_.pow(2).sum(1) * yt_.pow(2).sum(1) + self.eps).sqrt()
        return 1.0 - (num / den).mean()

    def forward(self, yp, yt, epoch=0):
        ramp = float(min(epoch / self.ramp_epochs, 1.0))
        return (self.w_mae * self._mae(yp, yt)
                + self.w_pcc * self._pcc(yp, yt)
                + self.cons(yp,   ramp=ramp)
                + self.smooth(yp, ramp=ramp))


# ── Batched Head Groups (fast variant) ───────────────────────────────────────

class BatchedHeadGroup(nn.Module):
    """
    Runs N same-depth U-Net heads in a single batched forward pass.

    Replaces the serial loop:
        for i in range(N): ys[i] = heads[i](e0[:, i])   # N kernel launches
    with two grouped-conv calls:
        limb_group(e0[:, :6])   # 1 launch
        chest_group(e0[:, 6:])  # 1 launch

    Each head retains independent weights via grouped convolutions.
    """
    def __init__(self, n_heads: int, base: int = 64,
                 depth: int = 1, pdrop: float = 0.0):
        super().__init__()
        assert depth in (1, 2, 3)
        self.n_heads = n_heads
        self.depth   = depth
        self.base    = base

        self.pool = nn.MaxPool1d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        def grp_conv(in_c, out_c, k=3):
            pad = (k - 1) // 2
            return nn.Conv1d(in_c * n_heads, out_c * n_heads, k,
                             padding=pad, groups=n_heads)

        def grp_gn(out_c):
            total = out_c * n_heads
            return nn.GroupNorm(min(8 * n_heads, total), total)

        def grp_block(in_c, out_c, k=3):
            return nn.Sequential(
                grp_conv(in_c, out_c, k), grp_gn(out_c), nn.GELU(),
                grp_conv(out_c, out_c, k), grp_gn(out_c), nn.GELU(),
                nn.Dropout(pdrop),
            )

        self.bott0 = grp_block(base, base)
        self.dec0  = grp_block(base, base)
        if depth >= 2:
            self.enc1  = grp_block(base, base * 2)
            self.bott1 = grp_block(base * 2, base * 2)
            self.dec1  = grp_block(base * 2 + base, base)
        if depth >= 3:
            self.enc2  = grp_block(base * 2, base * 4)
            self.bott2 = grp_block(base * 4, base * 4)
            self.dec2a = grp_block(base * 4 + base * 2, base * 2)
            self.dec2b = grp_block(base * 2 + base, base)

        self.out = nn.Conv1d(base * n_heads, n_heads, 1, groups=n_heads)
        self.apply(_kaiming)

    def _up(self, x, ref):
        u = self.up(x)
        if u.size(-1) > ref.size(-1): return u[..., :ref.size(-1)]
        if u.size(-1) < ref.size(-1): return F.pad(u, (0, ref.size(-1)-u.size(-1)))
        return u

    def forward(self, e0_group: torch.Tensor) -> torch.Tensor:
        """e0_group: (B, N, base, T)  →  (B, N, T)"""
        B, N, C, T = e0_group.shape
        x = e0_group.reshape(B, N * C, T)

        if self.depth == 1:
            return self.out(self.dec0(self.bott0(x)))
        if self.depth == 2:
            e1 = self.enc1(self._up(x, x) if False else self.pool(x))
            b  = self.bott1(e1)
            d  = self._up(b, x)
            d  = self.dec1(torch.cat([d, x], dim=1))
            return self.out(d)
        e1 = self.enc1(self.pool(x))
        e2 = self.enc2(self.pool(e1))
        b  = self.bott2(e2)
        d1 = self._up(b,  e1)
        d1 = self.dec2a(torch.cat([d1, e1], dim=1))
        d0 = self._up(d1, x)
        d0 = self.dec2b(torch.cat([d0, x], dim=1))
        return self.out(d0)


class TCNTrunk12UNetHeadsExtendedFast(nn.Module):
    """
    LeXT-Net — fast variant using BatchedHeadGroup.

    Identical architecture to TCNTrunk12UNetHeadsExtended but replaces the
    12-iteration serial head loop with two batched group calls:
        limb_heads(e0[:, :6])   # 1 CUDA kernel launch
        chest_heads(e0[:, 6:])  # 1 CUDA kernel launch

    Note: weights are not compatible with TCNTrunk12UNetHeadsExtended
    checkpoints due to the different grouped-conv layout.
    Use this class when training from scratch.
    """
    def __init__(
        self,
        input_size:         int   = 3,
        output_size:        int   = 12,
        trunk_feat_ch:      int   = 256,
        trunk_layers:       int   = 8,
        trunk_k:            int   = 9,
        trunk_pdrop:        float = 0.1,
        trunk_dilations           = None,
        use_temporal_gate:  bool  = True,
        gate_win:           int   = 9,
        head_base:          int   = 64,
        head_pdrop:         float = 0.0,
        limb_depth:         int   = 1,
        chest_depth:        int   = 3,
        uniform_depth:      bool  = False,
        uniform_depth_val:  int   = 2,
    ):
        super().__init__()
        assert output_size == N_LEADS
        self.use_temporal_gate = use_temporal_gate
        self.n_leads           = output_size

        self.trunk = TCNTrunk1DEx(
            in_ch=input_size, feat_ch=trunk_feat_ch,
            tcn_layers=trunk_layers, k=trunk_k,
            dilations=trunk_dilations, pdrop=trunk_pdrop)

        if use_temporal_gate:
            self.gate = BatchedTemporalGate(
                trunk_feat_ch, n_leads=output_size, win=gate_win)

        self.proj = BatchedProjection(
            trunk_feat_ch, head_base, n_leads=output_size, pdrop=head_pdrop)

        d = uniform_depth_val if uniform_depth else None
        self.limb_heads  = BatchedHeadGroup(
            6, base=head_base, depth=d or limb_depth,  pdrop=head_pdrop)
        self.chest_heads = BatchedHeadGroup(
            6, base=head_base, depth=d or chest_depth, pdrop=head_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Fmap = self.trunk(x)
        if self.use_temporal_gate:
            gated = self.gate(Fmap)
        else:
            gated = Fmap.unsqueeze(1).expand(-1, self.n_leads, -1, -1)
        e0        = self.proj(gated)
        limb_out  = self.limb_heads( e0[:, :6])
        chest_out = self.chest_heads(e0[:, 6:])
        return torch.cat([limb_out, chest_out], dim=1).permute(0, 2, 1)

    def param_summary(self):
        trunk_p = sum(p.numel() for p in self.trunk.parameters())
        gate_p  = (sum(p.numel() for p in self.gate.parameters())
                   if self.use_temporal_gate else 0)
        proj_p  = sum(p.numel() for p in self.proj.parameters())
        limb_p  = sum(p.numel() for p in self.limb_heads.parameters())
        chest_p = sum(p.numel() for p in self.chest_heads.parameters())
        total   = trunk_p + gate_p + proj_p + limb_p + chest_p
        print(f"  Trunk       : {trunk_p:>10,}")
        print(f"  Gate        : {gate_p:>10,}")
        print(f"  Projection  : {proj_p:>10,}")
        print(f"  Limb heads  : {limb_p:>10,}")
        print(f"  Chest heads : {chest_p:>10,}")
        print(f"  TOTAL       : {total:>10,}")


# ── LeXT-Net v1 alias ─────────────────────────────────────────────────────────
# Used in train_ptbxl.py for leXT-netv1 (no gate, uniform depth=2).

class TCNTrunk12UNetHeads(TCNTrunk12UNetHeadsExtended):
    """
    LeXT-Net v1 — uniform head depth, no temporal gate.
    Alias of TCNTrunk12UNetHeadsExtended(uniform_depth=True, use_temporal_gate=False).
    """
    def __init__(self, input_size=3, output_size=12,
                 trunk_feat_ch=256, trunk_layers=8, trunk_k=9,
                 trunk_pdrop=0.1, head_base=64, head_pdrop=0.0,
                 limb_depth=2, chest_depth=2, **kwargs):
        super().__init__(
            input_size=input_size, output_size=output_size,
            trunk_feat_ch=trunk_feat_ch, trunk_layers=trunk_layers,
            trunk_k=trunk_k, trunk_pdrop=trunk_pdrop,
            use_temporal_gate=False,
            head_base=head_base, head_pdrop=head_pdrop,
            uniform_depth=True, uniform_depth_val=2,
        )
